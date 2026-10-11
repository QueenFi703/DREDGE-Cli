"""Run inside the dedicated image as root, with --network none and no secrets.

Mount this file read-only and override the entrypoint with /opt/venv/bin/python.
This uses real bootstrap helpers, identities, app/health processes and UDS HTTP.
Only Cloudflare readiness is synthetic; no connector, account, grant, or deployment
is created. Full OAuth/PKCE and supervisor lifecycle tests run separately in CI.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import importlib.util
import os
from pathlib import Path
import socket
import sqlite3
import stat
import subprocess
import sys
import tempfile
import time

import httpx

ISSUER = "https://dredge.example.test"
ORIGIN = "http://dredge.example.test"
PORT = 8123
FIXTURES = {
    "DREDGE_CHATGPT_TUNNEL_ENABLED": "true",
    "DREDGE_CHATGPT_ENABLED": "true",
    "CHATGPT_ISSUER": ISSUER,
    "CHATGPT_RESOURCE": ISSUER + "/chatgpt/mcp",
    "CHATGPT_CLIENT_ID": "fictional-container-client",
    "CHATGPT_REDIRECT_URI": "https://chatgpt.com/connector/oauth/fictional-container",
    "SECRET_KEY": "fictional-container-session-key-at-least-32-bytes",
    "STUDIO_DB_PATH": "/data/studio/container-check.sqlite3",
    "TUNNEL_TOKEN": "fictional-container-token-never-connect",
    "PORT": str(PORT),
}
DB_MARKER = "fictional-container-private-database-marker"
PROBE_MARKER = "fictional-container-probe-must-not-be-logged"
SECRET_FILE = Path("/data/studio/container-app-secret")

# This process is the ONLY readiness stand-in. It never starts cloudflared and
# returns 200 for the same fixed local /ready check used by the production probe.
READY_STUB = r'''
from http.server import BaseHTTPRequestHandler, HTTPServer
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        self.send_response(200 if self.path == '/ready' else 404)
        self.send_header('Content-Length', '0')
        self.end_headers()
with HTTPServer(('127.0.0.1', 2000), Handler) as server:
    server.serve_forever()
'''

# Permission assertions run after actual Linux UID/GID/supplementary-group drops.
# A missing file or any unrelated error must fail rather than count as a denial.
PERMISSION_CHECK = r'''
import errno, os, socket, sys
from pathlib import Path
operation, target = sys.argv[1:3]
def denied(action):
    try:
        result = action()
    except OSError as error:
        assert error.errno in (errno.EACCES, errno.EPERM), 'Unexpected permission error'
    else:
        if isinstance(result, int): os.close(result)
        raise AssertionError('Forbidden action succeeded')
if operation == 'deny-read':
    denied(lambda: os.open(target, os.O_RDONLY))
elif operation == 'deny-write':
    denied(lambda: os.open(target, os.O_WRONLY))
elif operation == 'deny-unlink':
    denied(lambda: os.unlink(target))
elif operation == 'deny-create':
    denied(lambda: os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
elif operation == 'read-token':
    assert Path(target).read_text() == sys.argv[3], 'Wrong synthetic token'
elif operation == 'connect':
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(3)
        client.connect(target)
        client.sendall(b'GET /health HTTP/1.1\r\nHost: dredge.example.test\r\n'
                       b'X-Forwarded-Proto: https\r\nConnection: close\r\n\r\n')
        assert client.recv(4096).startswith(b'HTTP/1.1 200 '), 'UDS group access failed'
elif operation == 'deny-connect':
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(3)
        denied(lambda: client.connect(target))
else:
    raise AssertionError('Unknown permission check')
'''


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def load_supervisor():
    # Load exactly the root bootstrap file used by the image entrypoint.
    path = "/app/src/dredge/chatgpt_tunnel_supervisor.py"
    spec = importlib.util.spec_from_file_location("container_supervisor", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def file_identity(path, uid, gid, mode):
    info = path.lstat()
    require((info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode)) == (uid, gid, mode),
            "Unexpected owner/group/mode: " + str(path))
    require(not stat.S_ISLNK(info.st_mode), "Unexpected symlink: " + str(path))


def run_as(supervisor, spec, script, *args, groups=None):
    result = subprocess.run(
        [supervisor.PYTHON, "-I", "-c", script, *map(str, args)],
        cwd=supervisor.APP_DIR, env=supervisor.BASE_ENV,
        user=spec.uid, group=spec.gid,
        extra_groups=spec.extra_groups if groups is None else groups,
        stdin=subprocess.DEVNULL, capture_output=True, timeout=10,
        start_new_session=True, umask=0o077,
    )
    # Never print raw subprocess output; it could contain a failed secret check.
    require(result.returncode == 0, "Identity/permission subprocess failed: " + str(args[0]))


@contextmanager
def capture_child_output(log):
    sys.stdout.flush()
    sys.stderr.flush()
    saved = [os.dup(fd) for fd in (1, 2)]
    try:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        yield
    finally:
        for fd, original in zip((1, 2), saved):
            os.dup2(original, fd)
            os.close(original)


def process_identity(process, spec):
    lines = Path(f"/proc/{process.pid}/status").read_text().splitlines()
    fields = dict(line.split(":", 1) for line in lines if ":" in line)
    require([int(value) for value in fields["Uid"].split()] == [spec.uid] * 4,
            "Child UID did not drop")
    require([int(value) for value in fields["Gid"].split()] == [spec.gid] * 4,
            "Child GID did not drop")
    require([int(value) for value in fields["Groups"].split()] == list(spec.extra_groups),
            "Child inherited unexpected supplementary groups")


def tcp_listeners():
    listeners = set()
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        for line in Path(table).read_text().splitlines()[1:]:
            fields = line.split()
            if fields[3] == "0A":  # TCP_LISTEN
                address, port = fields[1].split(":")
                listeners.add((address, int(port, 16)))
    return listeners


def wait_status(client, url, status, children, supervisor, **kwargs):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        require(not any(supervisor._has_exited(child) for child in children),
                "A runtime child exited before readiness")
        try:
            response = client.get(url, **kwargs)
            if response.status_code == status:
                return response
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    raise AssertionError("Timed out waiting for the expected local readiness state")


def check_cloudflared(supervisor, connector):
    output = []
    for args in (("--version",), ("tunnel", "run", "--help"), ("tunnel", "--help")):
        result = subprocess.run(
            [supervisor.CLOUDFLARED, *args], env=connector.environment,
            cwd=supervisor.APP_DIR, user=connector.uid, group=connector.gid,
            extra_groups=connector.extra_groups, stdin=subprocess.DEVNULL,
            capture_output=True, timeout=15, text=True,
        )
        require(result.returncode == 0, "Official cloudflared CLI inspection failed")
        output.append(result.stdout + result.stderr)
    require("2026.10.0" in output[0], "Unexpected cloudflared version")
    require("--token-file" in output[1], "cloudflared run has no token-file flag")
    help_text = "\n".join(output[1:])
    for flag in ("--no-autoupdate", "--metrics", "--management-diagnostics", "--grace-period"):
        require(flag in help_text, "Missing configured cloudflared flag: " + flag)
    print("PASS: official cloudflared version/help flags (no tunnel connection)", flush=True)


def check_http(uds):
    headers = {"X-Forwarded-Proto": "https"}
    for path in ("/.well-known/oauth-authorization-server",
                 "/.well-known/oauth-protected-resource/chatgpt/mcp"):
        response = uds.get(ORIGIN + path, headers=headers)
        require(response.status_code == 200, "Discovery metadata unavailable over UDS")
        require(ISSUER in response.text, "Discovery lost its canonical HTTPS origin")
    for extra in ({}, {"Authorization": "Bearer " + PROBE_MARKER}):
        response = uds.post(ORIGIN + "/chatgpt/mcp", json={}, headers={**headers, **extra})
        require(response.status_code == 401, "MCP accepted missing or invalid authorization")
    for wrong in ({}, {"X-Forwarded-Proto": "http"},
                  {"X-Forwarded-Proto": "https", "Host": "wrong.example.test"},
                  [("X-Forwarded-Proto", "https"), ("X-Forwarded-Proto", "https")]):
        require(uds.get(ORIGIN + "/health", headers=wrong).status_code == 400,
                "Socket ingress accepted invalid or duplicate origin headers")
    for path in ("/ready", "/metrics", "/mcp", "/billing", "/api/studio/runs",
                 "/api/studio/report", "/docs"):
        require(uds.get(ORIGIN + path, headers=headers).status_code == 404,
                "Unexpected application route became reachable")
    require(uds.get(ORIGIN + "/auth/login", headers=headers).status_code == 200,
            "Dedicated login page unavailable")
    require(uds.post(ORIGIN + "/chatgpt/oauth/token", headers=headers,
                     data={"grant_type": "authorization_code", "code": PROBE_MARKER}).status_code == 400,
            "OAuth accepted a fabricated authorization code")


def main():
    require(os.geteuid() == 0, "Run only as root inside the isolated dedicated image")
    require({name for _, name in socket.if_nameindex()} == {"lo"},
            "Container must use --network none")
    os.environ.clear()  # Always use fictional constants, never the runner's secrets.
    os.umask(0o077)
    supervisor = load_supervisor()
    supervisor.harden_supervisor()
    config = supervisor.read_configuration(FIXTURES)
    supervisor.prepare_directories()
    for path, uid, gid, mode in (
        (supervisor.RUN_DIR, 0, 2000, 0o750),
        (supervisor.TOKEN_DIR, 0, 1001, 0o750),
        (supervisor.DATA_DIR, 1000, 1000, 0o700),
        (supervisor.DATA_DIR / "tmp", 1000, 1000, 0o700),
    ):
        file_identity(path, uid, gid, mode)
    db_path = Path(FIXTURES["STUDIO_DB_PATH"])
    require(not any(path.exists() or path.is_symlink() for path in
                    (db_path, SECRET_FILE, supervisor.TOKEN_PATH, supervisor.SOCKET_PATH)),
            "Use a fresh disposable container without mounted application data")
    children, listener, lock_fd = [], None, None
    with tempfile.TemporaryFile(dir=supervisor.RUN_DIR) as log:
        try:
            lock_fd = supervisor.acquire_lock()
            supervisor.write_token_file(supervisor.TOKEN_PATH, config.token)
            file_identity(supervisor.TOKEN_PATH, 1001, 1001, 0o400)
            listener = supervisor.bind_gateway_socket()
            file_identity(supervisor.SOCKET_PATH, 0, 2000, 0o660)
            require(stat.S_ISSOCK(supervisor.SOCKET_PATH.stat().st_mode), "Gateway is not a socket")
            with sqlite3.connect(db_path) as db:
                db.execute("CREATE TABLE container_fixture(value TEXT)")
                db.execute("INSERT INTO container_fixture VALUES(?)", (DB_MARKER,))
            SECRET_FILE.write_text(FIXTURES["SECRET_KEY"])
            for path in (db_path, SECRET_FILE):
                os.chown(path, 1000, 1000)
                os.chmod(path, 0o600)
            app, connector, health = supervisor.child_specs(config, listener.fileno())
            check_cloudflared(supervisor, connector)
            for spec in (app, connector):
                for operation, path in (
                    ("deny-write", Path(supervisor.__file__)),
                    ("deny-unlink", Path(supervisor.__file__)),
                    ("deny-unlink", supervisor.SOCKET_PATH),
                    ("deny-create", supervisor.RUN_DIR / "forbidden-replacement"),
                ):
                    run_as(supervisor, spec, PERMISSION_CHECK, operation, path)
            run_as(supervisor, app, PERMISSION_CHECK, "deny-read", supervisor.TOKEN_PATH)
            run_as(supervisor, connector, PERMISSION_CHECK, "read-token",
                   supervisor.TOKEN_PATH, config.token)
            for path in (db_path, SECRET_FILE):
                run_as(supervisor, connector, PERMISSION_CHECK, "deny-read", path)
            for spec in (app, connector):
                run_as(supervisor, spec, PERMISSION_CHECK, "deny-connect",
                       supervisor.SOCKET_PATH, groups=())
            print("PASS: real bootstrap ownership, 0660 socket and cross-UID secret isolation", flush=True)
            stub = replace(connector, name="synthetic-local-readiness",
                           argv=(supervisor.PYTHON, "-I", "-c", READY_STUB))
            # Start real app/health with their unmodified production ChildSpecs.
            with capture_child_output(log):
                children.append(supervisor.spawn_child(app))
                children.append(supervisor.spawn_child(health))
            with httpx.Client(transport=httpx.HTTPTransport(uds=str(supervisor.SOCKET_PATH)),
                              timeout=2, trust_env=False, follow_redirects=False) as uds, \
                 httpx.Client(timeout=3, trust_env=False, follow_redirects=False) as tcp:
                wait_status(uds, ORIGIN + "/health", 200, children, supervisor,
                            headers={"X-Forwarded-Proto": "https"})
                public_health = f"http://127.0.0.1:{PORT}/health"
                wait_status(tcp, public_health, 503, children, supervisor)
                with capture_child_output(log):
                    children.append(supervisor.spawn_child(stub))
                response = wait_status(tcp, public_health, 200, children, supervisor)
                require(response.content == b"", "Public readiness exposed content")
                for process, spec in zip(children, (app, health, stub)):
                    process_identity(process, spec)
                # Inspect under the matching UID; Docker root need not have CAP_SYS_PTRACE.
                for process, spec in zip(children, (app, health, stub)):
                    run_as(supervisor, spec,
                           "import pathlib,sys; value=pathlib.Path(sys.argv[1]).read_bytes(); "
                           "assert b\"TUNNEL_TOKEN=\" not in value; "
                           "assert sys.argv[2] == \"application\" or sys.argv[3].encode() not in value",
                           f"/proc/{process.pid}/environ", spec.name, FIXTURES["SECRET_KEY"])
                run_as(supervisor, app,
                       "from pathlib import Path; import sys; "
                       "assert str(Path(sys.argv[1]).readlink()) == sys.argv[2]",
                       f"/proc/{children[0].pid}/fd/{listener.fileno()}",
                       f"socket:[{os.fstat(listener.fileno()).st_ino}]")
                for spec in (app, connector):
                    run_as(supervisor, spec, PERMISSION_CHECK, "connect", supervisor.SOCKET_PATH)
                check_http(uds)
                for path in ("/ready", "/metrics", "/chatgpt/mcp", "/auth/login",
                             "/.well-known/oauth-authorization-server",
                             "/health?target=http://127.0.0.1:2000/ready",
                             "/health?path=/chatgpt/mcp"):
                    response = tcp.get(f"http://127.0.0.1:{PORT}" + path,
                                       headers={"X-Forwarded-Proto": "https", "Host": "dredge.example.test"})
                    require(response.status_code == 404 and response.content == b"",
                            "Public readiness accepted a path/probe bypass")
                response = tcp.post(public_health, content=PROBE_MARKER)
                require(response.status_code == 404 and response.content == b"",
                        "Public readiness accepted POST")
                require(tcp_listeners() == {("00000000", PORT), ("0100007F", 2000)},
                        "Unexpected TCP listener or exposed connector metrics")
                # Remove only the synthetic readiness process; real health must fail closed.
                stub_process = children[-1]
                # stop_children owns ALL children and is reserved for final cleanup.
                # This simple stub has no descendants. Keep its PID reserved until signalled.
                stub_process.terminate()
                try:
                    stub_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    stub_process.kill()
                    stub_process.wait(timeout=3)
                children.pop()
                response = wait_status(tcp, public_health, 503, children, supervisor)
                require(response.content == b"", "Failed readiness exposed content")
            print("PASS: real UID1000 UDS app/health, GID2000 access, routes and TCP isolation", flush=True)
        finally:
            supervisor.stop_children(children, 0.5)
            if listener is not None:
                listener.close()
                supervisor.SOCKET_PATH.unlink(missing_ok=True)
            supervisor.TOKEN_PATH.unlink(missing_ok=True)
            if lock_fd is not None:
                os.close(lock_fd)
            log.seek(0)
            output = log.read()
            for marker in (FIXTURES["TUNNEL_TOKEN"], FIXTURES["SECRET_KEY"], DB_MARKER, PROBE_MARKER):
                require(marker.encode() not in output, "Synthetic secret or request value leaked to stdout/stderr")
    require(not tcp_listeners(), "A runtime child left a TCP listener behind")
    require(not supervisor.SOCKET_PATH.exists() and not supervisor.TOKEN_PATH.exists(),
            "Runtime socket/token cleanup failed")
    print("PASS: child cleanup and captured stdout/stderr contain no synthetic secrets", flush=True)
    print("PASS: isolated container gate; provider sign-in and named-tunnel connectivity NOT exercised", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Fixed assertions are useful diagnostics; never dump child logs or HTTP bodies.
        detail = str(error) if isinstance(error, AssertionError) else type(error).__name__
        print("FAIL: " + detail, file=sys.stderr)
        raise SystemExit(1) from None
