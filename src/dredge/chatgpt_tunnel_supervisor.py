"""Opt-in container bootstrap for the dedicated ChatGPT Unix-socket service.

Only this supervisor starts as root. The application and health listener run as
UID 1000; the named-tunnel connector runs as UID 1001. Root-owned code and runtime
directories cannot be replaced by either child. No legacy server is launched.
The remotely managed Cloudflare ingress MUST target unix:/run/dredge/gateway.sock;
there is deliberately no quick-tunnel or TCP-origin fallback.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
import fcntl
import os
from pathlib import Path
import re
import resource
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
from typing import Mapping
from urllib.parse import urlsplit

APP_UID = APP_GID = 1000
TUNNEL_UID = TUNNEL_GID = 1001
SOCKET_GID = 2000
RUN_DIR = Path("/run/dredge")
SOCKET_PATH = RUN_DIR / "gateway.sock"
TOKEN_DIR = RUN_DIR / "cloudflared"
TOKEN_PATH = TOKEN_DIR / "token"
DATA_DIR = Path("/data/studio")
APP_DIR = Path("/app")
PYTHON = "/opt/venv/bin/python"
CLOUDFLARED = "/usr/local/bin/cloudflared"

# Exact allowlist, never a DREDGE_/OAUTH_ prefix wildcard. Live model/payment keys
# are unnecessary for this no-inference service and must not enter its environment.
APP_ENV_KEYS = frozenset({
    "CHATGPT_ISSUER", "CHATGPT_RESOURCE", "CHATGPT_CLIENT_ID", "CHATGPT_REDIRECT_URI",
    "DREDGE_CHATGPT_RUN_TOOLS_ENABLED", "DREDGE_CHATGPT_CONTROLS_ENABLED",
    "SECRET_KEY", "STUDIO_DB_PATH", "STUDIO_ROLES_JSON", "OAUTH_REDIRECT_BASE",
    "GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "GITHUB_CLIENT_ID", "GITHUB_CLIENT_SECRET",
})
BASE_ENV = {
    "PATH": "/opt/venv/bin:/usr/local/bin:/usr/bin:/bin",
    "PYTHONPATH": "/app/src",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUNBUFFERED": "1",
    "PYTHONNOUSERSITE": "1",
    "LANG": "C.UTF-8",
    "HOME": "/nonexistent",
}


class ConfigurationError(RuntimeError):
    """A fixed diagnostic only; never include environment values or tokens."""


@dataclass(frozen=True)
class Configuration:
    app_environment: dict[str, str] = field(repr=False)
    token: str = field(repr=False)
    port: int
    issuer: str


@dataclass(frozen=True)
class ChildSpec:
    name: str
    argv: tuple[str, ...]
    environment: dict[str, str] = field(repr=False)
    uid: int
    gid: int
    extra_groups: tuple[int, ...]
    pass_fds: tuple[int, ...] = ()


def read_configuration(environ: Mapping[str, str]) -> Configuration:
    for flag in ("DREDGE_CHATGPT_TUNNEL_ENABLED", "DREDGE_CHATGPT_ENABLED"):
        if environ.get(flag, "").strip().lower() != "true":
            raise ConfigurationError("Both explicit ChatGPT opt-in flags must be true.")
    runs_flag = environ.get("DREDGE_CHATGPT_RUN_TOOLS_ENABLED", "").strip().lower()
    if runs_flag not in {"", "0", "false", "off", "no", "true"}:
        raise ConfigurationError("DREDGE_CHATGPT_RUN_TOOLS_ENABLED must be true or false.")
    controls_flag = environ.get("DREDGE_CHATGPT_CONTROLS_ENABLED", "").strip().lower()
    if controls_flag not in {"", "0", "false", "off", "no", "true"}:
        raise ConfigurationError("DREDGE_CHATGPT_CONTROLS_ENABLED must be true or false.")
    token = environ.get("TUNNEL_TOKEN", "")
    if not token or len(token) > 16384 or any(c.isspace() or ord(c) < 33 for c in token):
        raise ConfigurationError("A nonempty named-tunnel token is required.")
    port_text = environ.get("PORT", "8000")
    if not re.fullmatch(r"[0-9]{1,5}", port_text):
        raise ConfigurationError("PORT must be an unprivileged TCP port.")
    port = int(port_text)
    if not 1024 <= port <= 65535 or port == 2000:
        raise ConfigurationError("PORT must not conflict with the private metrics listener.")
    issuer = environ.get("CHATGPT_ISSUER", "")
    try:
        parsed = urlsplit(issuer)
        valid = (parsed.scheme == "https" and parsed.hostname and not parsed.port
                 and issuer == "https://" + parsed.hostname
                 and re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", parsed.hostname))
    except ValueError:
        valid = False
    if not valid:
        raise ConfigurationError("CHATGPT_ISSUER must be a canonical HTTPS origin.")
    if (environ.get("CHATGPT_RESOURCE") != issuer + "/chatgpt/mcp"
            or not environ.get("CHATGPT_CLIENT_ID") or not environ.get("CHATGPT_REDIRECT_URI")):
        raise ConfigurationError("Complete ChatGPT OAuth configuration is required.")
    if len(environ.get("SECRET_KEY", "").encode()) < 32:
        raise ConfigurationError("A stable session key of at least 32 bytes is required.")
    db_text = environ.get("STUDIO_DB_PATH", "")
    db = Path(db_text)
    if (not db.is_absolute() or ".." in db.parts or db == DATA_DIR
            or not db.is_relative_to(DATA_DIR) or str(db) != db_text):
        raise ConfigurationError("STUDIO_DB_PATH must be a file within /data/studio.")
    app_environment = {key: environ[key] for key in APP_ENV_KEYS if key in environ}
    app_environment["DREDGE_CHATGPT_RUN_TOOLS_ENABLED"] = (
        "true" if runs_flag == "true" else "false"
    )
    app_environment["DREDGE_CHATGPT_CONTROLS_ENABLED"] = (
        "true" if controls_flag == "true" else "false"
    )
    # This origin is deliberately tied to ChatGPT's single canonical issuer.
    if app_environment.get("OAUTH_REDIRECT_BASE", issuer) != issuer:
        raise ConfigurationError("Provider callbacks must use the canonical ChatGPT origin.")
    app_environment["OAUTH_REDIRECT_BASE"] = issuer
    return Configuration(app_environment, token, port, issuer)


def child_specs(config: Configuration, socket_fd: int) -> tuple[ChildSpec, ...]:
    if socket_fd < 3:
        raise ConfigurationError("A prebound inherited socket descriptor is required.")
    app_env = {**BASE_ENV, **config.app_environment,
               "DREDGE_CHATGPT_ENABLED": "true", "DREDGE_CHATGPT_TUNNEL_ENABLED": "true",
               "DREDGE_TUNNEL_SOCKET_FD": str(socket_fd), "TMPDIR": "/data/studio/tmp"}
    app = ChildSpec("application", (
        PYTHON, "-m", "uvicorn", "dredge.chatgpt_tunnel:create_tunnel_app", "--factory",
        "--fd", str(socket_fd), "--no-proxy-headers", "--no-access-log",
    ), app_env, APP_UID, APP_GID, (SOCKET_GID,), (socket_fd,))
    tunnel = ChildSpec("connector", (
        CLOUDFLARED, "tunnel", "--no-autoupdate", "--metrics", "127.0.0.1:2000",
        "--management-diagnostics=false", "--grace-period", "5s",
        "run", "--token-file", str(TOKEN_PATH),
    ), {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/nonexistent", "LANG": "C.UTF-8"},
        TUNNEL_UID, TUNNEL_GID, (SOCKET_GID,))
    health = ChildSpec("health", (PYTHON, "-m", "dredge.chatgpt_tunnel_health"),
                       {**BASE_ENV, "PORT": str(config.port), "CHATGPT_ISSUER": config.issuer},
                       APP_UID, APP_GID, (SOCKET_GID,))
    return app, tunnel, health


def _trusted_path(path: Path) -> None:
    info = path.lstat()
    if info.st_uid != 0 or info.st_mode & 0o022 or stat.S_ISLNK(info.st_mode):
        raise ConfigurationError("Runtime assets must be root-owned and non-writable by children.")


def _directory(path: Path, uid: int, gid: int, mode: int) -> None:
    """No symlink traversal at the final component; callers establish parents."""
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        pass
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        os.fchown(fd, uid, gid)
        os.fchmod(fd, mode)
    finally:
        os.close(fd)


def prepare_directories() -> None:
    # The image installs the complete source and venv tree root-owned with no
    # group/other writes. Fail closed if its trusted roots were replaced/mounted.
    for path in (Path("/"), Path("/run"), APP_DIR, APP_DIR / "src",
                 APP_DIR / "src/dredge", Path(__file__), Path("/opt"), Path("/opt/venv"),
                 Path("/opt/venv/bin"), Path(PYTHON).resolve(), Path(CLOUDFLARED)):
        _trusted_path(path)
    _directory(Path("/data"), 0, 0, 0o755)
    _directory(DATA_DIR, APP_UID, APP_GID, 0o700)
    # Application-owned children cannot escape the private parent directory;
    # reject symlinks before root modifies any pre-existing temporary directory.
    _directory(DATA_DIR / "tmp", APP_UID, APP_GID, 0o700)
    _directory(RUN_DIR, 0, SOCKET_GID, 0o750)
    _directory(TOKEN_DIR, 0, TUNNEL_GID, 0o750)


def acquire_lock() -> int:
    fd = os.open(RUN_DIR / "supervisor.lock",
                 os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_nlink != 1:
            raise ConfigurationError("The supervisor lock is not a trusted regular file.")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.fchmod(fd, 0o600)
        return fd
    except BaseException:
        os.close(fd)
        raise


def bind_gateway_socket(path: Path = SOCKET_PATH, *, owner_uid: int = 0,
                        socket_gid: int = SOCKET_GID) -> socket.socket:
    """Bind under restrictive umask before chmod/listen: never world-connectable."""
    try:
        existing = path.lstat()
    except FileNotFoundError:
        existing = None
    if existing is not None:
        if not stat.S_ISSOCK(existing.st_mode) or existing.st_uid != owner_uid:
            raise ConfigurationError("Refusing to replace an untrusted socket path.")
        path.unlink()
    previous_umask = os.umask(0o177)
    listener = None
    try:
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(path))
        os.chown(path, owner_uid, socket_gid, follow_symlinks=False)
        os.chmod(path, 0o660)
        listener.listen(128)
        return listener
    except BaseException:
        if listener is not None:
            listener.close()
        raise
    finally:
        os.umask(previous_umask)


def write_token_file(path: Path, token: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o400)
    try:
        os.fchown(fd, TUNNEL_UID, TUNNEL_GID)
        os.fchmod(fd, 0o400)
        with os.fdopen(fd, "w", closefd=False) as stream:
            stream.write(token)
            stream.flush()
    finally:
        os.close(fd)


def spawn_child(spec: ChildSpec, *, cwd: Path = APP_DIR,
                drop_privileges: bool = True) -> subprocess.Popen:
    identity = {"user": spec.uid, "group": spec.gid, "extra_groups": spec.extra_groups}
    return subprocess.Popen(spec.argv, env=spec.environment, cwd=cwd,
                            stdin=subprocess.DEVNULL, close_fds=True, pass_fds=spec.pass_fds,
                            start_new_session=True, umask=0o077,
                            **(identity if drop_privileges else {}))


def _has_exited(process: subprocess.Popen) -> bool:
    # WNOWAIT keeps the session leader's PID reserved until its entire group is
    # signalled. Popen.poll() here would reap it and introduce a PGID reuse race.
    return os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None


def stop_children(children: list[subprocess.Popen], grace_seconds: float) -> None:
    def signal_groups(sig: int) -> None:
        for child in children:
            if child.pid <= 1 or child.pid == os.getpgrp():
                raise RuntimeError("Refusing to signal the supervisor process group.")
            try:
                os.killpg(child.pid, sig)
            except ProcessLookupError:
                pass
    if not children:
        return
    signal_groups(signal.SIGTERM)
    # Keep leaders unreaped throughout the bounded grace period so a surviving
    # grandchild is also killed, even when its direct parent exits first.
    time.sleep(grace_seconds)
    signal_groups(signal.SIGKILL)
    for child in children:
        child.wait()
    # PID 1 (or the Linux subreaper below) also owns orphaned grandchildren.
    # This supervisor owns no unrelated children in the dedicated container.
    # SIGKILL delivery and reparenting are asynchronous. Wait for adopted
    # grandchildren too; returning on the first WNOHANG result would leak zombies.
    deadline = time.monotonic() + 2.0
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            break
        if pid == 0:
            if time.monotonic() >= deadline:
                raise RuntimeError("A descendant did not terminate after process-group shutdown.")
            time.sleep(0.01)


def supervise(specs: tuple[ChildSpec, ...] | list[ChildSpec], stop: threading.Event, *,
              cwd: Path = APP_DIR, drop_privileges: bool = True,
              grace_seconds: float = 5.0, poll_seconds: float = 0.1) -> int:
    children = []
    try:
        for spec in specs:
            if stop.is_set():
                break
            children.append(spawn_child(spec, cwd=cwd, drop_privileges=drop_privileges))
        while not stop.is_set():
            if any(_has_exited(child) for child in children):
                return 1  # A clean child exit is still failure of this service.
            stop.wait(poll_seconds)
        # A requested stop succeeds only if no child already exited. Keep the
        # leaders unreaped here, and let cleanup failures override this result.
        return 1 if any(_has_exited(child) for child in children) else 0
    finally:
        stop_children(children, grace_seconds)


def harden_supervisor() -> None:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    libc = ctypes.CDLL(None, use_errno=True)
    # No core/ptrace disclosure; adopt and reap orphaned descendants when not PID1.
    for option, value in ((4, 0), (36, 1)):  # PR_SET_DUMPABLE, PR_SET_CHILD_SUBREAPER
        if libc.prctl(option, value, 0, 0, 0) != 0:
            raise ConfigurationError("Required Linux supervisor hardening failed.")


def main(environ: Mapping[str, str] | None = None) -> int:
    listener = None
    lock_fd = None
    token_created = False
    old_handlers = {}
    stop = threading.Event()
    try:
        if os.geteuid() != 0:
            raise ConfigurationError("This dedicated bootstrap must start as root.")
        harden_supervisor()
        try:
            config = read_configuration(os.environ if environ is None else environ)
        finally:
            # Child environments are assembled explicitly; remove inherited
            # secrets even if validation fails. Never print the source values.
            os.environ.clear()
        for sig in (signal.SIGTERM, signal.SIGINT):
            old_handlers[sig] = signal.signal(sig, lambda *_: stop.set())
        os.umask(0o077)
        prepare_directories()
        lock_fd = acquire_lock()
        if TOKEN_PATH.exists() or TOKEN_PATH.is_symlink():
            info = TOKEN_PATH.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_uid != TUNNEL_UID or info.st_nlink != 1:
                raise ConfigurationError("Refusing to replace an untrusted token file.")
            TOKEN_PATH.unlink()  # trusted stale runtime token, never a mounted credential
        write_token_file(TOKEN_PATH, config.token)
        token_created = True
        listener = bind_gateway_socket()
        return supervise(child_specs(config, listener.fileno()), stop)
    except (RuntimeError, OSError, ValueError):
        # Do not include exception repr, argv, environment values or token bytes.
        print("Dedicated ChatGPT tunnel startup or supervision failed; check configuration and permissions.", file=sys.stderr)
        return 1
    finally:
        if listener is not None:
            listener.close()
            SOCKET_PATH.unlink(missing_ok=True)
        if token_created:
            TOKEN_PATH.unlink(missing_ok=True)
        if lock_fd is not None:
            os.close(lock_fd)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
