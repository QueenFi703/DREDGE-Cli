"""Isolated bootstrap and real local subprocess tests; no tunnel/provider calls."""
import json
import os
from pathlib import Path
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from dredge import chatgpt_tunnel_supervisor as supervisor


FIXTURE_ENV = {
    "DREDGE_CHATGPT_TUNNEL_ENABLED": "true",
    "DREDGE_CHATGPT_ENABLED": "true",
    "CHATGPT_ISSUER": "https://dredge.example.test",
    "CHATGPT_RESOURCE": "https://dredge.example.test/chatgpt/mcp",
    "CHATGPT_CLIENT_ID": "fictional-client",
    "CHATGPT_REDIRECT_URI": "https://chatgpt.com/connector/oauth/fictional",
    "SECRET_KEY": "fictional-session-key-at-least-32-bytes",
    "STUDIO_DB_PATH": "/data/studio/studio.sqlite3",
    "TUNNEL_TOKEN": "fictional-not-a-real-tunnel-token",
    "PORT": "8123",
}


class ConfigurationTests(unittest.TestCase):
    def test_fail_closed_requires_both_explicit_flags(self):
        for name in ("DREDGE_CHATGPT_ENABLED", "DREDGE_CHATGPT_TUNNEL_ENABLED"):
            for value in ("", "false", "1", "yes"):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(supervisor.ConfigurationError):
                        supervisor.read_configuration({**FIXTURE_ENV, name: value})

    def test_token_and_db_and_public_origin_required(self):
        for overrides in (
            {"TUNNEL_TOKEN": ""}, {"TUNNEL_TOKEN": "fictional\ninvalid"},
            {"STUDIO_DB_PATH": "/tmp/app.sqlite"},
            {"STUDIO_DB_PATH": "/data/studio/../../tmp/app.sqlite"},
            {"STUDIO_DB_PATH": "/data/studio"},
            {"PORT": "2000"}, {"PORT": "80"}, {"PORT": "70000"},
            {"CHATGPT_ISSUER": "http://dredge.example.test"},
            {"CHATGPT_ISSUER": "https://dredge.example.test/path"},
            {"SECRET_KEY": "short"},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaises(supervisor.ConfigurationError):
                    supervisor.read_configuration({**FIXTURE_ENV, **overrides})

    def test_secrets_are_not_in_repr_or_untrusted_child_environments(self):
        source = {**FIXTURE_ENV, "GOOGLE_CLIENT_SECRET": "fictional-provider-secret",
                  "PATH": "/tmp/evil", "PYTHONPATH": "/tmp/evil",
                  "LD_PRELOAD": "/tmp/evil.so", "DREDGE_UNKNOWN_SECRET": "fictional-other",
                  "STRIPE_RESTRICTED_KEY": "fictional-payment-secret",
                  "OPENAI_API_KEY": "fictional-live-provider-secret"}
        config = supervisor.read_configuration(source)
        specs = supervisor.child_specs(config, 42)
        self.assertNotIn(source["TUNNEL_TOKEN"], repr(config))
        self.assertNotIn(source["SECRET_KEY"], repr(config))
        app, tunnel, health = specs
        self.assertEqual(app.environment["GOOGLE_CLIENT_SECRET"], "fictional-provider-secret")
        self.assertEqual(app.environment["DREDGE_TUNNEL_SOCKET_FD"], "42")
        self.assertEqual(app.pass_fds, (42,))
        self.assertEqual(app.uid, 1000)
        self.assertEqual(tunnel.uid, 1001)
        self.assertEqual(health.uid, 1000)
        for spec in specs:
            self.assertEqual(spec.extra_groups, (2000,))
            self.assertNotIn("TUNNEL_TOKEN", spec.environment)
            self.assertNotIn("LD_PRELOAD", spec.environment)
            self.assertNotIn("DREDGE_UNKNOWN_SECRET", spec.environment)
            self.assertNotIn("STRIPE_RESTRICTED_KEY", spec.environment)
            self.assertNotIn("OPENAI_API_KEY", spec.environment)
            self.assertNotIn("/tmp/evil", repr(spec.environment))
            self.assertNotIn(source["TUNNEL_TOKEN"], repr(spec))
        for spec in (tunnel, health):
            self.assertNotIn("SECRET_KEY", spec.environment)
            self.assertNotIn("GOOGLE_CLIENT_SECRET", spec.environment)
            self.assertNotIn("STUDIO_DB_PATH", spec.environment)
            self.assertEqual(spec.pass_fds, ())
        self.assertEqual(health.environment["CHATGPT_ISSUER"], source["CHATGPT_ISSUER"])
        self.assertEqual(health.environment["PORT"], "8123")

    def test_only_dedicated_fd_service_and_authenticated_named_tunnel(self):
        app, tunnel, health = supervisor.child_specs(supervisor.read_configuration(FIXTURE_ENV), 9)
        self.assertEqual(app.argv[1:5], ("-m", "uvicorn", "dredge.chatgpt_tunnel:create_tunnel_app", "--factory"))
        self.assertIn("--no-proxy-headers", app.argv)
        self.assertIn("--no-access-log", app.argv)
        self.assertNotIn("--port", app.argv)
        self.assertNotIn("--host", app.argv)
        self.assertEqual(app.argv[app.argv.index("--fd") + 1], "9")
        self.assertIn("run", tunnel.argv)
        self.assertIn("--token-file", tunnel.argv)
        self.assertNotIn("--token", tunnel.argv)
        self.assertNotIn("--url", tunnel.argv)
        self.assertIn("127.0.0.1:2000", tunnel.argv)
        self.assertIn("--no-autoupdate", tunnel.argv)
        self.assertEqual(health.argv[1:], ("-m", "dredge.chatgpt_tunnel_health"))

    def test_nonroot_startup_fails_without_children_or_side_effects(self):
        with patch.object(os, "geteuid", return_value=1000), patch.object(subprocess, "Popen") as spawn:
            self.assertNotEqual(supervisor.main(FIXTURE_ENV), 0)
            spawn.assert_not_called()


def require_unix_socket(test_case):
    try:
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    except PermissionError:
        test_case.skipTest("AF_UNIX is denied by this sandbox; run this real-socket test in CI/container.")
    else:
        probe.close()


class FilesystemTests(unittest.TestCase):
    def test_prebound_socket_has_safe_modes_before_listen_and_is_inheritable_only_explicitly(self):
        require_unix_socket(self)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gateway.sock"
            def assert_private_before_group_access(*args, **kwargs):
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            with patch.object(os, "chown", side_effect=assert_private_before_group_access) as chown:
                listener = supervisor.bind_gateway_socket(path, owner_uid=os.getuid(), socket_gid=os.getgid())
            try:
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o660)
                self.assertEqual(listener.family, socket.AF_UNIX)
                self.assertEqual(listener.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN), 1)
                self.assertFalse(os.get_inheritable(listener.fileno()))
                chown.assert_called_once_with(path, os.getuid(), os.getgid(), follow_symlinks=False)
            finally:
                listener.close()

    def test_refuses_to_unlink_non_socket_or_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gateway.sock"
            path.write_text("preserve")
            with self.assertRaises(supervisor.ConfigurationError):
                supervisor.bind_gateway_socket(path)
            self.assertEqual(path.read_text(), "preserve")
            path.unlink()
            path.symlink_to(Path(tmp) / "other")
            with self.assertRaises(supervisor.ConfigurationError):
                supervisor.bind_gateway_socket(path)

    def test_token_written_exclusively_with_0400_and_distinct_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "token"
            with patch.object(os, "fchown") as chown:
                supervisor.write_token_file(path, "fictional-token")
            self.assertEqual(path.read_text(), "fictional-token")
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o400)
            chown.assert_called_once_with(unittest.mock.ANY, 1001, 1001)
            with self.assertRaises(FileExistsError):
                supervisor.write_token_file(path, "do-not-replace")
            self.assertEqual(path.read_text(), "fictional-token")

    def test_token_symlink_is_never_followed(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "target"
            target.write_text("preserve")
            path = Path(tmp) / "token"
            path.symlink_to(target)
            with self.assertRaises(OSError):
                supervisor.write_token_file(path, "fictional-token")
            self.assertEqual(target.read_text(), "preserve")


class ProcessLifecycleTests(unittest.TestCase):
    def spec(self, script, *args):
        return supervisor.ChildSpec(
            "fixture", (sys.executable, "-c", script, *args),
            {"PATH": "/usr/bin:/bin"}, os.getuid(), os.getgid(), (), (),
        )

    def run_specs(self, specs, stop=None):
        return supervisor.supervise(specs, stop or threading.Event(),
                                    cwd=Path.cwd(), drop_privileges=False,
                                    grace_seconds=0.10, poll_seconds=0.01)

    def assert_child_exit_is_failure(self, exitcode):
        created = []
        original = supervisor.spawn_child
        def track(*args, **kwargs):
            child = original(*args, **kwargs)
            created.append(child)
            return child
        with patch.object(supervisor, "spawn_child", side_effect=track):
            result = self.run_specs([
                self.spec(f"import sys,time; time.sleep(0.15); sys.exit({exitcode})"),
                self.spec("import time; time.sleep(30)"),
                self.spec("import time; time.sleep(30)"),
            ])
        self.assertNotEqual(result, 0)
        self.assertEqual(len(created), 3)
        self.assertEqual(created[0].returncode, exitcode)
        for process in created:
            self.assertIsNotNone(process.returncode)
            with self.assertRaises(ChildProcessError):
                os.waitpid(process.pid, os.WNOHANG)

    def test_even_clean_child_exit_is_failure_and_reaps_siblings(self):
        self.assert_child_exit_is_failure(0)

    def test_nonzero_child_exit_is_failure_and_reaps_siblings(self):
        self.assert_child_exit_is_failure(7)

    def test_stop_before_start_succeeds_without_spawning(self):
        stop = threading.Event()
        stop.set()
        with patch.object(supervisor, "spawn_child") as spawn:
            result = self.run_specs([self.spec("raise AssertionError('must not start')")], stop)
        self.assertEqual(result, 0)
        spawn.assert_not_called()

    def test_stop_during_start_reaps_started_child_without_spawning_more(self):
        stop = threading.Event()
        created = []
        original = supervisor.spawn_child
        def track(*args, **kwargs):
            child = original(*args, **kwargs)
            created.append(child)
            stop.set()
            return child
        with patch.object(supervisor, "spawn_child", side_effect=track):
            result = self.run_specs([
                self.spec("import time; time.sleep(30)"),
                self.spec("raise AssertionError('must not start')"),
            ], stop)
        self.assertEqual(result, 0)
        self.assertEqual(len(created), 1)
        self.assertIsNotNone(created[0].returncode)
        with self.assertRaises(ChildProcessError):
            os.waitpid(created[0].pid, os.WNOHANG)

    def test_stop_does_not_mask_an_already_exited_child(self):
        for exitcode in (0, 7):
            with self.subTest(exitcode=exitcode):
                stop = threading.Event()
                created = []
                original = supervisor.spawn_child
                def track(*args, **kwargs):
                    child = original(*args, **kwargs)
                    created.append(child)
                    # Wait for a real exit without reaping the group leader.
                    os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOWAIT)
                    stop.set()
                    return child
                with patch.object(supervisor, "spawn_child", side_effect=track):
                    result = self.run_specs([self.spec(f"raise SystemExit({exitcode})")], stop)
                self.assertNotEqual(result, 0)
                self.assertEqual(created[0].returncode, exitcode)

    def test_requested_stop_does_not_swallow_child_cleanup_failure(self):
        stop = threading.Event()
        original_spawn = supervisor.spawn_child
        original_cleanup = supervisor.stop_children
        def spawn_and_stop(*args, **kwargs):
            child = original_spawn(*args, **kwargs)
            stop.set()
            return child
        def cleanup_then_fail(*args, **kwargs):
            original_cleanup(*args, **kwargs)
            raise RuntimeError("fixture cleanup failure")
        with patch.object(supervisor, "spawn_child", side_effect=spawn_and_stop), \
                patch.object(supervisor, "stop_children", side_effect=cleanup_then_fail):
            with self.assertRaisesRegex(RuntimeError, "fixture cleanup failure"):
                self.run_specs([self.spec("import time; time.sleep(30)")], stop)

    def test_spawn_explicitly_drops_all_inherited_groups(self):
        spec = supervisor.child_specs(supervisor.read_configuration(FIXTURE_ENV), 7)[0]
        with patch.object(subprocess, "Popen") as spawn:
            supervisor.spawn_child(spec)
        kwargs = spawn.call_args.kwargs
        self.assertEqual(kwargs["user"], 1000)
        self.assertEqual(kwargs["group"], 1000)
        self.assertEqual(kwargs["extra_groups"], (2000,))
        self.assertTrue(kwargs["start_new_session"])
        self.assertTrue(kwargs["close_fds"])
        self.assertEqual(kwargs["pass_fds"], (7,))
        self.assertEqual(kwargs["umask"], 0o077)

    def assert_real_signal_reaps_children_and_grandchildren(self, stop_signal):
        # A separate local test harness becomes a subreaper, so this test cannot
        # adopt unrelated pytest children. Its grandchild ignores SIGTERM.
        harness = r'''
import ctypes, os, signal, sys, threading
from pathlib import Path
from dredge import chatgpt_tunnel_supervisor as s
assert ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) == 0
stop = threading.Event()
signal.signal(signal.SIGTERM, lambda *_: stop.set())
signal.signal(signal.SIGINT, lambda *_: stop.set())
script = "import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); child=os.fork(); print('READY',os.getpid(),flush=True); time.sleep(30)"
spec = s.ChildSpec('fixture', (sys.executable, '-c', script), {}, os.getuid(), os.getgid(), (), ())
result = s.supervise([spec], stop, cwd=Path.cwd(), drop_privileges=False, grace_seconds=0.10)
try:
    os.waitpid(-1, os.WNOHANG)
except ChildProcessError:
    print('ALL_REAPED', result, flush=True)
else:
    raise AssertionError('A child was not reaped')
raise SystemExit(result)
'''
        process = subprocess.Popen([sys.executable, "-u", "-c", harness],
                                   env={"PYTHONPATH": str(Path(supervisor.__file__).parents[1])},
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                   start_new_session=True)
        try:
            pids = [int(process.stdout.readline().split()[1]) for _ in range(2)]
            process.send_signal(stop_signal)
            stdout, stderr = process.communicate(timeout=5)
            self.assertEqual(process.returncode, 0, stderr)
            self.assertIn("ALL_REAPED 0", stdout)
            for pid in pids:
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()

    def test_real_sigterm_reaps_children_and_grandchildren(self):
        self.assert_real_signal_reaps_children_and_grandchildren(signal.SIGTERM)

    def test_real_sigint_reaps_children_and_grandchildren(self):
        self.assert_real_signal_reaps_children_and_grandchildren(signal.SIGINT)

    def test_stop_terminates_and_kills_child_ignoring_sigterm(self):
        stop = threading.Event()
        timer = threading.Timer(0.25, stop.set)
        created = []
        original = supervisor.spawn_child
        def track(*args, **kwargs):
            process = original(*args, **kwargs)
            created.append(process)
            return process
        timer.start()
        try:
            with patch.object(supervisor, "spawn_child", side_effect=track):
                result = self.run_specs([
                    self.spec("import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)")
                ], stop)
        finally:
            timer.cancel()
        self.assertEqual(result, 0)
        self.assertEqual(created[0].returncode, -signal.SIGKILL)

    def test_spawn_failure_reaps_already_started_child(self):
        created = []
        original = supervisor.spawn_child
        def track(spec, **kwargs):
            process = original(spec, **kwargs)
            created.append(process)
            return process
        invalid = supervisor.ChildSpec("missing", ("/definitely/not/an/executable",), {},
                                       os.getuid(), os.getgid(), (), ())
        with patch.object(supervisor, "spawn_child", side_effect=track):
            with self.assertRaises(FileNotFoundError):
                self.run_specs([self.spec("import time; time.sleep(30)"), invalid])
        self.assertIsNotNone(created[0].returncode)

    def test_main_exit_status_preserves_configuration_and_cleanup_failures(self):
        # Isolate main's environment clearing and signal handlers in a process.
        # Only privileged runtime assets are replaced; supervise and main keep
        # their actual return/exception behavior and drive the harness exit code.
        harness = r'''
import json, os, sys
from unittest.mock import Mock, patch
from dredge import chatgpt_tunnel_supervisor as s
stage, config = sys.argv[1], json.loads(sys.argv[2])
token = Mock()
token.exists.return_value = False
token.is_symlink.return_value = False
socket_path = Mock()
listener = Mock()
listener.fileno.return_value = 9
close_lock = Mock()
original_supervise = s.supervise
def requested_stop(specs, stop):
    stop.set()
    return original_supervise([], stop)
def fail_cleanup(*args, **kwargs):
    print('INJECTED', stage, flush=True)
    raise OSError('fixture cleanup failure')
cleanup = s.stop_children
if stage == 'configuration':
    config['DREDGE_CHATGPT_ENABLED'] = 'false'
elif stage == 'children':
    cleanup = fail_cleanup
elif stage == 'listener':
    listener.close.side_effect = fail_cleanup
elif stage == 'socket':
    socket_path.unlink.side_effect = fail_cleanup
elif stage == 'token':
    token.unlink.side_effect = fail_cleanup
elif stage == 'lock':
    close_lock.side_effect = fail_cleanup
with patch.object(os, 'geteuid', return_value=0), \
        patch.object(s, 'harden_supervisor'), \
        patch.object(s, 'prepare_directories'), \
        patch.object(s, 'acquire_lock', return_value=9), \
        patch.object(s, 'TOKEN_PATH', token), \
        patch.object(s, 'SOCKET_PATH', socket_path), \
        patch.object(s, 'write_token_file'), \
        patch.object(s, 'bind_gateway_socket', return_value=listener), \
        patch.object(s, 'supervise', side_effect=requested_stop), \
        patch.object(s, 'stop_children', side_effect=cleanup), \
        patch.object(os, 'close', close_lock):
    raise SystemExit(s.main(config))
'''
        for stage in ("success", "configuration", "children", "listener", "socket", "token", "lock"):
            with self.subTest(stage=stage):
                result = subprocess.run(
                    [sys.executable, "-c", harness, stage, json.dumps(FIXTURE_ENV)],
                    env={"PYTHONPATH": str(Path(supervisor.__file__).parents[1])},
                    capture_output=True, text=True, timeout=5,
                )
                self.assertEqual(result.returncode, 0 if stage == "success" else 1, result.stderr)
                if stage not in ("success", "configuration"):
                    self.assertIn("INJECTED " + stage, result.stdout)
                if stage in ("configuration", "children"):
                    self.assertIn("startup or supervision failed", result.stderr)

    def test_inherited_socket_is_usable_and_environment_is_not_inherited(self):
        require_unix_socket(self)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "listener.sock"
            listener = socket.socket(socket.AF_UNIX)
            listener.bind(str(path))
            listener.listen()
            spec = supervisor.ChildSpec("fd-fixture", (
                sys.executable, "-c",
                "import os,socket; s=socket.socket(fileno=int(os.environ['FD'])); "
                "c,_=s.accept(); c.sendall(b'isolated' if 'TUNNEL_TOKEN' not in os.environ else b'leaked'); c.close()",
            ), {"FD": str(listener.fileno())}, os.getuid(), os.getgid(), (), (listener.fileno(),))
            with patch.dict(os.environ, {"TUNNEL_TOKEN": "fictional-parent-only"}):
                process = supervisor.spawn_child(spec, cwd=Path.cwd(), drop_privileges=False)
            try:
                client = socket.socket(socket.AF_UNIX)
                client.settimeout(2)
                client.connect(str(path))
                self.assertEqual(client.recv(20), b"isolated")
                client.close()
                self.assertEqual(process.wait(timeout=2), 0)
            finally:
                listener.close()
                if process.poll() is None:
                    process.kill()
                    process.wait()




class ControlsEnvironmentTests(unittest.TestCase):
    def test_controls_flag_is_validated_and_only_application_receives_it(self):
        key = 'DREDGE_CHATGPT_CONTROLS_ENABLED'
        for value in (None, '', 'false', '0', 'off', 'no', 'true', ' TRUE '):
            with self.subTest(value=value):
                source = dict(FIXTURE_ENV)
                if value is not None:
                    source[key] = value
                app, connector, health = supervisor.child_specs(supervisor.read_configuration(source), 42)
                self.assertEqual(app.environment[key], 'true' if value in ('true', ' TRUE ') else 'false')
                self.assertNotIn(key, connector.environment)
                self.assertNotIn(key, health.environment)
        for value in ('1', 'yes', 'enabled', 'secret-value'):
            with self.subTest(value=value):
                with self.assertRaises(supervisor.ConfigurationError) as error:
                    supervisor.read_configuration({**FIXTURE_ENV, key: value})
                self.assertNotIn(value, str(error.exception))

    def test_supervisor_worker_environment_reaches_real_configuration_and_discovery(self):
        from flask import Flask
        from dredge.chatgpt_transport import configure_chatgpt
        from dredge.chatgpt_foundation import tool_definitions, supported_scopes
        from dredge.chatgpt_controls import CONTROL_NAMES
        key = 'DREDGE_CHATGPT_CONTROLS_ENABLED'
        for flag in ('false', 'true'):
            with self.subTest(flag=flag):
                source = {**FIXTURE_ENV, key: flag, 'DREDGE_CHATGPT_RUN_TOOLS_ENABLED': 'true',
                          'OPENAI_API_KEY': 'fictional-never-forward', 'STRIPE_RESTRICTED_KEY': 'fictional-never-forward'}
                worker = supervisor.child_specs(supervisor.read_configuration(source), 42)[0]
                app = Flask(__name__)
                # No filesystem/network OAuth setup: exercise real env parsing and schemas.
                with patch.dict(os.environ, worker.environment, clear=True), patch(
                        'dredge.chatgpt_oauth.register_chatgpt_oauth'):
                    configure_chatgpt(app)
                names = {tool['name'] for tool in tool_definitions(app)}
                self.assertEqual(len(names), 11 if flag == 'true' else 6)
                self.assertEqual(bool(names & CONTROL_NAMES), flag == 'true')
                self.assertEqual('studio:budget:write' in supported_scopes(app), flag == 'true')
                self.assertNotIn('OPENAI_API_KEY', worker.environment)
                self.assertNotIn('STRIPE_RESTRICTED_KEY', worker.environment)


if __name__ == "__main__":
    unittest.main()
