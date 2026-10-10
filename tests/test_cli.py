"""Tests for the DREDGE CLI."""
import subprocess
import sys
import json
from importlib.metadata import distribution
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

import dredge
from dredge.cli import DREDGECLI, cli, main


def _run_cli(*args):
    """Run CLI via module invocation so tests do not rely on editable installs."""
    return subprocess.run(
        [sys.executable, "-m", "dredge", *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_cli_entry_point():
    """Test that the CLI is invokable and reports the current version."""
    result = _run_cli("--version")
    assert result.returncode == 0
    assert dredge.__version__ in result.stdout


def test_cli_help():
    """Test that the dredge-cli command shows help."""
    result = _run_cli("--help")
    assert result.returncode == 0
    assert "DREDGE x Dolly" in result.stdout
    assert "serve" in result.stdout


def test_cli_serve_help():
    """Test that the dredge-cli serve command shows help."""
    result = _run_cli("serve", "--help")
    assert result.returncode == 0
    assert "--host" in result.stdout
    assert "--port" in result.stdout
    assert "--debug" in result.stdout


def test_cli_module_invocation():
    """Test that python -m dredge also works."""
    result = _run_cli("--version")
    assert result.returncode == 0
    assert dredge.__version__ in result.stdout


@pytest.mark.parametrize("entrypoint", ["dredge", "dredge-cli"])
def test_installed_console_entrypoints(entrypoint):
    result = subprocess.run(
        [entrypoint, "--version"], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert dredge.__version__ in result.stdout


def test_all_installed_entrypoints_resolve():
    entrypoints = {
        entry.name: entry
        for entry in distribution("dredge-studio").entry_points
        if entry.group == "console_scripts"
    }
    expected = {
        "dredge": "dredge.cli:main",
        "dredge-cli": "dredge.cli:main",
        "dredge-server": "dredge.server:run",
        "dredge-advanced": "dredge.server:run",
    }
    for name, target in expected.items():
        assert entrypoints[name].value == target
        assert callable(entrypoints[name].load())


def test_package_declares_supported_runtime_and_its_dependencies():
    from packaging.requirements import Requirement
    package = distribution('dredge-studio')
    assert package.metadata['Requires-Python'] == '>=3.10'
    dependencies = {Requirement(value).name.lower() for value in package.requires}
    assert {'flask', 'flask-login', 'authlib', 'stripe', 'cryptography',
            'pillow', 'pypdf', 'click', 'torch', 'pyyaml'} <= dependencies


@pytest.mark.parametrize(
    "command", ["pipeline", "translate", "analyze", "status", "interactive", "version"]
)
def test_pipeline_commands_remain_available(command):
    result = CliRunner().invoke(cli, [command, "--help"])
    assert result.exit_code == 0, result.output


def test_pipeline_dispatch_remains_available(monkeypatch):
    calls = []

    async def run_pipeline(self, data):
        calls.append((data, self.config.pipeline_type))
        return {"result": "mock pipeline result"}

    monkeypatch.setattr(DREDGECLI, "run_pipeline", run_pipeline)
    result = CliRunner().invoke(
        cli, ["--json", "pipeline", "--query", "test query", "--pipeline-type", "ios_swift"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"result": "mock pipeline result"}
    assert calls == [({"query": "test query"}, "ios_swift")]


def test_pipeline_config_flags_remain_available():
    result = CliRunner().invoke(cli, ["config", "--no-cache", "--log-level", "DEBUG"])
    assert result.exit_code == 0, result.output
    assert "Configuration updated" in result.output


@pytest.mark.parametrize("command", ["serve", "mcp"])
def test_runtime_server_dispatch(monkeypatch, command):
    from dredge import runtime_cli

    calls = []
    module = "dredge.server" if command == "serve" else "dredge.mcp_server"
    function = "run" if command == "serve" else "run_mcp_server"

    def run(**kwargs):
        calls.append(kwargs)

    monkeypatch.setitem(sys.modules, module, SimpleNamespace(**{function: run}))
    monkeypatch.setattr(runtime_cli, "load_config", lambda: {})
    args = [command, "--host", "127.0.0.1", "--port", "6543", "--debug", "--threads", "1"]
    expected = {"host": "127.0.0.1", "port": 6543, "debug": True}
    if command == "mcp":
        args.extend(["--device", "cpu"])
        expected["device"] = "cpu"
    assert main(args) == 0
    assert calls == [expected]


def test_runtime_health_preserves_failure_exit_status(monkeypatch, capsys):
    from dredge import runtime_cli

    report = {"status": "unhealthy", "checks": {}, "system": {}}
    monkeypatch.setattr(runtime_cli, "check_health", lambda: report)
    assert main(["health", "--json"]) == 1
    assert json.loads(capsys.readouterr().out) == report


def test_runtime_sync_dispatch(monkeypatch, tmp_path, capsys):
    from dredge import sync

    calls = []
    monkeypatch.setattr(sync, "sync", lambda manifest_path: calls.append(manifest_path))
    manifest = tmp_path / "custom.yaml"
    assert main(["sync", "--manifest", str(manifest)]) == 0
    assert calls == [manifest]
    assert "sync complete" in capsys.readouterr().out


def test_runtime_github_event_file(tmp_path, capsys):
    events = tmp_path / "events.jsonl"
    events.write_text('{"type": "push"}\n{"type": "pull_request"}\n', encoding="utf-8")
    assert main(["github-event", "--event-file", str(events), "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["total_processed"] == 2
    assert summary["successful"] == 2
    assert summary["errors"] == 0


def test_main_returns_usage_error(capsys):
    assert main(["not-a-command"]) == 2
    assert "No such command" in capsys.readouterr().err
