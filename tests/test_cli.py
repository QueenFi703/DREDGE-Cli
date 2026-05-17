"""Tests for the DREDGE CLI."""
import subprocess
import sys

import dredge


def _run_cli(*args):
    """Run CLI via module invocation so tests do not rely on editable installs."""
    return subprocess.run(
        [sys.executable, "-m", "dredge", *args],
        capture_output=True,
        text=True,
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
