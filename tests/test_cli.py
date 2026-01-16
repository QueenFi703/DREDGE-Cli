"""Tests for the DREDGE CLI."""
import subprocess
import sys


def test_cli_entry_point():
    """Test that the dredge-cli command is available as an entry point."""
    result = subprocess.run(
        ["dredge-cli", "--version"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "0.1.4" in result.stdout


def test_cli_help():
    """Test that the dredge-cli command shows help."""
    result = subprocess.run(
        ["dredge-cli", "--help"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "DREDGE x Dolly" in result.stdout
    assert "serve" in result.stdout


def test_cli_serve_help():
    """Test that the dredge-cli serve command shows help."""
    result = subprocess.run(
        ["dredge-cli", "serve", "--help"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "--host" in result.stdout
    assert "--port" in result.stdout
    assert "--debug" in result.stdout


def test_cli_module_invocation():
    """Test that python -m dredge also works."""
    result = subprocess.run(
        [sys.executable, "-m", "dredge", "--version"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "0.1.4" in result.stdout
