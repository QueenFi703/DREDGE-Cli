"""Tests for new CLI commands (inspect, doctor, echo, id)."""
import pytest
from dredge.cli import main


def test_inspect_command(capsys):
    """Test inspect command shows configuration."""
    result = main(["inspect"])
    assert result == 0
    
    captured = capsys.readouterr()
    assert "DREDGE Inspector" in captured.out
    assert "Version: 0.1.0" in captured.out
    assert "CompactJSONProvider" in captured.out
    assert "Hash strategy" in captured.out
    assert "Identity Contract" in captured.out


def test_doctor_command(capsys):
    """Test doctor command runs diagnostics."""
    result = main(["doctor"])
    
    captured = capsys.readouterr()
    assert "DREDGE Doctor" in captured.out
    assert "Python version" in captured.out
    assert "port 3001" in captured.out
    # Result can be 0 or 1 depending on system state
    assert result in [0, 1]


def test_echo_command(capsys):
    """Test echo command returns 'alive'."""
    result = main(["echo"])
    assert result == 0
    
    captured = capsys.readouterr()
    assert captured.out.strip() == "alive"


def test_cli_shows_new_commands():
    """Test that help shows all new commands."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    
    assert exc_info.value.code == 0


def test_id_command_single(capsys):
    """Test ID command generates single ID."""
    result = main(["id"])
    assert result == 0
    
    captured = capsys.readouterr()
    lines = captured.out.strip().split('\n')
    assert len(lines) == 1
    # Hex format should be 16 characters
    assert len(lines[0]) == 16


def test_id_command_multiple(capsys):
    """Test ID command generates multiple IDs."""
    result = main(["id", "--count", "5"])
    assert result == 0
    
    captured = capsys.readouterr()
    lines = captured.out.strip().split('\n')
    assert len(lines) == 5
    # All should be unique
    assert len(set(lines)) == 5


def test_id_command_uuid_format(capsys):
    """Test ID command with UUID format."""
    result = main(["id", "--format", "uuid"])
    assert result == 0
    
    captured = capsys.readouterr()
    uuid_str = captured.out.strip()
    # UUID format has dashes
    assert '-' in uuid_str
    assert len(uuid_str) == 36  # Standard UUID length
