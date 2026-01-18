"""Tests for DREDGE health check and configuration features."""
import subprocess
import sys
import json
import tempfile
from pathlib import Path
import pytest


def test_health_command():
    """Test the health command."""
    result = subprocess.run(
        ["dredge-cli", "health"],
        capture_output=True,
        text=True
    )
    # Health check may fail if dependencies aren't installed, but command should work
    assert "Health Status:" in result.stdout
    assert "Dependency Checks:" in result.stdout


def test_health_command_json():
    """Test the health command with JSON output."""
    result = subprocess.run(
        ["dredge-cli", "health", "--json"],
        capture_output=True,
        text=True
    )
    # Should be valid JSON
    try:
        data = json.loads(result.stdout)
        assert "status" in data
        assert "checks" in data
        assert "system" in data
    except json.JSONDecodeError:
        pytest.fail("Health command did not output valid JSON")


def test_info_command():
    """Test the info command."""
    result = subprocess.run(
        ["dredge-cli", "info"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "System Information:" in result.stdout
    assert "Python:" in result.stdout
    assert "Platform:" in result.stdout


def test_version_info_command():
    """Test the --version-info flag."""
    result = subprocess.run(
        ["dredge-cli", "--version-info"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "DREDGE version" in result.stdout
    assert "System Information:" in result.stdout


def test_config_init():
    """Test config init command."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = Path(tmpdir) / ".dredge.json"
        
        result = subprocess.run(
            ["dredge-cli", "config", "init"],
            capture_output=True,
            text=True,
            cwd=tmpdir
        )
        
        assert result.returncode == 0
        assert "Configuration file created" in result.stdout
        assert config_file.exists()
        
        # Verify it's valid JSON
        with open(config_file) as f:
            config = json.load(f)
            assert "server" in config
            assert "mcp" in config


def test_config_show():
    """Test config show command."""
    result = subprocess.run(
        ["dredge-cli", "config", "show"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    
    # Should be valid JSON
    try:
        config = json.loads(result.stdout)
        assert "server" in config
        assert "mcp" in config
    except json.JSONDecodeError:
        pytest.fail("Config show did not output valid JSON")


def test_config_path():
    """Test config path command."""
    result = subprocess.run(
        ["dredge-cli", "config", "path"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert ".dredge.json" in result.stdout


def test_health_module():
    """Test health module functions."""
    from dredge.health import get_system_info, check_dependencies, check_health
    
    # Test get_system_info
    info = get_system_info()
    assert "python_version" in info
    assert "platform" in info
    assert "torch_version" in info
    
    # Test check_dependencies
    deps = check_dependencies()
    assert isinstance(deps, dict)
    assert "flask" in deps
    assert "torch" in deps
    
    # Test check_health
    health = check_health()
    assert "status" in health
    assert "checks" in health
    assert "system" in health


def test_config_module():
    """Test config module functions."""
    from dredge.config import load_config, DEFAULT_CONFIG
    
    # Test load_config returns valid structure
    config = load_config()
    assert isinstance(config, dict)
    assert "server" in config
    assert "mcp" in config
    
    # Test defaults are present
    assert config["server"]["port"] == DEFAULT_CONFIG["server"]["port"]
