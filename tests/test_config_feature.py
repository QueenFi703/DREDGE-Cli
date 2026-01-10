"""Tests for DREDGE configuration management feature."""
import os
import tempfile
import shutil
from io import StringIO
from unittest.mock import patch
import pytest
from dredge.cli import (
    main,
    load_config,
    save_config,
    get_config_path,
    get_default_config,
    cmd_config_list,
    cmd_config_get,
    cmd_config_set,
    cmd_config_reset,
    cmd_config_path
)


@pytest.fixture
def temp_config_dir():
    """Create a temporary directory for config files."""
    temp_dir = tempfile.mkdtemp()
    original_home = os.environ.get('HOME')
    os.environ['HOME'] = temp_dir
    
    yield temp_dir
    
    # Cleanup
    if original_home:
        os.environ['HOME'] = original_home
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_get_default_config():
    """Test getting default configuration."""
    config = get_default_config()
    
    assert 'server' in config
    assert config['server']['host'] == '0.0.0.0'
    assert config['server']['port'] == 3001
    assert config['server']['debug'] is False
    assert config['server']['reload'] is False
    
    assert 'id' in config
    assert config['id']['default_strategy'] == 'fast'
    assert config['id']['default_count'] == 1


def test_config_path_resolution(temp_config_dir):
    """Test configuration file path resolution."""
    # Should return preferred path when no config exists
    path = get_config_path()
    assert path.endswith('.config/dredge/config.toml')


def test_save_and_load_config(temp_config_dir):
    """Test saving and loading configuration."""
    test_config = {
        'server': {
            'host': 'localhost',
            'port': 8080,
            'debug': True
        },
        'id': {
            'default_strategy': 'infrastructure',
            'default_count': 5
        }
    }
    
    save_config(test_config)
    loaded_config = load_config()
    
    assert 'server' in loaded_config
    assert loaded_config['server']['host'] == 'localhost'
    assert loaded_config['server']['port'] == 8080
    assert loaded_config['server']['debug'] is True
    
    assert 'id' in loaded_config
    assert loaded_config['id']['default_strategy'] == 'infrastructure'
    assert loaded_config['id']['default_count'] == 5


def test_config_list_command(temp_config_dir):
    """Test config list command."""
    # Create a test config
    test_config = {
        'server': {
            'host': 'localhost',
            'port': 8080
        }
    }
    save_config(test_config)
    
    # Test text format
    with patch('sys.stdout', new=StringIO()) as fake_out:
        result = cmd_config_list(format_type="text")
        output = fake_out.getvalue()
    
    assert result == 0
    assert '[server]' in output
    assert 'host = localhost' in output
    assert 'port = 8080' in output


def test_config_list_json_format(temp_config_dir):
    """Test config list with JSON format."""
    test_config = {
        'server': {
            'host': 'localhost',
            'port': 8080
        }
    }
    save_config(test_config)
    
    with patch('sys.stdout', new=StringIO()) as fake_out:
        result = cmd_config_list(format_type="json")
        output = fake_out.getvalue()
    
    assert result == 0
    assert '"server"' in output
    assert '"host": "localhost"' in output


def test_config_get_command(temp_config_dir):
    """Test config get command."""
    test_config = {
        'server': {
            'port': 8080
        }
    }
    save_config(test_config)
    
    with patch('sys.stdout', new=StringIO()) as fake_out:
        result = cmd_config_get("server.port")
        output = fake_out.getvalue()
    
    assert result == 0
    assert '8080' in output


def test_config_get_nonexistent_key(temp_config_dir):
    """Test config get with nonexistent key."""
    with patch('sys.stdout', new=StringIO()) as fake_out:
        result = cmd_config_get("nonexistent.key")
        output = fake_out.getvalue()
    
    assert result == 1
    assert 'not found' in output


def test_config_set_command(temp_config_dir):
    """Test config set command."""
    with patch('sys.stdout', new=StringIO()) as fake_out:
        result = cmd_config_set("server.port", "9000")
        output = fake_out.getvalue()
    
    assert result == 0
    assert 'Set server.port = 9000' in output
    
    # Verify the value was saved
    config = load_config()
    assert config['server']['port'] == 9000


def test_config_set_boolean_value(temp_config_dir):
    """Test config set with boolean value."""
    with patch('sys.stdout', new=StringIO()):
        result = cmd_config_set("server.debug", "true")
    
    assert result == 0
    
    config = load_config()
    assert config['server']['debug'] is True


def test_config_set_nested_key(temp_config_dir):
    """Test config set with nested key."""
    with patch('sys.stdout', new=StringIO()):
        result = cmd_config_set("server.logging.level", "debug")
    
    assert result == 0
    
    config = load_config()
    assert 'server' in config
    assert 'logging' in config['server']
    assert config['server']['logging']['level'] == 'debug'


def test_config_reset_without_confirm(temp_config_dir):
    """Test config reset without confirmation."""
    # Create a config file
    save_config({'server': {'port': 8080}})
    
    with patch('sys.stdout', new=StringIO()) as fake_out:
        result = cmd_config_reset(confirm=False)
        output = fake_out.getvalue()
    
    assert result == 1
    assert 'Run with --confirm' in output
    
    # Config should still exist
    config_path = get_config_path()
    assert os.path.exists(config_path)


def test_config_reset_with_confirm(temp_config_dir):
    """Test config reset with confirmation."""
    # Create a config file
    config_path = get_config_path()
    save_config({'server': {'port': 8080}})
    assert os.path.exists(config_path)
    
    with patch('sys.stdout', new=StringIO()) as fake_out:
        result = cmd_config_reset(confirm=True)
        output = fake_out.getvalue()
    
    assert result == 0
    assert 'Configuration reset' in output
    
    # Config should be removed
    assert not os.path.exists(config_path)


def test_config_path_command(temp_config_dir):
    """Test config path command."""
    with patch('sys.stdout', new=StringIO()) as fake_out:
        result = cmd_config_path()
        output = fake_out.getvalue()
    
    assert result == 0
    assert 'config' in output.lower() or 'dredge' in output.lower()


def test_config_path_with_existing_file(temp_config_dir):
    """Test config path when file exists."""
    save_config({'server': {'port': 8080}})
    
    with patch('sys.stdout', new=StringIO()) as fake_out:
        result = cmd_config_path()
        output = fake_out.getvalue()
    
    assert result == 0
    assert 'Configuration file:' in output


def test_config_command_via_main(temp_config_dir):
    """Test config command through main CLI."""
    # Test list
    with patch('sys.stdout', new=StringIO()) as fake_out:
        result = main(['config', 'list'])
        output = fake_out.getvalue()
    
    assert result == 0
    assert 'server' in output or 'No config file found' in output


def test_config_set_via_main(temp_config_dir):
    """Test config set through main CLI."""
    with patch('sys.stdout', new=StringIO()):
        result = main(['config', 'set', 'server.port', '7000'])
    
    assert result == 0
    
    # Verify with get
    with patch('sys.stdout', new=StringIO()) as fake_out:
        result = main(['config', 'get', 'server.port'])
        output = fake_out.getvalue()
    
    assert result == 0
    assert '7000' in output


def test_config_no_action():
    """Test config command without action."""
    with patch('sys.stdout', new=StringIO()) as fake_out:
        result = main(['config'])
        output = fake_out.getvalue()
    
    assert result == 1
    assert 'Usage' in output


def test_load_config_missing_file(temp_config_dir):
    """Test loading config when file doesn't exist."""
    config = load_config()
    assert config == {}


def test_config_list_yaml_format(temp_config_dir):
    """Test config list with YAML format."""
    test_config = {
        'server': {
            'host': 'localhost',
            'port': 8080
        }
    }
    save_config(test_config)
    
    with patch('sys.stdout', new=StringIO()) as fake_out:
        result = cmd_config_list(format_type="yaml")
        output = fake_out.getvalue()
    
    assert result == 0
    assert 'server:' in output
    assert 'host: localhost' in output
    assert 'port: 8080' in output
