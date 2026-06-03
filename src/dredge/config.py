"""
Configuration management for DREDGE.
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional


DEFAULT_CONFIG = {
    "server": {
        "host": "0.0.0.0",
        "port": 3000,
        "debug": False,
        "threads": 1,
    },
    "mcp": {
        "host": "0.0.0.0",
        "port": 3002,
        "debug": False,
        "device": "auto",
        "threads": 1,
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    },
}


def get_config_path() -> Path:
    """Get the path to the configuration file."""
    # Check for config in current directory first
    local_config = Path.cwd() / ".dredge.json"
    if local_config.exists():
        return local_config
    
    # Check for config in home directory
    home_config = Path.home() / ".dredge.json"
    if home_config.exists():
        return home_config
    
    # Default to local directory
    return local_config


def load_config() -> Dict[str, Any]:
    """Load configuration from file or return defaults."""
    config_path = get_config_path()
    
    if not config_path.exists():
        return DEFAULT_CONFIG.copy()
    
    try:
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        
        # Merge with defaults
        config = DEFAULT_CONFIG.copy()
        for key in user_config:
            if key in config and isinstance(config[key], dict):
                config[key].update(user_config[key])
            else:
                config[key] = user_config[key]
        
        return config
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Failed to load config from {config_path}: {e}")
        return DEFAULT_CONFIG.copy()


def save_config(config: Dict[str, Any], path: Optional[Path] = None) -> None:
    """Save configuration to file."""
    if path is None:
        path = get_config_path()
    
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)


def init_config(path: Optional[Path] = None) -> Path:
    """Initialize a new configuration file with defaults."""
    if path is None:
        path = get_config_path()
    
    if path.exists():
        raise FileExistsError(
            f"Configuration file already exists at {path}. "
            f"Use 'dredge-cli config show' to view it or remove the file first."
        )
    
    save_config(DEFAULT_CONFIG, path)
    return path


def get_server_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get server configuration."""
    if config is None:
        config = load_config()
    return config.get("server", DEFAULT_CONFIG["server"])


def get_mcp_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get MCP server configuration."""
    if config is None:
        config = load_config()
    return config.get("mcp", DEFAULT_CONFIG["mcp"])


class DREDGEConfig:
    """Mutable configuration wrapper used by the interactive API."""

    def __init__(self, path: Optional[str] = None):
        self.path = Path(path) if path else get_config_path()
        self.data = load_config() if not path else self._load_from_path(self.path)

    def _load_from_path(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return DEFAULT_CONFIG.copy()

        try:
            with open(path, "r") as f:
                user_config = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Failed to load config from {path}: {e}")
            return DEFAULT_CONFIG.copy()

        config = DEFAULT_CONFIG.copy()
        for key, value in user_config.items():
            if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                config[key].update(value)
            else:
                config[key] = value
        return config

    def update(self, category: str, key: str, value: Any, validate: bool = True) -> Dict[str, Any]:
        if validate and not category:
            raise ValueError("category is required")
        if validate and not key:
            raise ValueError("key is required")

        section = self.data.setdefault(category, {})
        if not isinstance(section, dict):
            raise ValueError(f"Configuration category '{category}' is not an object")

        old_value = section.get(key)
        section[key] = value
        save_config(self.data, self.path)
        return {
            "category": category,
            "key": key,
            "old_value": old_value,
            "new_value": value,
            "path": str(self.path),
        }

    def to_dict(self) -> Dict[str, Any]:
        return self.data

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                category: {
                    "type": "object",
                    "properties": {
                        key: {"type": type(value).__name__}
                        for key, value in values.items()
                    },
                }
                for category, values in DEFAULT_CONFIG.items()
            },
        }
