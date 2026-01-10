# DREDGE Configuration Schema

Configuration schema for DREDGE CLI and server. Future implementation will support config files.

## Configuration File Locations

DREDGE will look for configuration in these locations (in order):

1. `./dredge.toml` (project-local)
2. `~/.config/dredge/config.toml` (user-level)
3. `~/.dredge/config.toml` (legacy location)
4. Environment variables (highest priority)

## Schema Definition

### Complete TOML Example

```toml
[dredge]
version = "0.1.0"

[server]
host = "0.0.0.0"
port = 3001
debug = false
reload = false

[server.logging]
level = "info"  # debug, info, warning, error, critical
quiet = false
verbose = false

[server.performance]
json_compact = true
hash_strategy = "fast"  # fast, infrastructure
metrics_window = 1000   # sliding window size for metrics

[id]
default_strategy = "fast"  # fast, infrastructure, timestamp, uuid4
default_count = 1

[output]
default_format = "text"  # text, json, yaml, ndjson

[plugins]
discover = true
entry_point = "dredge.plugins"

[diagnostics]
verbose = false
check_dependencies = true
check_ports = true
check_python_version = true
```

### Environment Variables

Environment variables take precedence over config files:

```bash
# Server configuration
DREDGE_HOST=localhost
DREDGE_PORT=8080
DREDGE_DEBUG=true
DREDGE_RELOAD=true

# Logging
DREDGE_LOG_LEVEL=debug
DREDGE_QUIET=false
DREDGE_VERBOSE=true

# Performance
DREDGE_JSON_COMPACT=true
DREDGE_HASH_STRATEGY=infrastructure
DREDGE_METRICS_WINDOW=2000

# ID generation
DREDGE_ID_STRATEGY=fast
DREDGE_ID_COUNT=1

# Output
DREDGE_OUTPUT_FORMAT=json

# Plugins
DREDGE_PLUGINS_DISCOVER=true
```

## Schema Details

### `[server]` Section

Server-related configuration for `dredge serve`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `host` | string | `"0.0.0.0"` | Server bind address |
| `port` | integer | `3001` | Server port |
| `debug` | boolean | `false` | Enable Flask debug mode |
| `reload` | boolean | `false` | Enable hot reload |

**Example:**
```toml
[server]
host = "127.0.0.1"
port = 5000
debug = true
reload = true
```

### `[server.logging]` Section

Logging configuration for server operations.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `level` | string | `"info"` | Log level: debug, info, warning, error, critical |
| `quiet` | boolean | `false` | Only show fatal errors |
| `verbose` | boolean | `false` | Show detailed output |

**Example:**
```toml
[server.logging]
level = "debug"
verbose = true
```

### `[server.performance]` Section

Performance tuning options.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `json_compact` | boolean | `true` | Use compact JSON encoding |
| `hash_strategy` | string | `"fast"` | Hash strategy for IDs: fast, infrastructure |
| `metrics_window` | integer | `1000` | Sliding window size for metrics |

**Example:**
```toml
[server.performance]
json_compact = true
hash_strategy = "infrastructure"
metrics_window = 2000
```

**Identity Contract:**
- `fast`: 64-bit rolling hash (16 hex chars). Suitable for <5B items.
- `infrastructure`: 128-bit BLAKE2b (32 hex chars). Suitable for large-scale infrastructure.

**Collision Behavior:** Last write wins. No defensive checks by default.

### `[id]` Section

ID generation defaults for `dredge id` command.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `default_strategy` | string | `"fast"` | Default ID strategy: fast, infrastructure, timestamp, uuid4 |
| `default_count` | integer | `1` | Default number of IDs to generate |

**Example:**
```toml
[id]
default_strategy = "infrastructure"
default_count = 5
```

**Strategies:**
- `fast`: 64-bit rolling hash (16 hex chars)
- `infrastructure`: 128-bit BLAKE2b (32 hex chars)  
- `timestamp`: Nanosecond precision (20 digits)
- `uuid4`: Standard UUID4 format

### `[output]` Section

Default output format for commands.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `default_format` | string | `"text"` | Default output format: text, json, yaml, ndjson |

**Example:**
```toml
[output]
default_format = "json"
```

### `[plugins]` Section

Plugin system configuration.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `discover` | boolean | `true` | Automatically discover plugins |
| `entry_point` | string | `"dredge.plugins"` | Entry point group for plugins |

**Example:**
```toml
[plugins]
discover = true
entry_point = "dredge.plugins"
```

### `[diagnostics]` Section

Configuration for `dredge doctor` command.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `verbose` | boolean | `false` | Show verbose diagnostics |
| `check_dependencies` | boolean | `true` | Check dependency integrity |
| `check_ports` | boolean | `true` | Check port availability |
| `check_python_version` | boolean | `true` | Verify Python version compatibility |

**Example:**
```toml
[diagnostics]
verbose = true
check_dependencies = true
check_ports = true
check_python_version = true
```

## Configuration Precedence

Configuration is resolved in this order (highest to lowest priority):

1. **Command-line arguments** (e.g., `--port 8080`)
2. **Environment variables** (e.g., `DREDGE_PORT=8080`)
3. **Project config** (`./dredge.toml`)
4. **User config** (`~/.config/dredge/config.toml`)
5. **Built-in defaults**

## Future: Config Commands

Future versions will include config management commands:

```bash
# Get configuration value
dredge config get server.port

# Set configuration value
dredge config set server.port 8080

# List all configuration
dredge config list

# Reset to defaults
dredge config reset

# Show config file location
dredge config path

# Validate configuration
dredge config validate
```

## Creating a Plugin

Plugins extend DREDGE via entry points. Example `pyproject.toml`:

```toml
[project.entry-points."dredge.plugins"]
analytics = "dredge_analytics:AnalyticsPlugin"
```

Plugin class structure:

```python
from dredge.plugin import DredgePlugin

class AnalyticsPlugin(DredgePlugin):
    """Analytics plugin for DREDGE."""
    
    name = "analytics"
    version = "1.0.0"
    description = "Add analytics tracking to DREDGE"
    
    def register_commands(self, cli):
        """Register plugin commands."""
        @cli.command()
        def analytics():
            """Analytics command."""
            click.echo("Analytics tracking enabled")
    
    def on_load(self):
        """Called when plugin is loaded."""
        pass
    
    def on_unload(self):
        """Called when plugin is unloaded."""
        pass
```

## Validation

Configuration values are validated on load:

- **Port**: Must be 1-65535
- **Host**: Must be valid IP or hostname
- **Log level**: Must be valid Python log level
- **Strategies**: Must be supported strategy names
- **Boolean**: Must be true/false
- **Integer**: Must be valid integer in allowed range

Invalid configuration will:
1. Log a warning
2. Fall back to defaults
3. Continue execution (fail-safe)

## Security Considerations

- Config files should not contain secrets
- Use environment variables for sensitive data
- File permissions should restrict access to config files
- Plugin discovery can be disabled if needed

## Example Configurations

### Development Setup
```toml
[server]
host = "localhost"
port = 3001
debug = true
reload = true

[server.logging]
level = "debug"
verbose = true
```

### Production Setup
```toml
[server]
host = "0.0.0.0"
port = 3001
debug = false
reload = false

[server.logging]
level = "warning"
quiet = true

[server.performance]
json_compact = true
hash_strategy = "infrastructure"
metrics_window = 5000
```

### CI/CD Setup
```toml
[server]
host = "0.0.0.0"
port = 3001

[output]
default_format = "json"

[diagnostics]
verbose = true
check_dependencies = true
```
