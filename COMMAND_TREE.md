# DREDGE Command Tree

Complete reference for all DREDGE CLI commands and their options.

```
dredge
├── --version                    # Show version number
├── --help                       # Show help message
│
├── print [TEXT]                 # Print text or clean newline
│   ├── --format [text|json|yaml|ndjson]
│   └── [TEXT]                   # Optional text to print
│
├── serve                        # Start the DREDGE x Dolly server
│   ├── --host HOST              # Server host (default: 0.0.0.0)
│   ├── --port PORT              # Server port (default: 3001)
│   ├── --debug                  # Enable debug mode
│   ├── --quiet                  # Only show fatal errors
│   ├── --verbose                # Show detailed output
│   └── --reload                 # Watch files and auto-reload
│
├── inspect                      # Peer into DREDGE configuration
│   └── --format [text|json|yaml|ndjson]
│
├── doctor                       # Run system diagnostics
│   ├── --format [text|json|yaml|ndjson]
│   └── --verbose                # Show detailed diagnostics
│
├── echo                         # Signature touch - prints "alive"
│
├── id                           # Generate unique identifiers
│   ├── --count N                # Generate N IDs (default: 1)
│   ├── --strategy STRATEGY      # ID generation strategy
│   │   ├── fast                 # 64-bit rolling hash (default)
│   │   ├── infrastructure       # 128-bit BLAKE2b hash
│   │   ├── timestamp            # Nanosecond precision timestamp
│   │   └── uuid4                # Standard UUID4
│   └── --format [text|json|yaml|ndjson]
│
├── time                         # Display current time with precision
│   └── --format [text|json|yaml|ndjson|unix|unix_ms|unix_ns|iso]
│
└── plugin                       # Plugin system management
    ├── list                     # List installed plugins
    └── info PLUGIN_NAME         # Show plugin information
```

## Global Options

- `--version`: Show DREDGE version and exit
- `--help`: Show help message and exit

## Format Options

Many commands support `--format` flag with these values:

- **text**: Human-readable output (default)
- **json**: Structured JSON for machine parsing
- **yaml**: YAML format for configuration
- **ndjson**: Newline-delimited JSON for streaming

## Verbosity Modes

Server and diagnostic commands support verbosity control:

- **--quiet**: Only fatal errors (minimal output)
- **--verbose**: Detailed output with timings and decisions
- **--debug**: Full debug mode (most verbose)

## Examples

### Basic Commands
```bash
# Print message
dredge print "Hello, World!"

# Print clean newline
dredge print

# Check if alive
dredge echo
```

### Server Operations
```bash
# Start server (development)
dredge serve --reload --verbose

# Start server (production)
dredge serve --quiet

# Custom host and port
dredge serve --host localhost --port 8080
```

### Diagnostics
```bash
# Quick health check
dredge doctor

# Detailed diagnostics
dredge doctor --verbose

# Export diagnostics as JSON
dredge doctor --format json > diagnostics.json
```

### ID Generation
```bash
# Single ID (fast strategy)
dredge id

# Multiple IDs
dredge id --count 10

# Infrastructure-scale IDs (128-bit)
dredge id --strategy infrastructure --count 5

# Timestamp-based IDs
dredge id --strategy timestamp
```

### Time Operations
```bash
# Human-readable time
dredge time

# Unix timestamp (seconds)
dredge time --format unix

# ISO 8601 format
dredge time --format iso

# Complete time data as JSON
dredge time --format json
```

### Configuration Inspection
```bash
# View configuration
dredge inspect

# Export as YAML
dredge inspect --format yaml > config.yaml

# Machine-readable JSON
dredge inspect --format json | jq .
```

### Plugin Management
```bash
# List installed plugins
dredge plugin list

# Get plugin information
dredge plugin info dredge-analytics
```

## Philosophy

DREDGE commands follow these principles:

1. **Power Without Weight**: Commands are lightweight but powerful
2. **Output as First-Class**: Multiple output formats for different contexts
3. **Platform Thinking**: Server becomes a platform with observability
4. **Time & Identity**: Precise tracking and flexible ID generation
5. **Extensibility**: Plugin system allows ecosystem growth

## API Endpoints

When running `dredge serve`, these HTTP endpoints are available:

- **GET /** - API information and version
- **GET /health** - Health status with uptime and request metrics
- **GET /metrics** - Performance metrics (latency, request counts)
- **POST /lift** - Lift insights with Dolly integration

## Status Indicators

- ✓ Command succeeded
- ⚠ Warning (non-fatal issue detected)
- ✗ Command failed
- 🔧 Diagnostic information
- 🔥 Hot reload triggered
