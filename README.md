# DREDGE

[![DEPENDADREDGEABOT](https://img.shields.io/badge/dependencies-DEPENDADREDGEABOT-blueviolet?style=for-the-badge&logo=dependabot)](https://github.com/QueenFi703/DREDGE-Cli/blob/main/.github/dependabot.yml)
[![DREDGE MCP](https://img.shields.io/badge/GitHub-MCP%20Integration-blue?style=for-the-badge&logo=github-actions)](https://github.com/QueenFi703/DREDGE-Cli/blob/main/docs/GITHUB_MCP_INTEGRATION.md)

DREDGE — small Python package scaffold with String Theory integration.

## 🔮 GitHub MCP Integration

DREDGE now includes GitHub Actions integration where **GitHub events are prompts** and **DREDGE MCP is the responder**. Every push, pull request, or issue comment triggers DREDGE to analyze and respond with contextual insights.

**Key Features:**
- 🤖 **DEPENDADREDGEABOT Integration**: Automatically analyzes dependency updates
- 🔐 **Security Analysis**: Flags security updates for immediate attention  
- 🎯 **Event-Driven**: Responds to pushes, PRs, comments, and manual triggers
- 📊 **Ecosystem-Aware**: Understands Python, Swift, Docker, and GitHub Actions updates

See the [GitHub MCP Integration Guide](docs/GITHUB_MCP_INTEGRATION.md) for detailed documentation.

## 📖 Documentation

- **[BUILD.md](BUILD.md)** - Comprehensive build, test, and development guide
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines
- **[SWIFT_PACKAGE_GUIDE.md](SWIFT_PACKAGE_GUIDE.md)** - Swift development guide
- **[docs/VSCODE_SETUP.md](docs/VSCODE_SETUP.md)** - VS Code setup instructions
- **[docs/CONTAINER_ARCHITECTURE.md](docs/CONTAINER_ARCHITECTURE.md)** - Container architecture and deployment guide
- **[docs/CONTAINER_QUICKSTART.md](docs/CONTAINER_QUICKSTART.md)** - Quick start for container deployment
- **[docs/GITHUB_ACTIONS_CONTAINERS.md](docs/GITHUB_ACTIONS_CONTAINERS.md)** - GitHub Actions workflows for containers

## 🚀 Quick Start

### Local Development (with Makefile)

```bash
# Clone and enter repository
git clone https://github.com/QueenFi703/DREDGE-Cli.git
cd DREDGE-Cli

# Install all dependencies (Python + Swift)
make install-all

# Run DREDGE server (port 3001)
make serve

# Or run MCP server (port 3002)
make mcp

# Run tests
make test-all
```

### Container Development

Pre-built images are available on GitHub Container Registry:

```bash
# Pull and run latest CPU image
docker pull ghcr.io/queenfi703/dredge-cli:latest-cpu
docker run -p 3001:3001 ghcr.io/queenfi703/dredge-cli:latest-cpu

# Pull and run latest GPU image
docker pull ghcr.io/queenfi703/dredge-cli:latest-gpu
docker run -p 3002:3002 --gpus all ghcr.io/queenfi703/dredge-cli:latest-gpu
```

Or build and run locally:

```bash
# CPU-only Flask server
make docker-up-cpu

# GPU-enabled MCP server
make docker-up-gpu

# Full stack with monitoring
make docker-profile-full
```

See **[docs/CONTAINER_QUICKSTART.md](docs/CONTAINER_QUICKSTART.md)** for more container deployment options.

See **[BUILD.md](BUILD.md)** for complete build instructions, CI triggers, and troubleshooting.

## 🚀 Quick Start with VS Code

**Clone this repository directly into VS Code!**

See the detailed [VS Code Setup Guide](docs/VSCODE_SETUP.md) for complete instructions on cloning, setting up, and developing in VS Code.

**Quick clone command:**
```bash
git clone https://github.com/QueenFi703/DREDGE-Cli.git
code DREDGE-Cli
```

## Repository Structure

- **src/dredge/** - Python package source code
- **tests/** - Test files
- **docs/** - Documentation files (see [docs/mobile-optimization.md](docs/mobile-optimization.md) for mobile guidance)
- **benchmarks/** - Benchmark scripts and results
- **swift/** - Swift implementation files
- **DREDGE-Cli.xcworkspace** - Xcode workspace for Swift development
- **archives/** - Archived files (excluded from version control)

## Install

Create a virtual environment and install:

python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e .

## Server Usage

DREDGE includes two web servers:

### 1. DREDGE x Dolly Server (Port 3001)

A web server for API-based interaction with Dolly integration.

#### Starting the Server

```bash
python -m dredge serve
# or
dredge-cli serve --host 0.0.0.0 --port 3001 --debug
```

#### API Endpoints

- **GET /** - API information and available endpoints
- **GET /health** - Health check endpoint
- **POST /lift** - Lift an insight with Dolly integration
- **GET /quasimoto-gpu** - Quasimoto GPU visualization (repository language statistics)

#### Example Usage

```bash
# Get API info
curl http://localhost:3001/

# Check health
curl http://localhost:3001/health

# Lift an insight
curl -X POST http://localhost:3001/lift \
  -H "Content-Type: application/json" \
  -d '{"insight_text": "Digital memory must be human-reachable."}'

# View Quasimoto GPU visualization
open http://localhost:3001/quasimoto-gpu
```

### 2. MCP Server (Port 3002) - Quasimoto Integration

A Model Context Protocol server for serving Quasimoto neural wave function models with String Theory integration.

#### Starting the MCP Server

```bash
python -m dredge mcp
# or
dredge-cli mcp --host 0.0.0.0 --port 3002 --debug
```

#### MCP Protocol Endpoint

- **GET /** - MCP server capabilities and model information
- **POST /mcp** - MCP protocol endpoint for model operations

#### Available Operations

1. **list_capabilities** - List available models and operations
2. **load_model** - Load Quasimoto models (1D, 4D, 6D, ensemble) or String Theory models
3. **inference** - Run inference on loaded models
4. **get_parameters** - Retrieve model parameters
5. **benchmark** - Run performance benchmarks
6. **string_spectrum** - Compute string theory vibrational spectrum
7. **string_parameters** - Calculate fundamental string theory parameters
8. **unified_inference** - Run unified DREDGE + Quasimoto + String Theory inference
9. **get_dependabot_alerts** - Retrieve Dependabot security alerts for a repository
10. **explain_dependabot_alert** - Get detailed explanation of a specific Dependabot alert
11. **update_dependabot_alert** - Update Dependabot alert status (dismiss or reopen)

#### Example MCP Request

```bash
# List capabilities
curl http://localhost:3002/

# Load a model
curl -X POST http://localhost:3002/mcp \
  -H "Content-Type: application/json" \
  -d '{"operation": "load_model", "params": {"model_type": "quasimoto_1d"}}'

# Run inference
curl -X POST http://localhost:3002/mcp \
  -H "Content-Type: application/json" \
  -d '{"operation": "inference", "params": {"model_id": "quasimoto_1d_0", "inputs": {"x": [0.5], "t": [0.0]}}}'
```

#### Available Models

- **quasimoto_1d** - 1D wave function (8 parameters)
- **quasimoto_4d** - 4D spatiotemporal wave function (13 parameters)
- **quasimoto_6d** - 6D high-dimensional wave function (17 parameters)
- **quasimoto_ensemble** - Configurable ensemble models
- **string_theory** - String theory neural network (configurable dimensions)

#### String Theory Integration

The MCP server now includes string theory models that integrate with Quasimoto wave functions:

```bash
# Compute string vibrational spectrum
curl -X POST http://localhost:3002/mcp \
  -H "Content-Type: application/json" \
  -d '{"operation": "string_spectrum", "params": {"max_modes": 10, "dimensions": 10}}'

# Calculate fundamental string parameters
curl -X POST http://localhost:3002/mcp \
  -H "Content-Type: application/json" \
  -d '{"operation": "string_parameters", "params": {"energy_scale": 1.0, "coupling_constant": 0.1}}'

# Run unified inference (DREDGE + Quasimoto + String Theory)
curl -X POST http://localhost:3002/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "unified_inference",
    "params": {
      "dredge_insight": "Digital memory must be human-reachable",
      "quasimoto_coords": [0.5, 0.5, 0.5],
      "string_modes": [1, 2, 3]
    }
  }'

# Load a string theory model
curl -X POST http://localhost:3002/mcp \
  -H "Content-Type: application/json" \
  -d '{"operation": "load_model", "params": {"model_type": "string_theory", "config": {"dimensions": 10, "hidden_size": 64}}}'
```

#### Dependabot Alert Management

The MCP server now includes Dependabot alert integration for conversational dependency management:

```bash
# Get all Dependabot alerts for a repository
curl -X POST http://localhost:3002/mcp \
  -H "Content-Type: application/json" \
  -d '{"operation": "get_dependabot_alerts", "params": {"repo_owner": "QueenFi703", "repo_name": "DREDGE-Cli"}}'

# Explain a specific alert with AI-powered recommendations
curl -X POST http://localhost:3002/mcp \
  -H "Content-Type: application/json" \
  -d '{"operation": "explain_dependabot_alert", "params": {"alert_id": 1, "repo_owner": "QueenFi703", "repo_name": "DREDGE-Cli"}}'

# Update an alert status (dismiss or reopen)
curl -X POST http://localhost:3002/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "update_dependabot_alert",
    "params": {
      "alert_id": 1,
      "state": "dismissed",
      "dismissed_reason": "not_used",
      "dismissed_comment": "This dependency is not used in production"
    }
  }'
```

**Note:** Dependabot operations require a `GITHUB_TOKEN` environment variable with the `security_events` scope.

**Available dismissed reasons:**
- `fix_started` - A fix has already been started
- `inaccurate` - This alert is inaccurate or incorrect
- `no_bandwidth` - No bandwidth to fix this
- `not_used` - Dependency is not used
- `tolerable_risk` - Risk is tolerable

### GitHub Codespaces

The repository includes `.devcontainer/devcontainer.json` configured to:
- Automatically forward ports 3001 and 3002 when running in GitHub Codespaces
- Fetch the full git repository history (unshallow the repository) for complete commit access

## Swift Development

DREDGE includes a Swift CLI implementation with MCP client and String Theory support. You can develop using:

### Xcode Workspace
```bash
open DREDGE-Cli.xcworkspace
```

### Swift Package Manager
```bash
# Build from root
swift build
swift run dredge-cli

# Or build from swift/ directory
cd swift
swift build
swift run dredge-cli
```

### Swift Features

The Swift implementation includes:
- **String Theory Models** - 10D superstring vibrational modes and energy calculations
- **MCP Client** - Connect to MCP server for model operations
- **Unified Integration** - Combine DREDGE insights, Quasimoto coordinates, and String Theory modes

See [SWIFT_PACKAGE_GUIDE.md](SWIFT_PACKAGE_GUIDE.md) for detailed Swift development information.

## GitHub Event Processing

DREDGE CLI can process GitHub events as part of the MCP integration:

```bash
# Process a GitHub event
dredge-cli github-event \
  --event "pull_request" \
  --payload '{"action": "opened", "pull_request": {...}}' \
  --ref "refs/pull/42/merge" \
  --repo "owner/repo" \
  --sha "abc123"

# Get help
dredge-cli github-event --help
```

This command is used by the [GitHub Actions workflow](docs/GITHUB_MCP_INTEGRATION.md) to automatically respond to repository events.

## Test

Run tests with pytest:

pip install -U pytest
pytest

## Development

- Edit code in src/dredge
- Update version in pyproject.toml
- Tag releases with v<version> and push tags