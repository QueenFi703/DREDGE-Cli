# BUILD.md

> **Single source of truth for building, testing, and running DREDGE locally or in containers.**

## 🚀 Quickstart Matrix

| Environment | Use Case | Prerequisites | Command |
|-------------|----------|---------------|---------|
| **Local Native (Python)** | Fast iteration on Python code, API servers | Python 3.9-3.11, pip | `make install-python && make serve` |
| **Local Native (Swift)** | Swift CLI development, MCP client | Swift 5.9+, macOS/Linux | `make build-swift && make run-swift` |
| **Containerized (CPU)** | Production Flask server, no GPU | Docker, Docker Compose | `make docker-up-cpu` |
| **Containerized (GPU)** | MCP server with PyTorch/CUDA | Docker + nvidia-docker, NVIDIA GPU | `make docker-up-gpu` |

---

## 📦 Dependency Installation

### Python Dependencies

**Primary method: pip with `requirements.txt`**

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Optional: Development tools
pip install pytest pytest-cov black ruff mypy
```

**Dependencies:**
- `flask>=3.0.0` - Web server for DREDGE x Dolly API
- `torch>=2.0.0` - PyTorch for Quasimoto neural models
- `numpy>=1.24.0` - Numerical operations
- `matplotlib>=3.5.0` - Visualization

**Platform-specific PyTorch:**
- **CUDA (Linux + NVIDIA GPU):** `pip install torch --index-url https://download.pytorch.org/whl/cu118`
- **MPS (macOS Apple Silicon):** `pip install torch` (MPS included in standard wheels, requires macOS 12.3+)
- **CPU-only:** `pip install torch` (auto-detects)

### Swift Dependencies

**Primary method: Swift Package Manager**

```bash
# Resolve dependencies (if any external packages exist)
swift package resolve

# Build
swift build

# Or use Xcode
open DREDGE-Cli.xcworkspace
```

**Swift version required:** 5.9+

### Docker

**Optional containerized path:**

```bash
# CPU-only build (Flask server)
docker compose up dredge-server

# GPU build (MCP server with CUDA)
docker compose up quasimoto-mcp

# Or use Makefile
make docker-build-cpu
make docker-build-gpu
```

---

## 🔧 Build Commands

### Using Makefile (Recommended)

```bash
# Install all dependencies
make install-all

# Python only
make install-python

# Swift only
make build-swift

# Run Python server (port 3001)
make serve

# Run MCP server (port 3002)
make mcp

# Run Swift CLI
make run-swift

# Run tests
make test-all          # Both Python and Swift
make test-python       # Python only
make test-swift        # Swift only

# Lint and format
make lint-all          # Run all linters
make format-all        # Format all code

# Docker
make docker-build-cpu  # Build CPU image
make docker-build-gpu  # Build GPU image
make docker-up-cpu     # Start CPU container
make docker-up-gpu     # Start GPU container
make docker-down       # Stop all containers
```

### Manual Build Commands

#### Python

```bash
# Install in development mode
pip install -e .

# Run CLI
dredge-cli --help
dredge-cli --version
dredge-cli --version-info       # Detailed version and system info

# Health and diagnostics
dredge-cli health               # Check system health and dependencies
dredge-cli health --json        # JSON output for scripts
dredge-cli info                 # Show system information

# Configuration management
dredge-cli config init          # Create default config file (.dredge.json)
dredge-cli config show          # Show current configuration
dredge-cli config path          # Show config file path

# Servers
dredge-cli serve --host 0.0.0.0 --port 3001
dredge-cli mcp --host 0.0.0.0 --port 3002

# Or use module syntax
python -m dredge serve
python -m dredge mcp

# Test
pytest tests/ -v
pytest tests/ -v --cov=src/dredge --cov-report=term-missing

# Lint and format
black src/ tests/
ruff check src/ tests/
mypy src/
```

#### Swift

```bash
# Build
swift build --configuration release

# Run CLI
swift run dredge-cli

# Or run built executable
.build/release/dredge-cli

# Test
swift test

# From swift/ directory
cd swift
swift build
swift run dredge-cli
```

---

## ⚙️ Configuration

DREDGE supports configuration via `.dredge.json` file for persistent settings.

### Create Configuration File

```bash
# Initialize with defaults
dredge-cli config init

# Show current config file path
dredge-cli config path

# View configuration
dredge-cli config show
```

### Configuration File Format

The `.dredge.json` file can be placed in:
1. Current directory (`./.dredge.json`) - takes precedence
2. Home directory (`~/.dredge.json`) - fallback

**Example configuration:**

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 3001,
    "debug": false,
    "threads": 1
  },
  "mcp": {
    "host": "0.0.0.0",
    "port": 3002,
    "debug": false,
    "device": "auto",
    "threads": 1
  },
  "logging": {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  }
}
```

**Note:** Command-line arguments override configuration file settings.

---

## 🔍 CI/CD Workflows

### Workflow Files

Located in `.github/workflows/`:

1. **`ci-python.yml`** - Python CI pipeline
2. **`ci-swift.yml`** - Swift CI pipeline
3. **`release.yml`** - Release automation

### CI Triggers

**Python CI (`ci-python.yml`):**
- **Triggers:** Push to any branch, pull requests to any branch
- **Matrix:** Python 3.9, 3.10, 3.11 on `ubuntu-latest`
- **Steps:**
  1. Checkout code
  2. Setup Python
  3. Cache pip packages (`~/.cache/pip`)
  4. Install dependencies (`requirements.txt`, `pytest`, `pytest-cov`)
  5. Run tests with coverage (`pytest tests/ -v --cov=src/dredge`)
  6. Test CLI commands (`dredge-cli --version`, `dredge-cli --help`)

**Swift CI (`ci-swift.yml`):**
- **Triggers:** Push to any branch, pull requests to any branch
- **Runner:** `macos-latest`
- **Steps:**
  1. Checkout code
  2. Show Swift version
  3. Cache SwiftPM artifacts (`.build`, `.swiftpm`)
  4. Resolve dependencies (`swift package resolve`)
  5. Build release configuration (`swift build --configuration release`)
  6. Run tests (`swift test`)

### Workflow Generation

**Status:** Workflows are **manually maintained** YAML files (not generated).

- No code generation tools (e.g., Pkl) are currently used
- YAMLs are minimal and focused on essential CI steps
- Caching is used for pip packages and SwiftPM artifacts to speed up builds

### Required Tools

| Tool | Purpose | Install |
|------|---------|---------|
| **Python 3.9-3.11** | Python runtime | [python.org](https://python.org) |
| **pip** | Python package manager | Included with Python |
| **Swift 5.9+** | Swift compiler | [swift.org](https://swift.org) or Xcode |
| **Docker** | Container runtime | [docker.com](https://docker.com) |
| **nvidia-docker** | GPU support (optional) | [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) |
| **pytest** | Python testing | `pip install pytest` |
| **black** | Python formatter | `pip install black` |
| **ruff** | Python linter | `pip install ruff` |
| **SwiftFormat** | Swift formatter (optional) | [SwiftFormat](https://github.com/nicklockwood/SwiftFormat) |
| **SwiftLint** | Swift linter (optional) | [SwiftLint](https://github.com/realm/SwiftLint) |

---

## 🪝 Pre-commit Hooks

Pre-commit hooks ensure code quality before commits. See [.pre-commit-config.yaml](.pre-commit-config.yaml).

### Installation

```bash
# Install pre-commit tool
pip install pre-commit

# Install hooks
make install-hooks

# Or manually
pre-commit install
```

### Hooks Configured

**Python:**
- **black** - Code formatter (line length 88)
- **ruff** - Fast linter (replaces flake8, isort, etc.)

**Swift:**
- **SwiftFormat** - Code formatter (if installed)
- **SwiftLint** - Linter (if installed)

**General:**
- **trailing-whitespace** - Remove trailing whitespace
- **end-of-file-fixer** - Ensure newline at EOF
- **check-yaml** - Validate YAML syntax
- **check-added-large-files** - Prevent large file commits

### Manual Hook Execution

```bash
# Run on all files
pre-commit run --all-files

# Run on staged files only
pre-commit run
```

---

## 🐛 Troubleshooting

### Python Issues

**Import errors after install:**
```bash
# Ensure virtual environment is activated
source .venv/bin/activate
pip install -e .
```

**PyTorch CUDA not available:**
```bash
# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall with CUDA support
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**MPS (Apple Silicon) not available:**
```bash
# Check MPS support (requires macOS 12.3+)
python -c "import torch; print(torch.backends.mps.is_available())"
```

### Swift Issues

**Swift version too old:**
```bash
swift --version  # Should be 5.9+
# Update via Xcode or swift.org
```

**Build failures:**
```bash
# Clean build artifacts
swift package clean
rm -rf .build

# Rebuild
swift build
```

### Docker Issues

**GPU not accessible in container:**
```bash
# Verify nvidia-docker is installed
docker run --rm --gpus all nvidia/cuda:11.8.0-runtime-ubuntu22.04 nvidia-smi

# Check compose GPU configuration
docker compose config
```

**Port conflicts:**
```bash
# Ports 3001 and 3002 must be free
lsof -i :3001
lsof -i :3002

# Stop conflicting services or change ports in docker-compose.yml
```

---

## 📚 Additional Resources

- **[README.md](README.md)** - Project overview, quick start, server usage
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines, code style, PR process
- **[SWIFT_PACKAGE_GUIDE.md](SWIFT_PACKAGE_GUIDE.md)** - Detailed Swift development guide
- **[docs/VSCODE_SETUP.md](docs/VSCODE_SETUP.md)** - VS Code configuration
- **[docs/BENCHMARK_USAGE.md](docs/BENCHMARK_USAGE.md)** - Benchmarking Quasimoto models

---

## 🔗 Quick Links

- **GitHub Repository:** [QueenFi703/DREDGE-Cli](https://github.com/QueenFi703/DREDGE-Cli)
- **Issue Tracker:** [GitHub Issues](https://github.com/QueenFi703/DREDGE-Cli/issues)
- **CI Status:** Check `.github/workflows/` for latest runs
