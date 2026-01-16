# Installation Guide

## Python Installation

### Using pip (from PyPI)
```bash
pip install dredge-cli
```

### From Source
```bash
git clone https://github.com/QueenFi703/DREDGE-Cli.git
cd DREDGE-Cli
pip install -e .
```

### Development Installation
```bash
git clone https://github.com/QueenFi703/DREDGE-Cli.git
cd DREDGE-Cli
pip install -r requirements.txt
pip install -e .
```

## Swift Installation

### Building from Source
```bash
# From repository root
swift build

# Run CLI
.build/debug/dredge-cli
```

### From swift/ Subdirectory
```bash
cd swift
swift build
swift run dredge-cli
```

### Running Tests
```bash
# From root
swift test

# From swift/ directory
cd swift && swift test
```

## Docker Installation

### Using Docker
```bash
docker build -t dredge-cli .
docker run -p 3001:3001 dredge-cli serve
```

### Using Docker Compose
```bash
docker-compose up
```

This starts both the DREDGE server (port 3001) and MCP server (port 3002).

## Verifying Installation

### Python
```bash
dredge-cli --version
# Should output: 0.1.4

dredge-cli --help
# Should show available commands
```

### Swift
```bash
swift run dredge-cli
# Should output: DREDGE-Cli v0.1.0
```

## Troubleshooting

### "Command not found: dredge-cli"
Ensure your Python scripts directory is in your PATH:
- **Linux**: Add `~/.local/bin` to PATH
- **macOS**: Add `~/Library/Python/3.x/bin` to PATH

Add to your shell profile (~/.bashrc, ~/.zshrc):
```bash
export PATH="$HOME/.local/bin:$PATH"
```

### Swift build fails
Ensure you have Xcode Command Line Tools installed:
```bash
xcode-select --install
```

### Python tests fail
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
pip install pytest
```

### Port already in use
If port 3001 or 3002 is already in use, specify a different port:
```bash
dredge-cli serve --port 3003
```

## Requirements

### Python
- Python 3.9 or higher
- Flask
- PyTorch (for Quasimoto benchmarks)
- NumPy
- Matplotlib (for visualizations)

### Swift
- Swift 5.9 or higher
- macOS 12.0+ (for macOS targets)

## Next Steps
After installation, see:
- <a>QUICKSTART.md</a> - Get started in 5 minutes
- <a>API_REFERENCE.md</a> - Complete API documentation
- <a>FULL_DOCUMENTATION.md</a> - Architecture details
