# DREDGE DevContainer Configuration

This directory contains the development container configuration for GitHub Codespaces and VS Code Remote Containers.

## Overview

The devcontainer is configured to provide a consistent development environment with:

- **Python 3.11** - Primary development language
- **Development Tools** - Black, Ruff, MyPy, Pytest, Pre-commit
- **VS Code Extensions** - Python, Jupyter, Git, GitHub Copilot support

### Python Interpreter

The Python interpreter path differs between environments by design:

- **Local VS Code**: `${workspaceFolder}/.venv/bin/python` (virtual environment)
- **DevContainer**: `/usr/local/bin/python` (system Python in container)

This is intentional because:
- Local development uses isolated virtual environments
- Containers are already isolated, so dependencies are installed system-wide
- Both approaches are valid and provide dependency isolation

When switching between environments, VS Code will automatically detect and use the correct Python interpreter for each context.

## Language Support

### Python (Fully Supported)
The container is fully configured for Python development with:
- Python 3.11 runtime
- All Python dependencies from `requirements.txt`
- Testing with pytest
- Formatting with Black
- Linting with Ruff and MyPy
- Pre-commit hooks

### Swift (Editor Support Only)
Swift extensions are included for viewing and editing Swift code, but the container does not include the Swift runtime. For full Swift development (building, testing, running):

- Use local development on macOS with Xcode
- Use the native Swift Package Manager workflow
- See [SWIFT_PACKAGE_GUIDE.md](../SWIFT_PACKAGE_GUIDE.md) for details

The Swift extensions will gracefully degrade in the container environment and allow you to:
- View and edit Swift code with syntax highlighting
- Browse the Swift package structure
- Make changes to Swift files

## Port Forwarding

The container forwards two ports:

- **3001** - DREDGE x Dolly Server
- **3002** - DREDGE MCP Server (Quasimoto)

## Customization

The devcontainer configuration is synchronized with the local VS Code settings in `.vscode/` to ensure a consistent experience whether you're developing locally or in Codespaces.

### Extensions

All recommended extensions from `.vscode/extensions.json` are automatically installed in the container.

### Settings

Python and editor settings are synchronized between local VS Code and Codespaces for consistency.

## Getting Started

1. Open this repository in GitHub Codespaces or VS Code with Remote Containers
2. Wait for the container to build and the postCreateCommand to complete
3. The container will automatically:
   - Unshallow git history if needed
   - Install Python dependencies from `requirements.txt`
   - Install the package in editable mode
   - Install development tools (black, ruff, mypy, pytest, pre-commit)
   - Check for Swift availability and notify if unavailable

The postCreateCommand runs as a single bash command to ensure all setup steps complete in sequence. If you prefer, you can break this into a separate script by creating `.devcontainer/setup.sh`, but the current approach keeps the configuration self-contained.

## Troubleshooting

### Swift Extensions Show Errors
This is expected. Swift extensions require the Swift runtime which is not available in the Python-based container. You can safely ignore these warnings or disable the Swift extensions in Codespaces if preferred.

### Python Dependencies Not Found
If dependencies are missing, run:
```bash
pip install -r requirements.txt
pip install -e .
pip install black ruff mypy pytest pre-commit
```

### Port Already in Use
If ports 3001 or 3002 are already in use, update the `forwardPorts` setting in `devcontainer.json`.
