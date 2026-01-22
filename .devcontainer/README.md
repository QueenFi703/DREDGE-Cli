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
- **3002** - DREDGE MCP Server (Quasimoto + Dependabot Integration)

## Environment Variables

### GitHub Token for Dependabot Integration

The MCP server now includes Dependabot alerts integration. To use these features, you need to configure a GitHub token:

1. **Create a GitHub Personal Access Token** (classic or fine-grained):
   - Classic: Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
   - Fine-grained: Go to GitHub Settings → Developer settings → Personal access tokens → Fine-grained tokens
   
2. **Required Permissions**:
   - Classic token: Select `security_events` scope (read:org, read:repo_security_events)
   - Fine-grained token: Grant "Dependabot alerts" read access
   
3. **Configure in Codespaces**:
   ```bash
   # Set as a Codespace secret (recommended for persistent access)
   # Go to: Settings → Codespaces → Secrets → New secret
   # Name: GITHUB_TOKEN
   # Value: your_token_here
   
   # Or set temporarily in your terminal session:
   export GITHUB_TOKEN=your_token_here
   ```

4. **Configure Locally**:
   ```bash
   # Add to your shell profile (~/.bashrc, ~/.zshrc, etc.):
   export GITHUB_TOKEN=your_token_here
   
   # Or create a .env file (never commit this file!)
   echo "GITHUB_TOKEN=your_token_here" > .env
   ```

**Dependabot MCP Operations** (requires GITHUB_TOKEN):
- `get_dependabot_alerts` - List all security alerts
- `explain_dependabot_alert` - Get vulnerability details with AI recommendations
- `update_dependabot_alert` - Dismiss or reopen alerts

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

4. **(Optional) Configure Dependabot Integration**:
   - Set up `GITHUB_TOKEN` as described in the Environment Variables section above
   - Required for MCP Dependabot operations: querying alerts, explaining vulnerabilities, updating alert statuses
   - Without token, all other MCP features (Quasimoto models, String Theory) work normally

The postCreateCommand runs as a single bash command to ensure all setup steps complete in sequence. If you prefer, you can break this into a separate script by creating `.devcontainer/setup.sh`, but the current approach keeps the configuration self-contained.

## Troubleshooting

### Dependabot Operations Return "GITHUB_TOKEN not set"

If you see this error when using MCP Dependabot operations:
```json
{"success": false, "error": "GITHUB_TOKEN not set"}
```

**Solutions:**
1. **Codespaces**: Add GITHUB_TOKEN as a Codespace secret (Settings → Codespaces → Secrets)
2. **Local container**: Export the token before starting VS Code: `export GITHUB_TOKEN=your_token`
3. **Verify token**: Run `echo $GITHUB_TOKEN` in the terminal to confirm it's set
4. **Check permissions**: Ensure token has `security_events` scope (read:org, read:repo_security_events)

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
