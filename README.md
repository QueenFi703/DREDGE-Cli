# DREDGE

DREDGE — small Python package scaffold.

## Install

Create a virtual environment and install:

python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e .

## Server Usage

DREDGE x Dolly now includes a web server for API-based interaction. The server is configured to run on **port 3001** by default, making it perfect for GitHub Codespaces.

### Starting the Server

Run the server with:

```bash
python -m dredge serve
```

Or with custom options:

```bash
python -m dredge serve --host 0.0.0.0 --port 3001 --debug
```

### API Endpoints

Once the server is running, you can interact with the following endpoints:

- **GET /** - API information and available endpoints
- **GET /health** - Health check endpoint
- **POST /lift** - Lift an insight with Dolly integration

Example usage:

```bash
# Get API info
curl http://localhost:3001/

# Check health
curl http://localhost:3001/health

# Lift an insight
curl -X POST http://localhost:3001/lift \
  -H "Content-Type: application/json" \
  -d '{"insight_text": "Digital memory must be human-reachable."}'
```

### GitHub Codespaces

The repository includes `.devcontainer/devcontainer.json` configured to automatically forward port 3001 when running in GitHub Codespaces. Simply start the server and the port will be accessible.

## Test

Run tests with pytest:

pip install -U pytest
pytest

## Development

- Edit code in src/dredge
- Update version in pyproject.toml
- Tag releases with v<version> and push tags

## Release Workflow

The repository includes a GitHub Actions workflow (`.github/workflows/release.yml`) that automatically publishes to PyPI when version tags are pushed.

### Triggering a Release

To trigger the release workflow:

1. Update the version in `pyproject.toml`
2. Create and push a version tag:
   ```bash
   git tag v0.1.5
   git push origin v0.1.5
   ```

The workflow is configured to trigger on tags matching the pattern `v*` (e.g., `v0.1.0`, `v1.2.3`).

### GitHub Actions Tag Pattern

**Important:** In GitHub Actions `push.tags` triggers, use the shorthand pattern without the `refs/tags/` prefix:

```yaml
on:
  push:
    tags:
      - "v*"  # Correct: matches tags like v1.0.0, v2.3.4
```

Do **not** use:
```yaml
tags:
  - "refs/tags/v*"  # Incorrect: will not trigger on tag pushes
```

GitHub Actions automatically resolves tag names, so the `refs/tags/` prefix should not be included in the pattern.