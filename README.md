# DREDGE

DREDGE — small Python package scaffold.

## Repository Structure

- **src/dredge/** - Python package source code
- **tests/** - Test files
- **docs/** - Documentation files
- **benchmarks/** - Benchmark scripts and results
- **swift/** - Swift implementation files
- **archives/** - Archived files (excluded from version control)

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