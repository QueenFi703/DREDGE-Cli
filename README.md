# DREDGE

**DREDGE** — Distill, Recall, Emerge, Detect, Guide, Evolve

A modular, event-driven architecture combining Swift (iOS/macOS) and Python implementations for digital memory and insight processing.

## Architecture

```
App -> DredgeCore -> SharedStore -> App Group Container
                        |
                     Widget
```

## Swift Implementation (iOS/macOS)

### Building with Swift Package Manager

```bash
swift build
swift run dredge-cli
```

### Running Tests

```bash
swift test
```

### Modules

- **DredgeCore** - Core processing engine with voice recognition and sentiment analysis
  - `DredgeEngine.swift` - Natural language sentiment processing
  - `VoiceDredger.swift` - Speech recognition and transcription
  - `SharedStore.swift` - App Group data persistence
  - `DredgeOperation.swift` - Background task processing

- **DredgeApp** - iOS SwiftUI application with voice capture and background tasks

- **DredgeCLI** - Command-line interface for testing and automation

## Python Implementation (Server/API)

### Install

Create a virtual environment and install:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e .
```

### Server Usage

DREDGE x Dolly includes a web server for API-based interaction. The server is configured to run on **port 3001** by default, making it perfect for GitHub Codespaces.

#### Starting the Server

Run the server with:

```bash
python -m dredge serve
```

Or with custom options:

```bash
python -m dredge serve --host 0.0.0.0 --port 3001 --debug
```

#### API Endpoints

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

### Python Tests

Run tests with pytest:

```bash
pip install -U pytest
pytest
```

## Development

### Swift Development
- Edit code in `Sources/DredgeCore`, `Sources/DredgeApp`, or `Sources/DredgeCLI`
- Update `Package.swift` for new dependencies or targets
- Run `swift build` to compile
- Run `swift test` for unit tests

### Python Development
- Edit code in `src/dredge`
- Update version in `pyproject.toml`
- Tag releases with v<version> and push tags

## Repository Structure

```
DREDGE-Cli/
├── Sources/
│   ├── DredgeCore/      # Swift core module (shared library)
│   ├── DredgeApp/       # iOS/macOS SwiftUI application
│   └── DredgeCLI/       # Swift command-line tool
├── Tests/
│   └── DredgeCoreTests/ # Swift unit tests
├── src/dredge/          # Python package source code
├── tests/               # Python test suite
├── docs/                # Documentation and guides
│   ├── assets/          # Images and visual assets
│   └── papers/          # Research papers (LaTeX)
├── benchmarks/          # Performance benchmarks and scripts
├── Package.swift        # Swift Package Manager manifest
└── pyproject.toml       # Python package configuration
```