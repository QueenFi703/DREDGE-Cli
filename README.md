# DREDGE

**DREDGE** — Distill, Recall, Emerge, Detect, Guide, Evolve

A modular, event-driven architecture combining Swift (iOS/macOS) and Python implementations for digital memory and insight processing, with integrated Quasimoto wave functions for signal processing and anomaly detection.

## Architecture

```
┌─────────────────────────────────────────┐
│          User Interface Layer           │
├─────────────┬───────────────────────────┤
│  CLI        │    MCP Server (Port 3001) │
│  Commands   │    AI Agent Tools         │
└──────┬──────┴──────┬────────────────────┘
       │             │
       ├─────────────┼──────────────────┐
       ▼             ▼                  ▼
┌──────────────┐ ┌──────────────┐ ┌─────────────┐
│  Quasimoto   │ │    Dolly     │ │  Insights   │
│  Wave Engine │ │ GPU Accel.   │ │  Storage    │
└──────────────┘ └──────────────┘ └─────────────┘
       │                  │              │
       └──────────────────┴──────────────┘
                    │
              Data Pipeline
```

**Swift Layer (iOS/macOS):**
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

## Python Implementation (Server/API + Quasimoto)

### Install

Create a virtual environment and install:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e .
```

For Quasimoto wave processing (optional):
```bash
pip install torch  # For GPU-accelerated wave fitting
```

### CLI Commands

#### Server
```bash
# Start the DREDGE x Dolly server
python -m dredge serve --host 0.0.0.0 --port 3001 --debug
```

#### Quasimoto Wave Processing
```bash
# Fit wave to signal data
python -m dredge fit-wave signal.json --output model.json --n-waves 16 --epochs 2000

# Process signal with trained wave
python -m dredge process-signal signal.json --model model.json --output predictions.json

# Detect anomalies in signal
python -m dredge analyze-anomaly signal.json --threshold 2.0 --n-waves 16

# Run benchmarks
python -m dredge benchmark --model quasimoto --epochs 2000
```

### Server Usage

DREDGE x Dolly includes a web server for API-based interaction with MCP tools for Quasimoto wave processing. The server is configured to run on **port 3001** by default.

#### API Endpoints

Core endpoints:
- **GET /** - API information and available endpoints
- **GET /health** - Health check endpoint
- **POST /lift** - Lift an insight with Dolly integration

MCP Tool endpoints:
- **GET /mcp/tools** - List available MCP tools
- **POST /mcp/call** - Call an MCP tool

Example usage:

```bash
# Get API info
curl http://localhost:3001/

# List MCP tools
curl http://localhost:3001/mcp/tools

# Fit Quasimoto wave via MCP
curl -X POST http://localhost:3001/mcp/call \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "fit_quasimoto_wave",
    "parameters": {
      "signal_data": [0.5, 0.8, 1.0, 0.9, 0.6, 0.3, 0.1, 0.2, 0.5, 0.8],
      "n_waves": 16,
      "epochs": 2000
    }
  }'

# Detect anomalies via MCP
curl -X POST http://localhost:3001/mcp/call \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "detect_signal_anomalies",
    "parameters": {
      "signal_data": [0.5, 0.6, 0.5, 0.6, 0.5, 10.0, 0.5, 0.6],
      "threshold": 2.0
    }
  }'

# Transform insight to wave representation
curl -X POST http://localhost:3001/mcp/call \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "transform_insight_wave",
    "parameters": {
      "insight_text": "Digital memory must be human-reachable"
    }
  }'
```

### MCP Tools

DREDGE provides 5 MCP tools for AI agents:

1. **fit_quasimoto_wave** - Fit wave ensemble to signal data with GPU acceleration
2. **detect_signal_anomalies** - Detect localized irregularities in signals
3. **transform_insight_wave** - Transform text insights into wave representations
4. **predict_signal_continuation** - Predict future signal values
5. **analyze_wave_similarity** - Compare two signals via wave parameters

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
│   ├── cli.py           # CLI commands (serve, fit-wave, process-signal, etc.)
│   ├── server.py        # Flask server with MCP endpoints
│   ├── quasimoto.py     # Quasimoto wave processing core
│   └── mcp_server.py    # MCP tool registry and implementations
├── tests/               # Python test suite
│   ├── test_basic.py
│   ├── test_server.py
│   └── test_integration.py  # Quasimoto integration tests
├── docs/                # Documentation and guides
│   ├── assets/          # Images and visual assets
│   └── papers/          # Research papers (LaTeX)
├── benchmarks/          # Performance benchmarks and scripts
│   ├── quasimoto_benchmark.py
│   ├── quasimoto_extended_benchmark.py
│   └── quasimoto_6d_benchmark.py
├── Package.swift        # Swift Package Manager manifest
└── pyproject.toml       # Python package configuration
```

## Integration Threads

### Thread 1: CLI → Quasimoto
Signal processing commands with file I/O and GPU acceleration via Dolly.

### Thread 2: MCP Server → Quasimoto
AI agent tools for wave fitting, anomaly detection, and insight transformation.

### Thread 3: Quasimoto → Insights
Wave-encoded insight storage and semantic retrieval via wave similarity.

### Thread 4: Dolly ↔ Quasimoto
GPU-accelerated training with unified memory architecture.
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