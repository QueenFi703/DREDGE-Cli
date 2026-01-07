# DREDGE

DREDGE — Distill, Recall, Emerge, Detect, Guide, Evolve

A modular platform for orchestrating verified compute workloads with formal isolation guarantees.

## Overview

DREDGE integrates three key components:

1. **DREDGE Core**: Python-based orchestration and workload management
2. **Dolly**: GPU-accelerated compute task lifting and transformation
3. **µH-iOS**: Formally verified micro-hypervisor providing isolation guarantees

## Components

### DREDGE Core (Python)

Small Python package for workload orchestration and API services.

### Dolly Integration

GPU-accelerated insight processing and transformation. See `DollyIntegration.md` for details.

### µH-iOS (Rust + Swift)

Formally verified micro-hypervisor nucleus for iOS/macOS providing:
- Memory non-interference guarantees
- Capability-based access control
- Deterministic VM exit handling
- Minimal trusted computing base (~3500 LOC)

**Documentation:**
- Architecture & Implementation: `uh-ios/README.md`
- Research Paper: `docs/uh-ios-paper.md`
- Integration Guide: `docs/uhios-dredge-integration.md`

## Install

### Python Components

Create a virtual environment and install:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e .
```

### µH-iOS (Optional)

To build the formally verified µH-iOS component:

```bash
# Build Rust core
cd uh-ios
cargo build --release
cargo test

# Build Swift application (requires macOS/iOS)
cd ../uh-ios-app
swift build
```

## Server Usage

DREDGE x Dolly includes a web server for API-based interaction. The server is configured to run on **port 3001** by default, making it perfect for GitHub Codespaces.

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

```bash
pip install -U pytest
pytest
```

Test µH-iOS core:

```bash
cd uh-ios
cargo test
```

## Development

- Edit Python code in `src/dredge`
- Edit Rust core in `uh-ios/src`
- Edit Swift app in `uh-ios-app/Sources`
- Update version in `pyproject.toml`
- Tag releases with v<version> and push tags

## Architecture

```
┌─────────────────────────────────────────────────────┐
│               DREDGE Orchestration                  │
│  (Policy Definition, Workload Management, Dolly)    │
└─────────────────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────┐
│              µH-iOS Verified Core                   │
│   (Memory Isolation, Capability Enforcement)        │
└─────────────────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────┐
│          Apple Hypervisor.framework (HVF)           │
└─────────────────────────────────────────────────────┘
```

## Features

### Formal Verification

µH-iOS provides mathematically proven guarantees:
- **Memory Non-Interference**: `∀ vm₁ vm₂. vm₁ ≠ vm₂ → memory(vm₁) ∩ memory(vm₂) = ∅`
- **Capability Soundness**: Actions require explicit capabilities
- **Deterministic Exit Handling**: Reproducible execution
- **Totality**: All VM exits handled safely

### GPU Acceleration

Dolly integration enables efficient compute task transformation with GPU support when available.

### Platform Compliance

Works on stock iOS/macOS devices without jailbreaks or kernel modifications.

## License

MIT License - See LICENSE file

## Contributing

Contributions welcome! Please see individual component READMEs for specific contribution guidelines.

## References

- µH-iOS Implementation: `uh-ios/`
- Research Paper: `docs/uh-ios-paper.md`
- Dolly Integration: `DollyIntegration.md`
- Integration Guide: `docs/uhios-dredge-integration.md`
