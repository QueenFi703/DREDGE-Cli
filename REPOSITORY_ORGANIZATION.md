# Repository Organization

This document describes the organization of the DREDGE-Cli repository after the architecture cleanup.

## Directory Structure

```
DREDGE-Cli/
├── src/dredge/          # Python package source code
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   └── server.py
├── tests/               # Python test files
│   ├── test_basic.py
│   ├── test_cli.py
│   ├── test_performance.py
│   └── test_server.py
├── docs/                # Documentation files
│   ├── ARCHITECTURE.txt
│   ├── ABOUT_DREDGE_NANOGPT.md
│   ├── BENCHMARK_USAGE.md
│   ├── DollyIntegration.md
│   ├── EXPERIMENTATION_GUIDE.md
│   ├── EXTENDED_BENCHMARK_README.md
│   ├── FULL_DOCUMENTATION.md
│   ├── LATEX_README.md
│   ├── MIGRATION_GUIDE.md
│   ├── OPTIMIZATION_SUMMARY.md
│   ├── PERFORMANCE.md
│   ├── PERFORMANCE_IMPROVEMENTS.md
│   ├── QUASIMOTO_6D_README.md
│   ├── RESUME_CONTENT.md
│   └── dredge-x-dolly_pitch_deck_Version6.md
├── benchmarks/          # Benchmark scripts and results
│   ├── benchmark_demo.py
│   ├── quasimoto_*.py
│   ├── quasimoto_*.png
│   ├── quasimoto_paper.tex
│   └── migrate_quasimoto.sh
├── swift/               # Swift implementation
│   ├── Package.swift
│   ├── Sources/
│   ├── DREDGE_MVP.swift
│   ├── SharedStore.swift
│   └── AboutStrings.strings
├── archives/            # Archived files (excluded from git)
│   ├── *.zip
│   ├── *.torrent
│   └── *.docx
├── .github/             # GitHub configuration
│   └── workflows/
├── .devcontainer/       # Dev container configuration
├── README.md            # Main repository README
├── CHANGELOG.md         # Version changelog
├── SECURITY.md          # Security policy
├── LICENSE              # MIT License
├── pyproject.toml       # Python project configuration
└── requirements.txt     # Python dependencies
```

## Key Principles

1. **Root Directory**: Keep minimal - only essential project files (README, LICENSE, config files)
2. **Source Code**: All Python source in `src/dredge/`, all Swift source in `swift/`
3. **Documentation**: All docs in `docs/` directory with descriptive names
4. **Benchmarks**: All benchmark scripts and results in `benchmarks/`
5. **Archives**: Large binary files and archives in `archives/` (excluded from git)

## Build and Test

### Python
```bash
# Install
pip install -e .

# Run CLI
dredge-cli --version
python -m dredge serve

# Test
pytest
```

### Swift
```bash
# Build
cd swift
swift build

# Run
swift run dredge-cli
```

## Adding New Files

- **Documentation**: Add to `docs/` directory
- **Benchmarks**: Add to `benchmarks/` directory
- **Python source**: Add to `src/dredge/` directory
- **Swift source**: Add to `swift/` directory or `swift/Sources/`
- **Tests**: Add to `tests/` directory
- **Archives**: Add to `archives/` (will be ignored by git)

## Migration Notes

This reorganization was performed to clean up the main branch architecture. Files were moved from the root directory into organized subdirectories while maintaining full functionality of both Python and Swift implementations.
