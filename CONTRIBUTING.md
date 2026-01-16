# Contributing to DREDGE

## Philosophy
DREDGE values clarity, modularity, and intentionality. Contributions should:
- Serve a clear purpose
- Be placed in the correct directory
- Include tests and documentation
- Follow the existing code style

## Directory Structure
- `src/dredge/` - Python source code
- `swift/` - Swift implementation
- `tests/` - Python tests (pytest)
- `benchmarks/` - Quasimoto neural architectures
- `docs/` - Documentation

## How to Contribute
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Add tests for new functionality
4. Update relevant documentation
5. Submit a pull request

## Running Tests

### Python Tests
```bash
pytest tests/ -v
```

### Swift Tests
```bash
swift test
```

## Code Style
- **Python**: Black formatting, type hints encouraged
- **Swift**: Standard Swift conventions

## Pull Request Process
1. Ensure all tests pass
2. Update documentation in `docs/` if needed
3. Update `CHANGELOG.md` with your changes
4. Reference any related issues in your PR description

## Reporting Bugs
Use the bug report issue template and include:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python/Swift version)

## Suggesting Features
Use the feature request issue template and explain:
- The problem you're trying to solve
- Your proposed solution
- Why this would be valuable to DREDGE users

## Questions?
Open a discussion issue or reach out to QueenFi703.
