# DREDGE Quick Start

Get running with DREDGE in 5 minutes.

## Installation
```bash
pip install dredge-cli
```

## Start the Server
```bash
dredge-cli serve
```

Server runs on `http://localhost:3001`

## Lift Your First Insight

### Using curl
```bash
curl -X POST http://localhost:3001/lift \
  -H "Content-Type: application/json" \
  -d '{"insight_text": "Digital memory must be human-reachable."}'
```

### Response
```json
{
  "id": "a3f8b2c1...",
  "text": "Digital memory must be human-reachable.",
  "lifted": true,
  "message": "Insight processed"
}
```

### Using Python
```python
import requests

response = requests.post(
    "http://localhost:3001/lift",
    json={"insight_text": "Digital memory must be human-reachable."}
)

print(response.json())
```

## Start the MCP Server (Quasimoto Models)
```bash
dredge-cli mcp
```

MCP server runs on `http://localhost:3002`

## Run Quasimoto Benchmarks

### 1D Benchmark (Glitchy Chirp)
```bash
cd benchmarks
python quasimoto_benchmark.py
```

This compares Quasimoto against SIREN and Random Fourier Features.

### Extended Benchmark (4D/6D)
```bash
python quasimoto_extended_benchmark.py
```

### 6D Benchmark
```bash
python quasimoto_6d_benchmark.py
```

### Interference Basis
```bash
python quasimoto_interference_benchmark.py
```

## Run Tests

### Python Tests
```bash
pytest tests/ -v
```

### Swift Tests
```bash
swift test
```

## Check Health
```bash
curl http://localhost:3001/health
```

Response:
```json
{
  "status": "healthy",
  "version": "0.1.4"
}
```

## Next Steps

### Documentation
- <a>INSTALLATION.md</a> - Detailed installation guide
- <a>API_REFERENCE.md</a> - Complete API documentation
- <a>FULL_DOCUMENTATION.md</a> - Architecture details

### Benchmarks
- <a>BENCHMARK_USAGE.md</a> - Benchmark guide
- <a>EXTENDED_BENCHMARK_README.md</a> - Extended benchmarks
- <a>QUASIMOTO_6D_README.md</a> - 6D architecture

### Research
- <a>benchmarks/quasimoto_paper.tex</a> - LaTeX paper
- <a>EXPERIMENTATION_GUIDE.md</a> - Experimentation guide
- <a>RESUME_CONTENT.md</a> - Portfolio content

## Common Commands

```bash
# Show version
dredge-cli --version

# Show help
dredge-cli --help

# Start server with custom port
dredge-cli serve --port 3003

# Start server with debug mode
dredge-cli serve --debug

# Start MCP server
dredge-cli mcp --port 3002
```

## Docker Quick Start

```bash
# Build
docker build -t dredge-cli .

# Run server
docker run -p 3001:3001 dredge-cli serve

# Run with docker-compose
docker-compose up
```

## Troubleshooting

**Port already in use?**
```bash
dredge-cli serve --port 3003
```

**Command not found?**
```bash
pip install --user dredge-cli
export PATH="$HOME/.local/bin:$PATH"
```

**Tests failing?**
```bash
pip install -r requirements.txt
pytest tests/ -v
```
