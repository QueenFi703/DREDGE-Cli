# DREDGE Quick Start

This guide is optimized for **first value in 5 minutes**.

## What DREDGE Is (in one line)

DREDGE is autonomous developer infrastructure: a lightweight runtime that accepts engineering intent and executes workflow intelligence through API-driven services.

## 5-Minute Hero Path

### 1) Install

```bash
pip install dredge-cli
```

### 2) Start DREDGE

```bash
dredge-cli serve
```

You now have a live service at `http://localhost:3001`.

### 3) Trigger a real action

```bash
curl -X POST http://localhost:3001/lift \
  -H "Content-Type: application/json" \
  -d '{"insight_text": "Summarize this into one high-leverage next action for engineering."}'
```

### 4) Verify successful outcome

```bash
curl http://localhost:3001/health
```

Expected shape:

```json
{
  "status": "healthy",
  "version": "..."
}
```

If steps 2–4 worked, you completed the core DREDGE loop: **activate → execute → verify**.

## Expand After First Value

Only after the hero path should you branch into deeper systems.

### Start MCP Operations (Port 3002)

```bash
dredge-cli mcp
```

### Run Test Suite

```bash
pytest tests/ -v
```

### Optional: Container Stack

```bash
docker-compose up
```

## Next Docs

- `docs/INSTALLATION.md`
- `docs/API_REFERENCE.md`
- `docs/FULL_DOCUMENTATION.md`
- `docs/github-app.md`
- `docs/SELL_ITSELF_PLAYBOOK.md`

## Common Commands

```bash
# Version
dredge-cli --version

# Help
dredge-cli --help

# Serve on custom port
dredge-cli serve --port 3003

# Serve in debug
dredge-cli serve --debug

# MCP server
dredge-cli mcp --port 3002
```
