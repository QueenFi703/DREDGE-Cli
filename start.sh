#!/bin/sh
set -eu

# Railway process guardrails:
# - one MCP child process
# - one foreground Orion Gateway process
# - PORT is supplied by Railway
# - terminate the child if the gateway exits

MCP_PORT="${MCP_PORT:-8001}"

cleanup() {
  if [ -n "${MCP_PID:-}" ]; then
    kill "$MCP_PID" 2>/dev/null || true
    wait "$MCP_PID" 2>/dev/null || true
  fi
}
trap cleanup INT TERM EXIT

# Avoid multiplying workers/processes on a small Railway service.
dredge-server --host 0.0.0.0 --port "$MCP_PORT" &
MCP_PID=$!

# Keep a single foreground gateway process so Railway can observe health,
# restart the service cleanly, and avoid worker/process congestion.
exec gunicorn \
  -k uvicorn.workers.UvicornWorker \
  --workers 1 \
  --bind "0.0.0.0:${PORT:-8000}" \
  --timeout 120 \
  --graceful-timeout 30 \
  --access-logfile - \
  --error-logfile - \
  --log-level info \
  dredge.orion_gateway:app
