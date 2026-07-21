#!/bin/sh
set -e

# Start the MCP server in the bckground
python -m dredge mcp --host 0.0.0.0 --port 3002 &

# Run the Orion Gateway (FastAPI/ASGI) in the foreground via gunicorn+uvicorn worker
exec gunicorn -k uvicorn.workers.UvicornWorker dredge,orion_gateway:app --bind 0.0.0.0:${PORT}

