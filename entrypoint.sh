#!/bin/bash
# Railway.app entrypoint script for DREDGE
# Handles startup, health checks, and graceful shutdown

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log function
log() {
    echo -e "${BLUE}[DREDGE]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

# Configuration
export FLASK_ENV="${FLASK_ENV:-production}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PORT="${PORT:-3001}"
export HOST="${HOST:-0.0.0.0}"

log "Starting DREDGE Studio v2.0"
log "Environment: $FLASK_ENV"
log "Port: $PORT"
log "Host: $HOST"

# Check if required environment variables are set
if [ -z "$GITHUB_CLIENT_ID" ] && [ "$FLASK_ENV" = "production" ]; then
    error "GITHUB_CLIENT_ID not set (required for production)"
    log "Set in Railway dashboard: Settings → Variables"
fi

if [ -z "$GITHUB_CLIENT_SECRET" ] && [ "$FLASK_ENV" = "production" ]; then
    error "GITHUB_CLIENT_SECRET not set (required for production)"
    log "Set in Railway dashboard: Settings → Variables"
fi

# Wait for dependencies if needed
log "Checking dependencies..."

# Check if Python is available
if ! command -v python &> /dev/null; then
    error "Python not found"
    exit 1
fi

success "Python: $(python --version 2>&1)"

# Check if Gunicorn is available
if ! command -v gunicorn &> /dev/null; then
    error "Gunicorn not found"
    log "Installing Gunicorn..."
    pip install gunicorn
fi

success "Gunicorn: $(gunicorn --version 2>&1 | head -1)"

# Check if Flask is available
if ! python -c "import flask" 2>/dev/null; then
    error "Flask not found"
    log "Installing Flask..."
    pip install -r requirements.txt
fi

success "Flask installed"

# Start DREDGE with Gunicorn
log "Starting Gunicorn with DREDGE WSGI app..."

exec gunicorn \
    --bind "${HOST}:${PORT}" \
    --workers 4 \
    --worker-class sync \
    --timeout 120 \
    --graceful-timeout 30 \
    --keep-alive 5 \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    --capture-output \
    "wsgi:app"
