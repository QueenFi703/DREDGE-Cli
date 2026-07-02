"""
Vercel API Route: Unified Auth Gateway
Entry point for Vercel/Production: api.gateway:app

This module imports the FastAPI application from unified_auth_gateway
and exports it as 'app' for Vercel/production deployments.

Usage:
  - Vercel: entrypoint = "api.gateway:app"
  - Local: python -m uvicorn api.gateway:app --host 127.0.0.1 --port 9000
  - Import: from api.gateway import app
"""

import sys
from pathlib import Path

# Add parent directory to sys.path so we can import our root-level modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the FastAPI application from unified_auth_gateway
from unified_auth_gateway import app

# Export as 'app' for Vercel and production deployments
# This is the ASGI application instance that Vercel will use
__all__ = ['app']
