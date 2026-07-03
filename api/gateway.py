"""
Vercel API Route: Unified Auth Gateway
Entry point for Vercel/Production: api.gateway:app

This module provides multiple FastAPI application sources:
  1. Primary: unified_auth_gateway (Authentication + API key management)
  2. Alternative: full_web_server (DREDGE Studio UI)

The 'app' variable is exported as the main ASGI instance for Vercel.

Usage:
  - Vercel: entrypoint = "api.gateway:app"
  - Local: python -m uvicorn api.gateway:app --host 127.0.0.1 --port 9000
  - Import: from api.gateway import app
"""

import sys
from pathlib import Path

# Add parent directory to sys.path so we can import our root-level modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import from full_web_server (DREDGE Studio)
# Fall back to unified_auth_gateway if not available
try:
    from full_web_server import app
    print("[OK] Using full_web_server (DREDGE Studio FastAPI)")
except ImportError as e:
    try:
        from unified_auth_gateway import app
        print("[OK] Using unified_auth_gateway (Auth Gateway FastAPI)")
    except ImportError as e2:
        # If neither is available, create a minimal app
        from fastapi import FastAPI
        app = FastAPI(title="DREDGE Gateway", version="2.0.0")
        
        @app.get("/health")
        async def health():
            return {"status": "error", "message": "Could not load application"}

# Export as 'app' for Vercel and production deployments
# This is the ASGI application instance that Vercel will use
__all__ = ['app']
