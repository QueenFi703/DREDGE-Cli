"""
ORION Deployment Entry Point - Single ASGI Application

This is the production entry point for Vercel and other ASGI servers.

What Vercel Sees:
  app = FastAPI()
  
What's Actually Running:
  - FastAPI core with native ASGI routes
  - Flask legacy adapter mounted via WSGIMiddleware
  - Single ASGI application (no dual servers)
  - All routes accessible from one gateway

This is a hybrid ASGI/WSGI pattern that allows:
  1. Modern FastAPI development
  2. Legacy Flask apps to work unchanged
  3. Single deployment and entry point
  4. Gradual migration path
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the hybrid gateway - the unified ASGI application
from hybrid_gateway import app

# This is what Vercel/ASGI servers see
# A single FastAPI() instance that internally mounts WSGI adapters
__all__ = ['app']
