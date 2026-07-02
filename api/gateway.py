"""
Vercel API Route: Unified Auth Gateway
Entry point for Vercel: api/gateway.py
Exports: app (FastAPI ASGI instance)
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from unified_auth_gateway import app

# Vercel requires 'app' as the top-level ASGI application
__all__ = ['app']
