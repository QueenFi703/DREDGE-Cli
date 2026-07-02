"""
Vercel API Route: Standalone Orion Gateway
Entry point: /api/orion
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from orion_gateway_authenticated import app

# Vercel expects 'app' as the ASGI application
export = app
