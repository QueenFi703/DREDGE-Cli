"""
Vercel API Route: Unified Auth Gateway
Entry point: /api/gateway
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from unified_auth_gateway import app

# Vercel expects 'app' as the ASGI application
export = app
