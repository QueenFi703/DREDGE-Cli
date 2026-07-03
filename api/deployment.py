"""
DREDGE Deployment Entry Point

This module provides the ASGI application for production deployment on Vercel.

Architecture:
  The Core Gateway (core_gateway.py) is a unified ASGI application spine that
  serves as the single entry point for all DREDGE services.

  All functionality is provided through a modular adapter system:
  - Studio Adapter: Web UI and dashboard
  - Auth Adapter: API key management
  - Health Adapter: System monitoring
  - Admin Adapter: Administrative operations

Entry Point:
  app = FastAPI application from core_gateway.py

Deployment:
  - Vercel: entrypoint = "api.deployment:app"
  - Local: python -m uvicorn api.deployment:app --port 9000
  - Direct: python core_gateway.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the core gateway (production entry point)
from core_gateway import app

# Export for Vercel/ASGI servers
__all__ = ['app']
