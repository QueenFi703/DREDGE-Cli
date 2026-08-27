"""
DREDGE Vercel API - Main Entry Point with Full Integration

This entry point includes:
  - Three-Layer Cognitive Architecture (GPT Sol, Tresh, DREDGE)
  - Security Hardening (CORS, rate limiting, headers)
  - Cognitive Nervous System integration
  - Complete API endpoints

Deployment:
  - Vercel: entrypoint = "api.deployment:app"
  - Local: python core_gateway_integrated.py
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# IMPORT INTEGRATED GATEWAY
# ============================================================================

try:
    from core_gateway_integrated import app as integrated_app, HAS_GPT_SOL, HAS_TRESH, HAS_NERVOUS_SYSTEM, HAS_SECURITY
    app = integrated_app
    logger.info("✅ Integrated gateway loaded")
except Exception as e:
    logger.error(f"Failed to load integrated gateway: {e}")
    
    # Fallback to basic app
    app = FastAPI(
        title="DREDGE API (Fallback)",
        version="2.0.0",
        docs_url="/swagger"
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/")
    async def root():
        return {
            "status": "operational (fallback mode)",
            "message": "Integrated gateway not available"
        }
    
    @app.get("/health")
    async def health():
        return {"status": "healthy"}
    
    logger.warning("⚠️ Using fallback gateway")


# ============================================================================
# VERCEL HANDLER
# ============================================================================

handler = app

__all__ = ['app', 'handler']
