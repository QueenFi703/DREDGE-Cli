"""
Vercel API - Main Entry Point for DREDGE Studio

This is the primary entry point for Vercel serverless deployment.

Provides:
  - Gateway status and health checks
  - DREDGE capabilities
  - Full API endpoints
  - Error handling with fallback

Entry point: api.index:app or api.index:handler
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

# Try to import the main gateway, but have a fallback
try:
    from core_gateway import app as core_app
    HAS_CORE = True
    logger.info("✅ Core gateway loaded")
except Exception as e:
    logger.warning(f"⚠️ Core gateway import failed: {e}")
    HAS_CORE = False
    core_app = None


# ============================================================================
# FALLBACK APP (if core gateway fails)
# ============================================================================

def create_fallback_app():
    """Create a minimal fallback FastAPI app"""
    app = FastAPI(
        title="DREDGE Studio API",
        description="DREDGE Studio - Unified Gateway",
        version="2.0.0"
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "status": "operational",
            "service": "DREDGE Studio",
            "version": "2.0.0",
            "mode": "fallback",
            "message": "Core gateway unavailable, running in fallback mode"
        }
    
    @app.get("/health")
    async def health():
        """Health check"""
        return {"status": "healthy", "service": "dredge-studio"}
    
    @app.get("/status")
    async def status():
        """Gateway status"""
        return {
            "status": "operational",
            "service": "dredge-studio",
            "version": "2.0.0",
            "mode": "fallback"
        }
    
    @app.get("/api")
    async def api_root():
        """API root"""
        return {
            "api": "DREDGE Studio API v2",
            "version": "2.0.0",
            "status": "operational",
            "endpoints": [
                "/health",
                "/status",
                "/api/status",
                "/api/capabilities",
                "/api/dredge/status",
                "/api/gordon/capabilities"
            ]
        }
    
    @app.get("/api/status")
    async def api_status():
        """API status"""
        return {
            "status": "operational",
            "version": "2.0.0",
            "deployment": "vercel",
            "mode": "fallback"
        }
    
    @app.get("/api/capabilities")
    async def capabilities():
        """API capabilities"""
        return {
            "status": "operational",
            "capabilities": [
                {
                    "name": "Health Check",
                    "endpoint": "/health",
                    "method": "GET",
                    "description": "Check API health"
                },
                {
                    "name": "Status",
                    "endpoint": "/api/status",
                    "method": "GET",
                    "description": "Get API status"
                },
                {
                    "name": "DREDGE Status",
                    "endpoint": "/api/dredge/status",
                    "method": "GET",
                    "description": "Get DREDGE status"
                },
                {
                    "name": "Gordon Capabilities",
                    "endpoint": "/api/gordon/capabilities",
                    "method": "GET",
                    "description": "Get Gordon integration capabilities"
                }
            ]
        }
    
    @app.get("/api/dredge/status")
    async def dredge_status():
        """DREDGE status"""
        return {
            "status": "operational",
            "service": "dredge-studio",
            "version": "2.0.0",
            "deployment": "vercel",
            "features": [
                "Model Management",
                "MCP Operations",
                "Insight Lifting",
                "DREDGE Pipeline",
                "Swift Toolchain",
                "Code Generation",
                "Dependabot Alerts",
                "Container Status",
                "API Testing",
                "Visualization"
            ]
        }
    
    @app.get("/api/gordon/capabilities")
    async def gordon_capabilities():
        """Gordon integration capabilities"""
        return {
            "gordon_version": "1.0.0",
            "dredge_integration": "active",
            "status": "operational",
            "capabilities": [
                {
                    "name": "Health Check",
                    "endpoint": "/health",
                    "method": "GET",
                    "description": "Check API health"
                },
                {
                    "name": "DREDGE Status",
                    "endpoint": "/api/dredge/status",
                    "method": "GET",
                    "description": "Get DREDGE operational status"
                },
                {
                    "name": "Gateway Status",
                    "endpoint": "/status",
                    "method": "GET",
                    "description": "Get gateway status"
                }
            ]
        }
    
    @app.exception_handler(Exception)
    async def exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception on {request.url}: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": str(exc),
                "path": str(request.url),
                "status": 500
            }
        )
    
    return app


# ============================================================================
# APPLICATION SELECTION
# ============================================================================

# Use core app if available, otherwise fallback
if HAS_CORE:
    app = core_app
    logger.info("✅ Using core gateway")
else:
    app = create_fallback_app()
    logger.warning("⚠️ Using fallback minimal app - core gateway not available")


# ============================================================================
# ADD MISSING API ROUTES TO CORE APP
# ============================================================================

if HAS_CORE:
    # Add API routes to core app if they don't exist
    
    @app.get("/api", include_in_schema=True)
    async def api_root():
        """API root"""
        return {
            "api": "DREDGE Studio API v2",
            "version": "2.0.0",
            "status": "operational",
            "endpoints": [
                "/health",
                "/status",
                "/api/status",
                "/api/capabilities",
                "/api/dredge/status",
                "/api/gordon/capabilities"
            ]
        }
    
    @app.get("/api/status", include_in_schema=True)
    async def api_status():
        """API status"""
        return {
            "status": "operational",
            "version": "2.0.0",
            "deployment": "vercel"
        }
    
    @app.get("/api/capabilities", include_in_schema=True)
    async def capabilities():
        """API capabilities"""
        return {
            "status": "operational",
            "capabilities": [
                {
                    "name": "Health Check",
                    "endpoint": "/health",
                    "method": "GET",
                    "description": "Check API health"
                },
                {
                    "name": "Status",
                    "endpoint": "/api/status",
                    "method": "GET",
                    "description": "Get API status"
                },
                {
                    "name": "DREDGE Status",
                    "endpoint": "/api/dredge/status",
                    "method": "GET",
                    "description": "Get DREDGE status"
                },
                {
                    "name": "Gordon Capabilities",
                    "endpoint": "/api/gordon/capabilities",
                    "method": "GET",
                    "description": "Get Gordon integration capabilities"
                }
            ]
        }
    
    @app.get("/api/dredge/status", include_in_schema=True)
    async def dredge_status():
        """DREDGE status"""
        return {
            "status": "operational",
            "service": "dredge-studio",
            "version": "2.0.0",
            "deployment": "vercel",
            "features": [
                "Model Management",
                "MCP Operations",
                "Insight Lifting",
                "DREDGE Pipeline",
                "Swift Toolchain",
                "Code Generation",
                "Dependabot Alerts",
                "Container Status",
                "API Testing",
                "Visualization"
            ]
        }
    
    @app.get("/api/gordon/capabilities", include_in_schema=True)
    async def gordon_capabilities():
        """Gordon integration capabilities"""
        return {
            "gordon_version": "1.0.0",
            "dredge_integration": "active",
            "status": "operational",
            "capabilities": [
                {
                    "name": "Health Check",
                    "endpoint": "/health",
                    "method": "GET",
                    "description": "Check API health"
                },
                {
                    "name": "DREDGE Status",
                    "endpoint": "/api/dredge/status",
                    "method": "GET",
                    "description": "Get DREDGE operational status"
                },
                {
                    "name": "Gateway Status",
                    "endpoint": "/status",
                    "method": "GET",
                    "description": "Get gateway status"
                }
            ]
        }


# ============================================================================
# VERCEL HANDLERS
# ============================================================================

# Vercel calls this handler
handler = app

__all__ = ['app', 'handler']
