"""
ORION SMART GATEWAY - Hybrid ASGI/WSGI Core Gateway

Architecture:
  A unified FastAPI (ASGI) application that serves as the core spine,
  with Flask (WSGI) legacy adapters mounted as subsystems via
  WSGIMiddleware, creating a single ASGI application that Vercel sees.

This allows:
  - Modern FastAPI routes alongside legacy Flask apps
  - Gradual migration from Flask to FastAPI
  - Single entry point: app = FastAPI()
  - Single ASGI server (Uvicorn)
  - Single deployment to Vercel

Pattern Benefits:
  1. Single ASGI entry point (clean)
  2. Legacy Flask apps work unchanged
  3. New features in FastAPI
  4. No dual servers needed
  5. Gradual migration path
  6. Professional hybrid architecture
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.wsgi import WSGIMiddleware
import logging
from typing import Dict, Any
from pathlib import Path
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CORE ASGI GATEWAY
# ============================================================================

app = FastAPI(
    title="ORION Smart Gateway",
    description="Hybrid ASGI/WSGI Gateway - FastAPI core with Flask legacy adapters",
    version="1.0.0",
    docs_url="/swagger",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# CORE FASTAPI ROUTES (Native ASGI)
# ============================================================================

@app.get("/", tags=["Core"])
async def root() -> Dict[str, Any]:
    """Root entry point"""
    return {
        "service": "ORION Smart Gateway",
        "version": "1.0.0",
        "architecture": "Hybrid ASGI/WSGI (FastAPI core + Flask legacy)",
        "status": "operational",
        "documentation": "/swagger",
        "health": "/health"
    }


@app.get("/health", tags=["Core"])
async def health() -> Dict[str, Any]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "orion-smart-gateway",
        "version": "1.0.0"
    }


@app.get("/status", tags=["Core"])
async def status() -> Dict[str, Any]:
    """Gateway status"""
    return {
        "status": "operational",
        "gateway": "ORION Smart Gateway",
        "version": "1.0.0",
        "type": "Hybrid ASGI/WSGI"
    }


@app.get("/adapters", tags=["Core"])
async def adapters() -> Dict[str, Any]:
    """List mounted adapters"""
    return {
        "status": "success",
        "adapters": {
            "fastapi": {
                "type": "ASGI Native",
                "routes": ["/", "/health", "/status", "/adapters"],
                "status": "active"
            },
            "flask_legacy": {
                "type": "WSGI (via WSGIMiddleware)",
                "prefix": "/legacy",
                "status": "active"
            }
        }
    }


# ============================================================================
# FLASK LEGACY ADAPTER (WSGI)
# ============================================================================

def create_flask_legacy_adapter():
    """
    Create a Flask application that acts as a WSGI adapter.
    This can be a legacy application mounted into the FastAPI core.
    """
    try:
        from flask import Flask, jsonify
        
        flask_app = Flask(__name__)
        
        @flask_app.route('/health', methods=['GET'])
        def flask_health():
            """Flask health endpoint"""
            return jsonify({
                "status": "healthy",
                "adapter": "flask_legacy",
                "type": "WSGI (mounted via WSGIMiddleware)"
            })
        
        @flask_app.route('/status', methods=['GET'])
        def flask_status():
            """Flask status endpoint"""
            return jsonify({
                "status": "operational",
                "adapter": "flask_legacy",
                "features": [
                    "Legacy routes",
                    "WSGI compatibility",
                    "Mounted in ASGI gateway"
                ]
            })
        
        @flask_app.route('/info', methods=['GET'])
        def flask_info():
            """Flask adapter info"""
            return jsonify({
                "adapter": "flask_legacy",
                "type": "WSGI Legacy Application",
                "mount_point": "/legacy",
                "parent": "ORION Smart Gateway (FastAPI)",
                "message": "This Flask app is mounted as a WSGI subsystem in the FastAPI core"
            })
        
        @flask_app.route('/hello', methods=['GET'])
        def flask_hello():
            """Example Flask route"""
            return jsonify({
                "message": "Hello from Flask legacy adapter",
                "mounted_in": "ORION Smart Gateway"
            })
        
        logger.info("[Adapter] Flask legacy adapter created")
        return flask_app
    
    except ImportError:
        logger.warning("[Adapter] Flask not installed, skipping legacy adapter")
        return None


# ============================================================================
# MOUNT FLASK ADAPTER AS WSGI SUBSYSTEM
# ============================================================================

def mount_flask_adapter():
    """Mount Flask legacy adapter to FastAPI core via WSGIMiddleware"""
    try:
        flask_app = create_flask_legacy_adapter()
        
        if flask_app:
            # Mount Flask app at /legacy prefix via WSGIMiddleware
            # This converts the WSGI Flask app to ASGI and mounts it
            app.mount("/legacy", WSGIMiddleware(flask_app))
            logger.info("[Mount] Flask legacy adapter mounted at /legacy")
            return True
    
    except Exception as e:
        logger.error(f"[Mount] Failed to mount Flask adapter: {e}")
        return False


# ============================================================================
# ADDITIONAL ASGI ROUTES (Can coexist with Flask routes)
# ============================================================================

@app.get("/orion/info", tags=["Gateway"])
async def orion_info() -> Dict[str, Any]:
    """ORION Gateway information"""
    return {
        "service": "ORION Smart Gateway",
        "architecture": "Hybrid ASGI/WSGI",
        "core": "FastAPI (ASGI)",
        "adapters": ["Flask Legacy (WSGI)"],
        "description": "Single ASGI application with WSGI subsystems"
    }


@app.get("/orion/endpoints", tags=["Gateway"])
async def orion_endpoints() -> Dict[str, Any]:
    """List all endpoints"""
    return {
        "status": "success",
        "endpoints": {
            "core_asgi": {
                "GET /": "Root entry point",
                "GET /health": "Health check",
                "GET /status": "Gateway status",
                "GET /adapters": "List adapters",
                "GET /orion/info": "ORION info",
                "GET /orion/endpoints": "This endpoint"
            },
            "flask_legacy_wsgi": {
                "prefix": "/legacy",
                "GET /legacy/health": "Flask health",
                "GET /legacy/status": "Flask status",
                "GET /legacy/info": "Adapter info",
                "GET /legacy/hello": "Example route"
            }
        }
    }


# ============================================================================
# STARTUP AND MOUNTING
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize gateway on startup"""
    print("\n" + "=" * 80)
    print("  ORION SMART GATEWAY - Hybrid ASGI/WSGI")
    print("=" * 80)
    print()
    print("Architecture: FastAPI (ASGI) Core + Flask (WSGI) Legacy Adapters")
    print()
    print("Mounting adapters...")
    print()
    
    # Mount Flask adapter
    mount_flask_adapter()
    
    print()
    print("Gateway Initialization Complete:")
    print("  - Core: FastAPI (ASGI)")
    print("  - Adapters: Flask legacy (WSGI)")
    print("  - Deployment: Single ASGI application")
    print("  - Framework: Uvicorn")
    print()
    print("Access Points:")
    print("  - Root:      http://127.0.0.1:8000/")
    print("  - Health:    http://127.0.0.1:8000/health")
    print("  - Status:    http://127.0.0.1:8000/status")
    print("  - Adapters:  http://127.0.0.1:8000/adapters")
    print("  - Swagger:   http://127.0.0.1:8000/swagger")
    print()
    print("Flask Legacy Routes:")
    print("  - /legacy/health")
    print("  - /legacy/status")
    print("  - /legacy/info")
    print("  - /legacy/hello")
    print()
    print("=" * 80)
    print()


# ============================================================================
# LOCAL EXECUTION
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
