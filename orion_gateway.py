"""
ORION Gateway - FastAPI Entry Point (FIXED - No Circular Imports)

This is the PRIMARY entry point for:
  - Local development: python orion_gateway.py
  - Vercel deployment: vercel deploy (uses hybrid_gateway.py or this file)

Contains:
  - FastAPI core application
  - DREDGE integration
  - Gordon bridge support
  - No imports from api/ package (avoids circular imports)
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import asyncio
import json
import logging
from pathlib import Path
import sys
import os

# Setup path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dredge-cli-repo', 'src'))

# Import DREDGE modules (NOT from api/index.py - avoids circular import)
try:
    from dredge.architecture import dredge_run_pipeline
    from dredge.providers import execute_translation_chain, execute_analysis_chain, get_provider_status
    from dredge.gordon_integration import GordonDREDGEBridge
    HAS_DREDGE = True
except ImportError as e:
    print(f"Warning: DREDGE modules not available: {e}")
    HAS_DREDGE = False

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CORE FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="ORION Smart Gateway",
    description="Hybrid ASGI/WSGI Gateway - FastAPI core with Flask adapters",
    version="1.0.0",
    docs_url="/swagger",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files (optional)
static_path = Path(__file__).parent / "dredge-cli-repo" / "src" / "dredge" / "static"
if static_path.exists():
    try:
        app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
    except Exception as e:
        logger.warning(f"Could not mount static files: {e}")

# Global state
gordon_bridge = None

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class PipelineRequest(BaseModel):
    """Pipeline execution request"""
    input_data: Dict[str, Any]
    pipeline_type: str = "standard"
    pipeline_id: Optional[str] = None


class TranslationRequest(BaseModel):
    """Translation request"""
    text: str
    source_language: str = "en"
    target_language: str = "es"


class AnalysisRequest(BaseModel):
    """Analysis request"""
    query: str
    context: Optional[Dict[str, Any]] = None


class GordonTaskRequest(BaseModel):
    """Gordon task request"""
    task_id: str
    task_type: str
    input_data: Dict[str, Any]
    priority: int = 5


# ============================================================================
# CORE ROUTES (FastAPI Native)
# ============================================================================

@app.get("/", tags=["Core"])
async def root():
    """Root entry point"""
    return {
        "service": "ORION Smart Gateway",
        "version": "1.0.0",
        "architecture": "Hybrid ASGI/WSGI",
        "status": "operational",
        "documentation": "/swagger",
        "health": "/health"
    }


@app.get("/health", tags=["Core"])
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "orion-smart-gateway",
        "version": "1.0.0"
    }


@app.get("/status", tags=["Core"])
async def status():
    """Gateway status"""
    return {
        "status": "operational",
        "gateway": "ORION Smart Gateway",
        "version": "1.0.0",
        "type": "Hybrid ASGI/WSGI",
        "components": {
            "fastapi_core": "operational",
            "dredge_integration": "available" if HAS_DREDGE else "unavailable",
            "gordon_bridge": "ready" if gordon_bridge else "not_initialized"
        }
    }


@app.get("/adapters", tags=["Core"])
async def adapters():
    """List all adapters"""
    return {
        "status": "success",
        "adapters": {
            "fastapi": {
                "type": "ASGI Native",
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
# DREDGE PIPELINE ENDPOINTS
# ============================================================================

@app.post("/pipeline/execute", tags=["DREDGE"])
async def execute_pipeline(request: PipelineRequest):
    """Execute DREDGE pipeline"""
    if not HAS_DREDGE:
        raise HTTPException(status_code=503, detail="DREDGE not available")
    
    try:
        result = await dredge_run_pipeline(
            input_data=request.input_data,
            pipeline_type=request.pipeline_type,
            pipeline_id=request.pipeline_id
        )
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/translate", tags=["DREDGE"])
async def translate(request: TranslationRequest):
    """Execute translation chain"""
    if not HAS_DREDGE:
        raise HTTPException(status_code=503, detail="DREDGE not available")
    
    try:
        result = await execute_translation_chain({
            "text": request.text,
            "source_language": request.source_language,
            "target_language": request.target_language
        })
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze", tags=["DREDGE"])
async def analyze(request: AnalysisRequest):
    """Execute analysis chain"""
    if not HAS_DREDGE:
        raise HTTPException(status_code=503, detail="DREDGE not available")
    
    try:
        result = await execute_analysis_chain({
            "query": request.query,
            "context": request.context or {}
        })
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/providers/status", tags=["DREDGE"])
async def provider_status():
    """Get provider status"""
    if not HAS_DREDGE:
        raise HTTPException(status_code=503, detail="DREDGE not available")
    
    try:
        status = await get_provider_status()
        return {"status": "success", "providers": status}
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# GORDON INTEGRATION ENDPOINTS
# ============================================================================

@app.post("/gordon/start", tags=["Gordon"])
async def start_gordon_bridge():
    """Start Gordon bridge"""
    if not HAS_DREDGE:
        raise HTTPException(status_code=503, detail="DREDGE not available")
    
    global gordon_bridge
    try:
        if gordon_bridge is None:
            gordon_bridge = GordonDREDGEBridge()
        
        logger.info("Gordon bridge initialized")
        return {"status": "success", "message": "Gordon bridge started"}
    except Exception as e:
        logger.error(f"Failed to start Gordon: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/gordon/stop", tags=["Gordon"])
async def stop_gordon_bridge():
    """Stop Gordon bridge"""
    if not HAS_DREDGE:
        raise HTTPException(status_code=503, detail="DREDGE not available")
    
    global gordon_bridge
    try:
        if gordon_bridge:
            await gordon_bridge.stop()
            gordon_bridge = None
        
        logger.info("Gordon bridge stopped")
        return {"status": "success", "message": "Gordon bridge stopped"}
    except Exception as e:
        logger.error(f"Failed to stop Gordon: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/gordon/capabilities", tags=["Gordon"])
async def gordon_capabilities():
    """Get DREDGE capabilities for Gordon"""
    return {
        "status": "success",
        "agent": "DREDGE",
        "version": "1.0.0",
        "capabilities": [
            {
                "name": "pipeline_execution",
                "description": "Execute DREDGE DAG pipelines",
                "endpoint": "/pipeline/execute",
                "method": "POST"
            },
            {
                "name": "text_translation",
                "description": "Translate text with multi-provider support",
                "endpoint": "/translate",
                "method": "POST"
            },
            {
                "name": "semantic_analysis",
                "description": "Perform semantic analysis on text",
                "endpoint": "/analyze",
                "method": "POST"
            },
            {
                "name": "provider_management",
                "description": "Monitor and manage provider health",
                "endpoint": "/providers/status",
                "method": "GET"
            }
        ],
        "max_concurrent_tasks": 10,
        "timeout": 120
    }


@app.post("/gordon/task/execute", tags=["Gordon"])
async def execute_gordon_task(request: GordonTaskRequest):
    """Execute task from Gordon"""
    if not HAS_DREDGE:
        raise HTTPException(status_code=503, detail="DREDGE not available")
    
    try:
        if request.task_type == "pipeline":
            result = await dredge_run_pipeline(request.input_data)
        elif request.task_type == "translate":
            result = await execute_translation_chain(request.input_data)
        elif request.task_type == "analyze":
            result = await execute_analysis_chain(request.input_data)
        else:
            raise ValueError(f"Unknown task type: {request.task_type}")

        return {
            "status": "success",
            "task_id": request.task_id,
            "result": result,
            "duration": 0.0
        }
    except Exception as e:
        logger.error(f"Task execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/gordon/status", tags=["Gordon"])
async def gordon_status():
    """Get Gordon bridge status"""
    return {
        "status": "success",
        "bridge": {
            "status": "operational" if gordon_bridge else "stopped",
            "gordon": "ready",
            "dredge": "operational" if HAS_DREDGE else "unavailable"
        }
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "detail": exc.detail}
    )


# ============================================================================
# STARTUP/SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Startup event"""
    logger.info("ORION Smart Gateway starting...")
    logger.info("Available at http://127.0.0.1:8000")
    logger.info("Swagger docs at http://127.0.0.1:8000/swagger")
    if HAS_DREDGE:
        logger.info("DREDGE integration active")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event"""
    global gordon_bridge
    if gordon_bridge:
        await gordon_bridge.stop()
    logger.info("ORION Smart Gateway shutting down...")


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 80)
    print("  ORION Smart Gateway")
    print("=" * 80)
    print("Starting server...")
    print()
    print("  API Docs: http://127.0.0.1:8000/swagger")
    print("  OpenAPI:  http://127.0.0.1:8000/openapi.json")
    print()
    print("=" * 80)
    print()
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
