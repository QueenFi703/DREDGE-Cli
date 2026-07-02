"""
DREDGE Orion Gateway - FastAPI Integration
Combines FastAPI/Orion with DREDGE AI and Gordon multi-agent framework
Runs on http://127.0.0.1:8080
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

# Import DREDGE modules
from dredge.architecture import dredge_run_pipeline
from dredge.providers import execute_translation_chain, execute_analysis_chain, get_provider_status
from dredge.gordon_integration import GordonDREDGEBridge

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI App
app = FastAPI(
    title="DREDGE Orion Gateway",
    description="AI Pipeline Gateway with DREDGE and Gordon Integration",
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

# Static files
static_path = Path(__file__).parent / "dredge-cli-repo" / "src" / "dredge" / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

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
# HEALTH & STATUS
# ============================================================================

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "components": {
            "pipeline_engine": "operational",
            "provider_chain": "operational",
            "gordon_bridge": "ready" if gordon_bridge else "not_initialized"
        }
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "DREDGE Orion Gateway",
        "version": "2.0.0",
        "description": "AI Pipeline Gateway with Gordon and DREDGE Integration",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "endpoints": {
            "health": "/health",
            "pipeline": "/pipeline/execute",
            "translate": "/translate",
            "analyze": "/analyze",
            "gordon": "/gordon/*",
            "providers": "/providers/status"
        }
    }


# ============================================================================
# DREDGE PIPELINE ENDPOINTS
# ============================================================================

@app.post("/pipeline/execute")
async def execute_pipeline(request: PipelineRequest):
    """Execute DREDGE pipeline"""
    try:
        result = await dredge_run_pipeline(
            input_data=request.input_data,
            pipeline_type=request.pipeline_type,
            pipeline_id=request.pipeline_id
        )
        return {
            "status": "success",
            "result": result
        }
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/translate")
async def translate(request: TranslationRequest):
    """Execute translation chain"""
    try:
        result = await execute_translation_chain({
            "text": request.text,
            "source_language": request.source_language,
            "target_language": request.target_language
        })
        return {
            "status": "success",
            "result": result
        }
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze")
async def analyze(request: AnalysisRequest):
    """Execute analysis chain"""
    try:
        result = await execute_analysis_chain({
            "query": request.query,
            "context": request.context or {}
        })
        return {
            "status": "success",
            "result": result
        }
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/providers/status")
async def provider_status():
    """Get provider status"""
    try:
        status = await get_provider_status()
        return {
            "status": "success",
            "providers": status
        }
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# GORDON INTEGRATION ENDPOINTS
# ============================================================================

@app.post("/gordon/start")
async def start_gordon_bridge():
    """Start Gordon bridge"""
    global gordon_bridge
    try:
        if gordon_bridge is None:
            gordon_bridge = GordonDREDGEBridge()
        
        logger.info("Gordon bridge initialized")
        return {
            "status": "success",
            "message": "Gordon bridge started"
        }
    except Exception as e:
        logger.error(f"Failed to start Gordon: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/gordon/stop")
async def stop_gordon_bridge():
    """Stop Gordon bridge"""
    global gordon_bridge
    try:
        if gordon_bridge:
            await gordon_bridge.stop()
            gordon_bridge = None
        
        logger.info("Gordon bridge stopped")
        return {
            "status": "success",
            "message": "Gordon bridge stopped"
        }
    except Exception as e:
        logger.error(f"Failed to stop Gordon: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/gordon/capabilities")
async def gordon_capabilities():
    """Get DREDGE capabilities for Gordon"""
    return {
        "status": "success",
        "agent": "DREDGE",
        "version": "2.0.0",
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


@app.post("/gordon/task/execute")
async def execute_gordon_task(request: GordonTaskRequest):
    """Execute task from Gordon"""
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


@app.get("/gordon/status")
async def gordon_status():
    """Get Gordon bridge status"""
    return {
        "status": "success",
        "bridge": {
            "status": "operational" if gordon_bridge else "stopped",
            "gordon": "ready",
            "dredge": "operational"
        }
    }


# ============================================================================
# STATIC PAGES
# ============================================================================

@app.get("/docs/advanced")
async def serve_advanced_dashboard():
    """Serve advanced dashboard HTML"""
    dashboard_path = Path(__file__).parent / "dredge-cli-repo" / "src" / "dredge" / "static" / "frontend_complete_swift.html"
    if dashboard_path.exists():
        return FileResponse(str(dashboard_path), media_type="text/html")
    
    # Fallback to simple dashboard
    simple_path = Path(__file__).parent / "dredge-cli-repo" / "src" / "dredge" / "static" / "dashboard_simple.html"
    if simple_path.exists():
        return FileResponse(str(simple_path), media_type="text/html")
    
    raise HTTPException(status_code=404, detail="Dashboard not found")


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "detail": exc.detail
        }
    )


# ============================================================================
# STARTUP/SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Startup event"""
    logger.info("DREDGE Orion Gateway starting...")
    logger.info("Available at http://127.0.0.1:8080")
    logger.info("Swagger docs at http://127.0.0.1:8080/docs")
    logger.info("Advanced dashboard at http://127.0.0.1:8080/docs/advanced")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event"""
    global gordon_bridge
    if gordon_bridge:
        await gordon_bridge.stop()
    logger.info("DREDGE Orion Gateway shutting down...")


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 80)
    print("  DREDGE Orion Gateway - FastAPI")
    print("=" * 80)
    print("Starting server...")
    print()
    print("  Dashboard: http://127.0.0.1:8080/docs/advanced")
    print("  API Docs:  http://127.0.0.1:8080/docs")
    print("  OpenAPI:   http://127.0.0.1:8080/openapi.json")
    print()
    print("DREDGE Endpoints:")
    print("  POST /pipeline/execute   - Execute pipeline")
    print("  POST /translate          - Translate text")
    print("  POST /analyze            - Analyze content")
    print("  GET  /providers/status   - Provider health")
    print()
    print("Gordon Endpoints:")
    print("  POST /gordon/start       - Start bridge")
    print("  POST /gordon/stop        - Stop bridge")
    print("  GET  /gordon/capabilities- List capabilities")
    print("  POST /gordon/task/execute- Execute task")
    print("  GET  /gordon/status      - Bridge status")
    print()
    print("=" * 80)
    print()
    
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8080,
        reload=False,
        log_level="info"
    )
