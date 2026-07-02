"""
Orion ↔ DREDGE ↔ Gordon Bridge
Proxy layer that integrates DREDGE pipelines and Gordon multi-agent with Orion Gateway
Runs on http://127.0.0.1:9999 and proxies to 8080
"""

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
import json
import logging
from typing import Optional, Dict, Any
import sys
import os

# Setup path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dredge-cli-repo', 'src'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DREDGE-Orion-Gordon Bridge",
    description="Unified bridge connecting Orion Gateway with DREDGE and Gordon",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
ORION_BASE = "http://127.0.0.1:8080"
DREDGE_BASE = "http://127.0.0.1:3001" if os.environ.get("DREDGE_ENABLED") else None


# ============================================================================
# PROXY ENDPOINTS
# ============================================================================

@app.get("/health")
async def health():
    """Health check for bridge"""
    orion_health = await check_orion_health()
    dredge_health = await check_dredge_health() if DREDGE_BASE else None
    
    return {
        "status": "healthy",
        "bridge": "operational",
        "orion": orion_health,
        "dredge": dredge_health or "not_configured",
        "gordon": "ready",
        "endpoints": {
            "orion_proxy": "/orion/*",
            "dredge": "/dredge/*",
            "gordon": "/gordon/*"
        }
    }


async def check_orion_health():
    """Check Orion health"""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{ORION_BASE}/health")
            return {"status": "operational", "code": resp.status_code}
    except Exception as e:
        return {"status": "error", "error": str(e)}


async def check_dredge_health():
    """Check DREDGE health"""
    if not DREDGE_BASE:
        return None
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{DREDGE_BASE}/health")
            return {"status": "operational", "code": resp.status_code}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/")
async def root():
    """Bridge root"""
    return {
        "name": "DREDGE-Orion-Gordon Bridge",
        "version": "2.0.0",
        "components": {
            "orion": "Gateway for inference",
            "dredge": "Pipeline orchestration",
            "gordon": "Multi-agent coordination"
        },
        "docs": "/docs",
        "health": "/health",
        "modes": {
            "standard": "Fast Orion inference",
            "dredge_pipeline": "DREDGE DAG execution",
            "gordon_orchestrated": "Multi-agent coordination"
        }
    }


# ============================================================================
# ORION PROXY ENDPOINTS
# ============================================================================

@app.post("/orion/invoke")
async def invoke(request: Dict[str, Any], x_api_key: Optional[str] = Header(None)):
    """Proxy to Orion invoke endpoint"""
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            headers = {"x-api-key": x_api_key} if x_api_key else {}
            resp = await client.post(
                f"{ORION_BASE}/invoke",
                json=request,
                headers=headers
            )
            return resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/orion/usage")
async def orion_usage(x_api_key: Optional[str] = Header(None)):
    """Proxy to Orion usage endpoint"""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            headers = {"x-api-key": x_api_key} if x_api_key else {}
            resp = await client.get(
                f"{ORION_BASE}/usage",
                headers=headers
            )
            return resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/orion/docs")
async def orion_docs():
    """Get Orion OpenAPI spec"""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{ORION_BASE}/openapi.json")
            return resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# DREDGE INTEGRATION ENDPOINTS
# ============================================================================

@app.post("/dredge/pipeline")
async def dredge_pipeline(request: Dict[str, Any]):
    """Execute DREDGE pipeline"""
    if not DREDGE_BASE:
        raise HTTPException(status_code=503, detail="DREDGE not configured")
    
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{DREDGE_BASE}/api/architecture/pipeline/execute",
                json=request
            )
            return resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/dredge/translate")
async def dredge_translate(request: Dict[str, Any]):
    """Execute DREDGE translation"""
    if not DREDGE_BASE:
        raise HTTPException(status_code=503, detail="DREDGE not configured")
    
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{DREDGE_BASE}/api/architecture/translate",
                json=request
            )
            return resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/dredge/analyze")
async def dredge_analyze(request: Dict[str, Any]):
    """Execute DREDGE analysis"""
    if not DREDGE_BASE:
        raise HTTPException(status_code=503, detail="DREDGE not configured")
    
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{DREDGE_BASE}/api/architecture/analyze",
                json=request
            )
            return resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dredge/health")
async def dredge_health():
    """Get DREDGE health"""
    if not DREDGE_BASE:
        return {"status": "disabled", "reason": "DREDGE_ENABLED not set"}
    
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{DREDGE_BASE}/health")
            return resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# GORDON INTEGRATION ENDPOINTS
# ============================================================================

@app.post("/gordon/invoke")
async def gordon_invoke(request: Dict[str, Any]):
    """Unified invoke through Gordon coordination"""
    # Route based on task type
    task_type = request.get("type", "orion")
    
    if task_type == "dredge":
        return await dredge_pipeline(request.get("payload", {}))
    elif task_type == "orion":
        return await invoke(request.get("payload", {}))
    else:
        raise HTTPException(status_code=400, detail=f"Unknown task type: {task_type}")


@app.get("/gordon/capabilities")
async def gordon_capabilities():
    """List all available capabilities from Gordon perspective"""
    return {
        "status": "success",
        "agent": "DREDGE-Orion-Gordon",
        "capabilities": [
            {
                "name": "orion_inference",
                "description": "Fast intent shaping with Orion",
                "endpoint": "/orion/invoke",
                "modes": ["standard", "deep", "transform", "analyze"]
            },
            {
                "name": "dredge_pipeline",
                "description": "DREDGE DAG pipeline execution",
                "endpoint": "/dredge/pipeline",
                "features": ["caching", "failover", "telemetry"]
            },
            {
                "name": "translation",
                "description": "Multi-provider translation",
                "endpoint": "/dredge/translate",
                "providers": ["google", "deepseek"]
            },
            {
                "name": "analysis",
                "description": "Semantic analysis with fallback",
                "endpoint": "/dredge/analyze",
                "modes": ["semantic", "syntactic", "pragmatic"]
            },
            {
                "name": "unified_routing",
                "description": "Route to Orion or DREDGE based on task type",
                "endpoint": "/gordon/invoke",
                "routing": "automatic"
            }
        ]
    }


@app.get("/gordon/status")
async def gordon_status():
    """Get bridge status"""
    orion = await check_orion_health()
    dredge = await check_dredge_health() if DREDGE_BASE else None
    
    return {
        "status": "operational",
        "components": {
            "orion": orion,
            "dredge": dredge or "disabled",
            "gordon": "operational",
            "bridge": "connected"
        }
    }


# ============================================================================
# COMPOSITE ENDPOINTS
# ============================================================================

@app.post("/execute")
async def unified_execute(request: Dict[str, Any]):
    """
    Unified execution endpoint
    Automatically routes to best component based on request
    """
    input_text = request.get("input", "")
    mode = request.get("mode", "auto")
    
    # Auto-routing logic
    if mode == "auto":
        # If input looks like it needs DAG/pipeline - use DREDGE
        if any(word in input_text.lower() for word in ["pipeline", "chain", "flow", "dag"]):
            return await dredge_pipeline({
                "input_data": request,
                "pipeline_type": "standard"
            })
        # Otherwise use Orion for fast inference
        else:
            return await invoke({
                "input": input_text,
                "mode": "standard",
                "context": request.get("context")
            })
    
    elif mode == "orion":
        return await invoke(request)
    elif mode == "dredge":
        return await dredge_pipeline(request)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown mode: {mode}")


@app.post("/batch")
async def batch_execute(requests: list[Dict[str, Any]]):
    """Batch execute multiple requests"""
    results = []
    for req in requests:
        try:
            result = await unified_execute(req)
            results.append({"status": "success", "result": result})
        except Exception as e:
            results.append({"status": "error", "error": str(e)})
    return {"results": results}


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 80)
    print("  DREDGE-Orion-Gordon Bridge")
    print("=" * 80)
    print()
    print("Bridge Configuration:")
    print(f"  Orion Gateway:  {ORION_BASE}")
    print(f"  DREDGE:         {DREDGE_BASE or 'Not configured'}")
    print(f"  Gordon:         Ready")
    print()
    print("Available Endpoints:")
    print("  /health                   - Bridge health")
    print("  /gordon/capabilities      - List capabilities")
    print("  /gordon/status            - Component status")
    print()
    print("Unified Routing:")
    print("  POST /execute             - Auto-route to best component")
    print("  POST /batch               - Execute multiple requests")
    print()
    print("Component Access:")
    print("  /orion/*                  - Direct Orion access")
    print("  /dredge/*                 - Direct DREDGE access")
    print("  /gordon/*                 - Gordon coordination")
    print()
    print("=" * 80)
    print()
    
    uvicorn.run(
        "orion_dredge_gordon_bridge:app",
        host="127.0.0.1",
        port=9999,
        reload=False,
        log_level="info"
    )
