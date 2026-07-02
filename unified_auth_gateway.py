"""
ORION GATEWAY - UNIFIED AUTH PROXY
Routes requests to 8000/8001/8080 with API key authentication
Runs on port 9000 as central authentication gateway
"""

from fastapi import FastAPI, Request, Header, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
import logging
from typing import Optional
import time

from api_key_manager import init_api_key_system
from unified_auth_middleware import UnifiedAuthMiddleware, get_auth_metadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# PORT CONFIGURATION
# ============================================================================

PORT_CONFIG = {
    "orion": {
        "port": 8080,
        "url": "http://127.0.0.1:8080",
        "endpoints": ["/invoke", "/usage", "/admin/stats"],
        "auth_required": True,
        "description": "Orion Gateway - Main inference"
    },
    "dredge": {
        "port": 3001,
        "url": "http://127.0.0.1:3001",
        "endpoints": ["/api/architecture/*", "/api/gordon/*"],
        "auth_required": True,
        "description": "DREDGE Pipeline - Advanced operations"
    },
    "advanced": {
        "port": 8000,
        "url": "http://127.0.0.1:8000",
        "endpoints": ["/advanced", "/features"],
        "auth_required": True,
        "description": "Advanced features - Dashboard"
    },
    "mcp": {
        "port": 8001,
        "url": "http://127.0.0.1:8001",
        "endpoints": ["/mcp/*"],
        "auth_required": False,
        "description": "MCP Gateway - Optional auth"
    }
}

# ============================================================================
# INITIALIZATION
# ============================================================================

app = FastAPI(
    title="Unified Auth Gateway",
    description="Central authentication proxy for all DREDGE/Orion services",
    version="2.0.0"
)

# Initialize API key system
key_store, tracker = init_api_key_system("./data/unified_api_keys.json")

# Add unified middleware - automatically protects all routes
app.add_middleware(
    UnifiedAuthMiddleware,
    key_store=key_store,
    tracker=tracker
)


# ============================================================================
# MIDDLEWARE FOR REQUEST TRACKING
# ============================================================================

@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Track all requests across all services"""
    start_time = time.time()
    
    # Extract API key
    api_key = request.headers.get("x-api-key")
    metadata = None
    
    if api_key:
        metadata = key_store.validate_key(api_key)
        request.state.key_metadata = metadata
    
    response = await call_next(request)
    
    # Record usage if key was provided
    if metadata and api_key:
        duration_ms = (time.time() - start_time) * 1000
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "")
        
        key_store.record_usage(
            key_id=metadata.key_id,
            endpoint=request.url.path,
            method=request.method,
            response_code=response.status_code,
            duration_ms=duration_ms,
            ip_address=client_ip,
            user_agent=user_agent
        )
        
        # Add rate limit headers
        headers = RateLimitHeaders.get_headers(metadata)
        for key, value in headers.items():
            response.headers[key] = value
    
    return response


# ============================================================================
# PUBLIC ENDPOINTS (No auth required)
# ============================================================================

@app.get("/")
async def root():
    """Gateway information"""
    return {
        "service": "Unified Auth Gateway",
        "version": "2.0.0",
        "status": "operational",
        "auth_required": "x-api-key header for most endpoints",
        "services": PORT_CONFIG,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health():
    """Health check - checks all backend services"""
    services_status = {}
    
    async with httpx.AsyncClient(timeout=2) as client:
        for service_name, config in PORT_CONFIG.items():
            try:
                resp = await client.get(f"{config['url']}/health")
                services_status[service_name] = {
                    "status": "operational",
                    "code": resp.status_code,
                    "port": config['port']
                }
            except Exception as e:
                services_status[service_name] = {
                    "status": "error",
                    "error": str(e),
                    "port": config['port']
                }
    
    return {
        "gateway": "operational",
        "timestamp": time.time(),
        "services": services_status
    }


@app.get("/services")
async def list_services():
    """List all available services"""
    return {
        "services": PORT_CONFIG
    }


# ============================================================================
# PROXY ENDPOINTS
# ============================================================================

@app.api_route("/{service}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_request(
    service: str,
    path: str,
    request: Request,
    x_api_key: Optional[str] = Header(None)
):
    """
    Universal proxy endpoint
    
    Usage:
        GET  /orion/health
        POST /orion/invoke
        GET  /dredge/api/architecture/health
        POST /dredge/pipeline/execute
    """
    
    # Get service config
    if service not in PORT_CONFIG:
        raise HTTPException(
            status_code=404,
            detail=f"Service '{service}' not found. Available: {list(PORT_CONFIG.keys())}"
        )
    
    config = PORT_CONFIG[service]
    
    # Check authentication if required
    if config["auth_required"]:
        if not x_api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"API key required for {service} service"
            )
        
        metadata = key_store.validate_key(x_api_key)
        if not metadata:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
        
        # Check rate limit
        allowed, msg = tracker.check_rate_limit(metadata.key_id)
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=msg
            )
    
    # Build target URL
    target_url = f"{config['url']}/{path}"
    if request.url.query:
        target_url += f"?{request.url.query}"
    
    # Get request body
    body = await request.body() if request.method in ["POST", "PUT", "PATCH"] else None
    
    # Forward request
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.request(
                method=request.method,
                url=target_url,
                content=body,
                headers={
                    key: value for key, value in request.headers.items()
                    if key.lower() not in ["host", "content-length"]
                }
            )
            
            # Log request
            logger.info(
                f"Proxy: {service} {request.method} /{path} -> {response.status_code}"
            )
            
            return JSONResponse(
                status_code=response.status_code,
                content=response.json() if response.text else {}
            )
    
    except Exception as e:
        logger.error(f"Proxy error: {service} /{path} - {e}")
        raise HTTPException(
            status_code=502,
            detail=f"Error proxying to {service}: {str(e)}"
        )


# ============================================================================
# ADMIN ENDPOINTS (Requires API key authentication)
# ============================================================================

@app.post("/admin/keys/create")
async def admin_create_key(
    request: Request,
    x_api_key: str = Header(None)
):
    """
    Create new API key
    
    Requires: Admin API key with full_access mode
    """
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Admin key required")
    
    # Validate admin key
    key_metadata = key_store.validate_key(x_api_key)
    if not key_metadata or key_metadata.mode.value != "full_access":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    body = await request.json()
    
    # Create new key
    new_key, metadata = key_store.create_key(
        name=body.get("name", "Unnamed Key"),
        tier=body.get("tier", "starter"),
        mode=body.get("mode", "invoke_only"),
        created_by=key_metadata.key_id,
        environment=body.get("environment", "production")
    )
    
    logger.info(f"New API key created by {key_metadata.key_id}: {metadata.key_id}")
    
    return {
        "status": "success",
        "key": new_key,
        "key_id": metadata.key_id,
        "metadata": metadata.to_dict(),
        "warning": "Store this key securely. You won't see it again."
    }


@app.get("/admin/keys/list")
async def admin_list_keys(x_api_key: str = Header(None)):
    """List all API keys"""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Admin key required")
    
    key_metadata = key_store.validate_key(x_api_key)
    if not key_metadata or key_metadata.mode.value != "full_access":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    keys = key_store.list_keys()
    
    return {
        "status": "success",
        "count": len(keys),
        "keys": keys
    }


@app.get("/admin/stats")
async def admin_stats(x_api_key: str = Header(None)):
    """View gateway statistics"""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Admin key required")
    
    key_metadata = key_store.validate_key(x_api_key)
    if not key_metadata or key_metadata.mode.value != "full_access":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Calculate stats
    total_keys = len(key_store.keys)
    active_keys = sum(1 for k in key_store.keys.values() if k.status.value == "active")
    total_requests = sum(len(v) for v in key_store.usage.values())
    
    return {
        "status": "success",
        "gateway_stats": {
            "total_keys": total_keys,
            "active_keys": active_keys,
            "total_requests": total_requests,
            "services": {
                name: {
                    "port": config["port"],
                    "status": "configured",
                    "auth_required": config["auth_required"]
                }
                for name, config in PORT_CONFIG.items()
            }
        }
    }


@app.post("/admin/keys/{key_id}/revoke")
async def admin_revoke_key(
    key_id: str,
    x_api_key: str = Header(None)
):
    """Revoke an API key"""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Admin key required")
    
    key_metadata = key_store.validate_key(x_api_key)
    if not key_metadata or key_metadata.mode.value != "full_access":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    success = key_store.revoke_key(key_id)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Key {key_id} not found")
    
    logger.info(f"API key revoked by {key_metadata.key_id}: {key_id}")
    
    return {
        "status": "success",
        "message": f"Key {key_id} revoked"
    }


@app.get("/admin/usage/{key_id}")
async def admin_usage(
    key_id: str,
    x_api_key: str = Header(None)
):
    """Get usage for a specific key"""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Admin key required")
    
    key_metadata = key_store.validate_key(x_api_key)
    if not key_metadata or key_metadata.mode.value != "full_access":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    stats = key_store.get_usage_stats(key_id)
    
    if not stats:
        raise HTTPException(status_code=404, detail=f"Key {key_id} not found")
    
    return {
        "status": "success",
        "usage": stats
    }


# ============================================================================
# USER ENDPOINTS (Requires valid API key)
# ============================================================================

@app.get("/usage")
async def user_usage(x_api_key: str = Header(None)):
    """Get your API usage"""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key required")
    
    metadata = key_store.validate_key(x_api_key)
    if not metadata:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    report = tracker.get_usage_report(metadata.key_id)
    
    return {
        "status": "success",
        "usage": report
    }


@app.get("/key/info")
async def key_info(x_api_key: str = Header(None)):
    """Get info about your API key"""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key required")
    
    metadata = key_store.validate_key(x_api_key)
    if not metadata:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return {
        "status": "success",
        "key": metadata.to_dict()
    }


# ============================================================================
# STARTUP
# ============================================================================

@app.on_event("startup")
async def startup():
    """Initialize on startup"""
    
    # Create default keys if none exist
    if len(key_store.keys) == 0:
        logger.info("Creating default API keys...")
        
        from api_key_manager import KeyTier, KeyMode
        
        # Test key
        test_key, _ = key_store.create_key(
            name="Test Key",
            tier=KeyTier.PRO,
            mode=KeyMode.INVOKE_ONLY,
            created_by="system",
            environment="development"
        )
        logger.info(f"Test key: {test_key}")
        
        # Admin key
        admin_key, _ = key_store.create_key(
            name="Admin Key",
            tier=KeyTier.ENTERPRISE,
            mode=KeyMode.FULL_ACCESS,
            created_by="system",
            environment="development"
        )
        logger.info(f"Admin key: {admin_key}")
    
    logger.info("Unified Auth Gateway started")


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*80)
    print("  UNIFIED AUTH GATEWAY - CENTRAL AUTHENTICATION PROXY")
    print("="*80)
    print()
    print("Gateway Port: 9000")
    print()
    print("Services Behind Gateway:")
    print()
    for name, config in PORT_CONFIG.items():
        print(f"  {name.upper():12} | Port {config['port']} | {config['description']}")
    print()
    print("Route Pattern: /<service>/<path>")
    print()
    print("Examples:")
    print("  GET  http://127.0.0.1:9000/orion/health")
    print("  POST http://127.0.0.1:9000/orion/invoke")
    print("  GET  http://127.0.0.1:9000/dredge/api/architecture/health")
    print()
    print("Admin Endpoints:")
    print("  POST /admin/keys/create          - Create API key")
    print("  GET  /admin/keys/list            - List all keys")
    print("  POST /admin/keys/{id}/revoke     - Revoke key")
    print("  GET  /admin/stats                - View statistics")
    print()
    print("Swagger: http://127.0.0.1:9000/docs")
    print("="*80)
    print()
    
    uvicorn.run(
        "unified_auth_gateway:app",
        host="127.0.0.1",
        port=9000,
        reload=False,
        log_level="info"
    )
