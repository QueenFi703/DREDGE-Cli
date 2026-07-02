"""
COMPLETE ORION GATEWAY WITH API KEY AUTHENTICATION
Full production-ready implementation
"""

from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import time
import logging
from datetime import datetime

from api_key_manager import (
    init_api_key_system, KeyTier, KeyMode
)
from fastapi_auth_middleware import APIKeyDependencies, RateLimitHeaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# INITIALIZATION
# ============================================================================

# Initialize API key system
key_store, tracker = init_api_key_system("./data/orion_api_keys.json")

# Initialize FastAPI
app = FastAPI(
    title="Orion Gateway - Production",
    description="AI Inference Gateway with secure API key authentication",
    version="2.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize dependencies
deps = APIKeyDependencies(key_store, tracker)


# ============================================================================
# MIDDLEWARE FOR REQUEST TRACKING
# ============================================================================

@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Track requests for usage billing"""
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
    """API information"""
    return {
        "service": "Orion Gateway",
        "version": "2.0.0",
        "status": "operational",
        "auth": {
            "required": True,
            "method": "x-api-key header",
            "format": "orion_key_xxxxxxxx"
        },
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health(key: APIKeyMetadata = Depends(deps.optional_api_key)):
    """
    Health check endpoint
    Optional: Provide x-api-key to track usage
    """
    return {
        "status": "healthy",
        "service": "orion-gateway",
        "timestamp": datetime.utcnow().isoformat(),
        "tracked": key is not None
    }


# ============================================================================
# MAIN INFERENCE ENDPOINT - REQUIRES AUTH
# ============================================================================

@app.post("/invoke")
async def invoke(
    request: Request,
    key: APIKeyMetadata = Depends(deps.verify_rate_limit)
):
    """
    Main inference endpoint
    
    Required header:
        x-api-key: <your_api_key>
    
    Request body:
        {
            "input": "Your question or prompt",
            "mode": "standard|deep|transform|analyze",
            "context": {...}
        }
    
    Response headers:
        X-RateLimit-Limit: Total requests available this month
        X-RateLimit-Remaining: Requests remaining this month
        X-RateLimit-Tier: Your subscription tier
    """
    try:
        body = await request.json()
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    
    input_text = body.get("input")
    mode = body.get("mode", "standard")
    context = body.get("context", {})
    
    if not input_text:
        raise HTTPException(status_code=400, detail="'input' field required")
    
    # Validate mode
    valid_modes = ["standard", "deep", "transform", "analyze"]
    if mode not in valid_modes:
        raise HTTPException(status_code=400, detail=f"Invalid mode. Choose from: {valid_modes}")
    
    # Simulate inference
    logger.info(f"Inference request from {key.key_id} (mode: {mode})")
    
    response = {
        "request_id": "req_" + str(int(time.time() * 1000))[-8:],
        "status": "success",
        "mode": mode,
        "input_length": len(input_text),
        "result": {
            "reasoning": "Mock inference result...",
            "confidence": 0.92,
            "tokens_used": len(input_text) // 4
        },
        "usage": {
            "tier": key.tier.value,
            "requests_this_month": key.requests_this_month,
            "requests_limit": key.requests_limit,
            "usage_percent": f"{key.usage_percent:.1f}%"
        }
    }
    
    return response


# ============================================================================
# USAGE ENDPOINT - REQUIRES AUTH
# ============================================================================

@app.get("/usage")
async def usage(key: APIKeyMetadata = Depends(deps.verify_api_key)):
    """
    Get current usage for your API key
    
    Required header:
        x-api-key: <your_api_key>
    
    Returns:
        - Current usage this month
        - Monthly limit
        - Reset date (1st of next month)
        - Breakdown by endpoint
    """
    
    report = tracker.get_usage_report(key.key_id)
    
    if not report:
        raise HTTPException(status_code=404, detail="Usage report not found")
    
    return {
        "status": "success",
        "key_id": key.key_id,
        "tier": key.tier.value,
        "usage": report
    }


# ============================================================================
# ADMIN ENDPOINTS - REQUIRES FULL_ACCESS
# ============================================================================

@app.post("/admin/keys/create")
async def admin_create_key(
    request: Request,
    key: APIKeyMetadata = Depends(deps.verify_api_key)
):
    """
    Admin: Create new API key
    
    Required header:
        x-api-key: <admin_key_with_full_access>
    
    Request body:
        {
            "name": "My New Key",
            "tier": "pro|starter|free|enterprise",
            "mode": "invoke_only|read_only|full_access",
            "environment": "development|staging|production"
        }
    """
    
    # Check permissions
    if key.mode != KeyMode.FULL_ACCESS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only FULL_ACCESS keys can create new keys"
        )
    
    body = await request.json()
    
    # Validate input
    required_fields = ["name"]
    for field in required_fields:
        if field not in body:
            raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
    
    # Create new key
    new_key, metadata = key_store.create_key(
        name=body.get("name"),
        tier=KeyTier(body.get("tier", "starter")),
        mode=KeyMode(body.get("mode", "invoke_only")),
        created_by=key.key_id,
        environment=body.get("environment", "production")
    )
    
    logger.info(f"New API key created by {key.key_id}: {metadata.key_id}")
    
    return {
        "status": "success",
        "message": "API key created successfully",
        "key": new_key,  # IMPORTANT: Only returned once!
        "key_id": metadata.key_id,
        "metadata": metadata.to_dict(),
        "warning": "Store this key in a secure location. You won't be able to see it again."
    }


@app.get("/admin/keys/list")
async def admin_list_keys(
    key: APIKeyMetadata = Depends(deps.verify_api_key)
):
    """
    Admin: List all API keys (without plaintext keys)
    
    Required header:
        x-api-key: <admin_key_with_full_access>
    """
    
    if key.mode != KeyMode.FULL_ACCESS:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    keys = key_store.list_keys()
    
    return {
        "status": "success",
        "count": len(keys),
        "keys": keys
    }


@app.post("/admin/keys/{key_id}/revoke")
async def admin_revoke_key(
    key_id: str,
    key: APIKeyMetadata = Depends(deps.verify_api_key)
):
    """
    Admin: Revoke an API key (disable it)
    
    Required header:
        x-api-key: <admin_key_with_full_access>
    
    Path parameter:
        key_id: The key ID to revoke (e.g., key_abc123xyz)
    """
    
    if key.mode != KeyMode.FULL_ACCESS:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    success = key_store.revoke_key(key_id)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Key {key_id} not found")
    
    logger.info(f"API key revoked by {key.key_id}: {key_id}")
    
    return {
        "status": "success",
        "message": f"Key {key_id} has been revoked"
    }


@app.get("/admin/stats")
async def admin_stats(key: APIKeyMetadata = Depends(deps.verify_api_key)):
    """
    Admin: View system statistics
    
    Required header:
        x-api-key: <admin_key_with_full_access>
    """
    
    if key.mode != KeyMode.FULL_ACCESS:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    # Calculate stats
    total_keys = len(key_store.keys)
    active_keys = sum(1 for k in key_store.keys.values() if k.status.value == "active")
    total_requests = sum(len(v) for v in key_store.usage.values())
    
    # By tier
    by_tier = {}
    for k in key_store.keys.values():
        tier = k.tier.value
        by_tier[tier] = by_tier.get(tier, 0) + 1
    
    return {
        "status": "success",
        "statistics": {
            "total_keys": total_keys,
            "active_keys": active_keys,
            "revoked_keys": sum(1 for k in key_store.keys.values() if k.status.value == "revoked"),
            "total_requests_all_time": total_requests,
            "keys_by_tier": by_tier,
            "timestamp": datetime.utcnow().isoformat()
        }
    }


@app.get("/admin/usage/{key_id}")
async def admin_key_usage(
    key_id: str,
    key: APIKeyMetadata = Depends(deps.verify_api_key)
):
    """
    Admin: Get detailed usage for specific key
    
    Required header:
        x-api-key: <admin_key_with_full_access>
    """
    
    if key.mode != KeyMode.FULL_ACCESS:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    stats = key_store.get_usage_stats(key_id)
    
    if not stats:
        raise HTTPException(status_code=404, detail=f"Key {key_id} not found")
    
    return {
        "status": "success",
        "usage": stats
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "detail": exc.detail,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# ============================================================================
# STARTUP
# ============================================================================

@app.on_event("startup")
async def startup():
    """Initialize test keys on startup"""
    
    # Check if we have any keys
    if len(key_store.keys) == 0:
        logger.info("Creating default test keys...")
        
        # Create test invoke key
        invoke_key, _ = key_store.create_key(
            name="Test Invoke Key",
            tier=KeyTier.PRO,
            mode=KeyMode.INVOKE_ONLY,
            created_by="system",
            environment="development"
        )
        logger.info(f"Test invoke key: {invoke_key}")
        
        # Create admin key
        admin_key, _ = key_store.create_key(
            name="Admin Key",
            tier=KeyTier.ENTERPRISE,
            mode=KeyMode.FULL_ACCESS,
            created_by="system",
            environment="development"
        )
        logger.info(f"Admin key: {admin_key}")
    
    logger.info("Orion Gateway with API Key Auth started successfully")


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("  ORION GATEWAY - PRODUCTION WITH API KEY AUTHENTICATION")
    print("="*80)
    print()
    print("Endpoints:")
    print()
    print("  PUBLIC (No auth):")
    print("    GET  /                 - API info")
    print("    GET  /health           - Health check (optional key for tracking)")
    print("    GET  /docs             - Swagger documentation")
    print()
    print("  PROTECTED (Auth required with x-api-key header):")
    print("    POST /invoke           - Main inference endpoint")
    print("    GET  /usage            - Get your usage statistics")
    print()
    print("  ADMIN (Full access required):")
    print("    POST /admin/keys/create           - Create new API key")
    print("    GET  /admin/keys/list             - List all keys")
    print("    POST /admin/keys/{key_id}/revoke  - Revoke a key")
    print("    GET  /admin/stats                 - View system statistics")
    print("    GET  /admin/usage/{key_id}        - View usage for key")
    print()
    print("="*80)
    print("Starting on http://127.0.0.1:8002")
    print("Swagger UI: http://127.0.0.1:8002/docs")
    print("="*80)
    print()
    
    uvicorn.run(
        "orion_gateway_authenticated:app",
        host="127.0.0.1",
        port=8002,
        reload=False,
        log_level="info"
    )
