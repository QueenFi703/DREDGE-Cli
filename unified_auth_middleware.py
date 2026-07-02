"""
UNIFIED MIDDLEWARE AUTHENTICATION SYSTEM
Single middleware handles all route protection (no per-route dependencies needed)
"""

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
import time
import logging
from datetime import datetime
import httpx

from api_key_manager import APIKeyStore, UsageTracker

logger = logging.getLogger(__name__)


# ============================================================================
# ROUTE PROTECTION CONFIGURATION
# ============================================================================

class RouteConfig:
    """Define which routes require authentication and what level"""
    
    # Routes that don't require authentication
    PUBLIC_ROUTES = {
        "/",
        "/health",
        "/docs",
        "/openapi.json",
        "/redoc"
    }
    
    # Routes that require authentication but track usage optionally
    OPTIONAL_AUTH_ROUTES = {
        "/health"
    }
    
    # Routes that require full admin access
    ADMIN_ROUTES = {
        "/admin",
    }
    
    @staticmethod
    def is_public(path: str) -> bool:
        """Check if route is public"""
        return path in RouteConfig.PUBLIC_ROUTES
    
    @staticmethod
    def is_optional_auth(path: str) -> bool:
        """Check if route has optional auth"""
        return path in RouteConfig.OPTIONAL_AUTH_ROUTES
    
    @staticmethod
    def is_admin_route(path: str) -> bool:
        """Check if route requires admin access"""
        return any(path.startswith(route) for route in RouteConfig.ADMIN_ROUTES)
    
    @staticmethod
    def requires_auth(path: str) -> bool:
        """Check if route requires authentication"""
        # Public routes don't require auth
        if RouteConfig.is_public(path):
            return False
        # Everything else requires auth
        return True


# ============================================================================
# UNIFIED AUTHENTICATION MIDDLEWARE
# ============================================================================

class UnifiedAuthMiddleware:
    """
    Central middleware for API key authentication
    Handles all route protection without per-route dependencies
    """
    
    def __init__(
        self,
        app: FastAPI,
        key_store: APIKeyStore,
        tracker: UsageTracker,
        config: RouteConfig = None
    ):
        self.app = app
        self.key_store = key_store
        self.tracker = tracker
        self.config = config or RouteConfig
    
    async def __call__(self, scope, receive, send):
        """ASGI middleware callable"""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request = Request(scope, receive)
        start_time = time.time()
        
        # Get path
        path = request.url.path
        
        # Check if route requires authentication
        if not self.config.requires_auth(path):
            # Public route - no auth needed
            logger.info(f"Public route accessed: {request.method} {path}")
            await self.app(scope, receive, send)
            return
        
        # Extract API key from header
        api_key = request.headers.get("x-api-key")
        
        # Check if auth is optional for this route
        if self.config.is_optional_auth(path):
            if not api_key:
                # Optional auth - allow without key
                logger.info(f"Optional auth route without key: {request.method} {path}")
                await self.app(scope, receive, send)
                return
        
        # Auth is required - validate key
        if not api_key:
            logger.warning(f"Missing API key: {request.method} {path}")
            
            response = JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "status": "error",
                    "detail": "Missing API key. Provide x-api-key header.",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            await response(scope, receive, send)
            return
        
        # Validate API key
        metadata = self.key_store.validate_key(api_key)
        if not metadata:
            logger.warning(f"Invalid API key: {request.method} {path}")
            
            response = JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "status": "error",
                    "detail": "Invalid or expired API key",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            await response(scope, receive, send)
            return
        
        # Check rate limit
        if self.config.is_admin_route(path):
            # Admin routes require FULL_ACCESS mode
            if metadata.mode.value != "full_access":
                logger.warning(f"Admin access denied: {metadata.key_id} tried {path}")
                
                response = JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={
                        "status": "error",
                        "detail": "Admin access required",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
                
                await response(scope, receive, send)
                return
        else:
            # Regular routes - check rate limit
            allowed, msg = self.tracker.check_rate_limit(metadata.key_id)
            if not allowed:
                logger.warning(f"Rate limit exceeded: {metadata.key_id} - {msg}")
                
                response = JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "status": "error",
                        "detail": msg,
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    headers={"Retry-After": "3600"}
                )
                
                await response(scope, receive, send)
                return
        
        # Store metadata in scope for route handlers
        scope["api_key_metadata"] = metadata
        scope["auth_start_time"] = start_time
        
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "")
        
        # Track response
        response_status = 200
        
        async def send_with_tracking(message):
            nonlocal response_status
            
            if message["type"] == "http.response.start":
                response_status = message["status"]
            
            # Track usage when response is sent
            if message["type"] == "http.response.body":
                duration_ms = (time.time() - start_time) * 1000
                
                self.key_store.record_usage(
                    key_id=metadata.key_id,
                    endpoint=path,
                    method=request.method,
                    response_code=response_status,
                    duration_ms=duration_ms,
                    ip_address=client_ip,
                    user_agent=user_agent
                )
                
                logger.info(
                    f"Request tracked: {metadata.key_id} "
                    f"{request.method} {path} → {response_status} ({duration_ms:.1f}ms)"
                )
            
            # Add rate limit headers
            if message["type"] == "http.response.start":
                remaining = max(0, metadata.requests_limit - metadata.requests_this_month)
                
                headers = list(message.get("headers", []))
                headers.append((b"x-ratelimit-limit", str(metadata.requests_limit).encode()))
                headers.append((b"x-ratelimit-remaining", str(remaining).encode()))
                headers.append((b"x-ratelimit-tier", metadata.tier.value.encode()))
                
                message["headers"] = headers
            
            await send(message)
        
        # Call app with tracking
        await self.app(scope, receive, send_with_tracking)


# ============================================================================
# HELPER FUNCTION TO ATTACH MIDDLEWARE TO APP
# ============================================================================

def add_unified_auth(
    app: FastAPI,
    key_store: APIKeyStore,
    tracker: UsageTracker
):
    """
    Attach unified authentication middleware to FastAPI app
    
    Usage:
        app = FastAPI()
        key_store, tracker = init_api_key_system()
        add_unified_auth(app, key_store, tracker)
        
        # Now all routes are automatically protected!
        # No need for per-route dependencies
    """
    middleware = UnifiedAuthMiddleware(app, key_store, tracker)
    
    # Wrap the app
    app.add_middleware(
        UnifiedAuthMiddleware,
        key_store=key_store,
        tracker=tracker
    )
    
    logger.info("Unified authentication middleware attached")
    return app


# ============================================================================
# ACCESS REQUEST FROM ROUTE HANDLERS
# ============================================================================

def get_auth_metadata(request: Request) -> Optional[Dict[str, Any]]:
    """
    Get authenticated user metadata from request
    
    Usage in route handler:
        @app.get("/protected")
        async def protected_route(request: Request):
            metadata = get_auth_metadata(request)
            if metadata:
                return {"key_id": metadata.key_id}
    """
    return request.scope.get("api_key_metadata")


def require_auth(request: Request) -> Dict[str, Any]:
    """
    Require authentication in route handler
    Raises 401 if not authenticated
    
    Usage:
        @app.get("/protected")
        async def protected_route(request: Request):
            metadata = require_auth(request)
            # If we reach here, metadata is guaranteed to be set
            return {"key_id": metadata.key_id}
    """
    metadata = request.scope.get("api_key_metadata")
    if not metadata:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    return metadata


# ============================================================================
# EXAMPLE: ORION GATEWAY WITH UNIFIED MIDDLEWARE
# ============================================================================

def create_orion_with_middleware(key_store: APIKeyStore, tracker: UsageTracker) -> FastAPI:
    """
    Create Orion Gateway with unified middleware
    All routes protected automatically
    """
    
    app = FastAPI(
        title="Orion Gateway - Middleware Auth",
        description="Inference gateway with unified middleware authentication",
        version="2.0.0"
    )
    
    # Add unified middleware - ONE LINE handles all auth!
    app.add_middleware(
        UnifiedAuthMiddleware,
        key_store=key_store,
        tracker=tracker
    )
    
    # ================================================================
    # ALL THESE ROUTES ARE AUTOMATICALLY PROTECTED
    # No per-route @requires_auth decorators needed!
    # ================================================================
    
    @app.get("/")
    async def root():
        """Public root endpoint"""
        return {
            "service": "Orion Gateway",
            "version": "2.0.0",
            "auth": "Unified middleware (automatic)"
        }
    
    @app.get("/health")
    async def health(request: Request):
        """Health check (optional auth)"""
        metadata = get_auth_metadata(request)
        return {
            "status": "healthy",
            "tracked": metadata is not None,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @app.post("/invoke")
    async def invoke(request: Request):
        """Main inference endpoint (auth required - automatic)"""
        # Metadata automatically set by middleware
        metadata = get_auth_metadata(request)
        
        try:
            body = await request.json()
        except:
            raise HTTPException(status_code=400, detail="Invalid JSON")
        
        input_text = body.get("input", "")
        if not input_text:
            raise HTTPException(status_code=400, detail="'input' field required")
        
        return {
            "request_id": f"req_{int(time.time() * 1000)}",
            "status": "success",
            "result": {"reasoning": "Mock result"},
            "usage": {
                "tier": metadata.tier.value,
                "requests_this_month": metadata.requests_this_month,
                "requests_limit": metadata.requests_limit,
                "usage_percent": f"{metadata.usage_percent:.1f}%"
            }
        }
    
    @app.get("/usage")
    async def usage_endpoint(request: Request):
        """Get usage stats (auth required - automatic)"""
        metadata = get_auth_metadata(request)
        
        # Middleware ensures we have metadata
        from fastapi_auth_middleware import UsageTracker
        
        report = tracker.get_usage_report(metadata.key_id)
        
        return {
            "status": "success",
            "usage": report
        }
    
    @app.post("/admin/keys/create")
    async def admin_create_key(request: Request):
        """Create API key (admin only - automatic)"""
        metadata = get_auth_metadata(request)
        
        # Middleware already checked full_access
        from api_key_manager import KeyTier, KeyMode
        
        body = await request.json()
        
        new_key, new_metadata = key_store.create_key(
            name=body.get("name", "Unnamed"),
            tier=KeyTier(body.get("tier", "starter")),
            mode=KeyMode(body.get("mode", "invoke_only")),
            created_by=metadata.key_id,
            environment=body.get("environment", "production")
        )
        
        return {
            "status": "success",
            "key": new_key,
            "key_id": new_metadata.key_id,
            "metadata": new_metadata.to_dict()
        }
    
    @app.get("/admin/keys/list")
    async def admin_list_keys(request: Request):
        """List all keys (admin only - automatic)"""
        metadata = get_auth_metadata(request)
        
        keys = key_store.list_keys()
        
        return {
            "status": "success",
            "count": len(keys),
            "keys": keys
        }
    
    @app.get("/admin/stats")
    async def admin_stats(request: Request):
        """View statistics (admin only - automatic)"""
        metadata = get_auth_metadata(request)
        
        total_keys = len(key_store.keys)
        active_keys = sum(1 for k in key_store.keys.values() if k.status.value == "active")
        total_requests = sum(len(v) for v in key_store.usage.values())
        
        return {
            "status": "success",
            "stats": {
                "total_keys": total_keys,
                "active_keys": active_keys,
                "total_requests": total_requests
            }
        }
    
    return app


if __name__ == "__main__":
    import uvicorn
    from api_key_manager import init_api_key_system, KeyTier, KeyMode
    
    # Initialize
    key_store, tracker = init_api_key_system()
    
    # Create test keys
    test_key, _ = key_store.create_key(
        name="Test Key",
        tier=KeyTier.PRO,
        mode=KeyMode.INVOKE_ONLY,
        created_by="system"
    )
    print(f"Test key: {test_key}")
    
    admin_key, _ = key_store.create_key(
        name="Admin Key",
        tier=KeyTier.ENTERPRISE,
        mode=KeyMode.FULL_ACCESS,
        created_by="system"
    )
    print(f"Admin key: {admin_key}")
    
    # Create app with unified middleware
    app = create_orion_with_middleware(key_store, tracker)
    
    print("\nStarting with unified middleware authentication...")
    print("All routes automatically protected!")
    print("No per-route dependencies needed!")
    
    uvicorn.run(app, host="127.0.0.1", port=8003)
