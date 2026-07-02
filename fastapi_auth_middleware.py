"""
FastAPI Security Dependencies & Middleware
Integrate API key authentication with FastAPI dependency injection
"""

from fastapi import FastAPI, Depends, HTTPException, Header, Request, status
from fastapi.security import HTTPBearer
from typing import Optional, Dict, Any
import time
import logging
from functools import wraps

from api_key_manager import (
    APIKeyStore, UsageTracker, APIKeyMetadata, KeyMode
)

logger = logging.getLogger(__name__)

# ============================================================================
# SECURITY SCHEME
# ============================================================================

security = HTTPBearer()


# ============================================================================
# DEPENDENCY FUNCTIONS
# ============================================================================

class APIKeyDependencies:
    """FastAPI dependency injection functions for API key auth"""
    
    def __init__(self, key_store: APIKeyStore, tracker: UsageTracker):
        self.key_store = key_store
        self.tracker = tracker
    
    async def verify_api_key(
        self,
        request: Request,
        x_api_key: Optional[str] = Header(None)
    ) -> APIKeyMetadata:
        """
        Dependency: Verify API key from header
        
        Usage in FastAPI:
            @app.get("/endpoint")
            async def endpoint(key: APIKeyMetadata = Depends(deps.verify_api_key)):
                return {"key_id": key.key_id}
        """
        if not x_api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing API key. Provide x-api-key header.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Validate key
        metadata = self.key_store.validate_key(x_api_key)
        if not metadata:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired API key",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return metadata
    
    async def verify_api_key_with_mode(
        self,
        required_mode: KeyMode
    ):
        """
        Dependency factory: Verify API key with specific mode requirement
        
        Usage in FastAPI:
            invoke_deps = deps.verify_api_key_with_mode(KeyMode.INVOKE_ONLY)
            
            @app.post("/invoke")
            async def invoke(key: APIKeyMetadata = Depends(invoke_deps)):
                return {"status": "ok"}
        """
        async def _verify(
            request: Request,
            x_api_key: Optional[str] = Header(None)
        ) -> APIKeyMetadata:
            if not x_api_key:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Missing API key",
                )
            
            metadata = self.key_store.validate_key(x_api_key)
            if not metadata:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid API key",
                )
            
            # Check if key has required mode
            if required_mode != KeyMode.FULL_ACCESS:
                if metadata.mode != required_mode and metadata.mode != KeyMode.FULL_ACCESS:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"API key does not have {required_mode} mode"
                    )
            
            return metadata
        
        return _verify
    
    async def verify_rate_limit(
        self,
        key: APIKeyMetadata = Depends(lambda self: self.verify_api_key)
    ) -> APIKeyMetadata:
        """
        Dependency: Verify API key and check rate limit
        
        Usage in FastAPI:
            @app.post("/invoke")
            async def invoke(key: APIKeyMetadata = Depends(deps.verify_rate_limit)):
                return {"status": "ok"}
        """
        # Check rate limit
        allowed, msg = self.tracker.check_rate_limit(key.key_id)
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=msg,
                headers={"Retry-After": "3600"}
            )
        
        return key
    
    async def optional_api_key(
        self,
        request: Request,
        x_api_key: Optional[str] = Header(None)
    ) -> Optional[APIKeyMetadata]:
        """
        Dependency: Optional API key (for public endpoints that track usage)
        
        Usage in FastAPI:
            @app.get("/health")
            async def health(key: Optional[APIKeyMetadata] = Depends(deps.optional_api_key)):
                return {"status": "ok"}
        """
        if not x_api_key:
            return None
        
        metadata = self.key_store.validate_key(x_api_key)
        return metadata


# ============================================================================
# MIDDLEWARE FOR REQUEST TRACKING
# ============================================================================

class APIKeyMiddleware:
    """Middleware for API key request/response tracking"""
    
    def __init__(
        self,
        app: FastAPI,
        key_store: APIKeyStore,
        tracker: UsageTracker
    ):
        self.app = app
        self.key_store = key_store
        self.tracker = tracker
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request = Request(scope, receive)
        start_time = time.time()
        
        # Extract API key from header
        api_key = request.headers.get("x-api-key")
        metadata = None
        
        if api_key:
            metadata = self.key_store.validate_key(api_key)
        
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "")
        
        # Store in scope for route handler access
        scope["api_key_metadata"] = metadata
        
        # Response tracking
        response_status = 200
        
        async def send_with_tracking(message):
            nonlocal response_status
            
            if message["type"] == "http.response.start":
                response_status = message["status"]
            
            # Track usage when response is sent
            if message["type"] == "http.response.body" and metadata:
                duration_ms = (time.time() - start_time) * 1000
                
                self.tracker.key_store.record_usage(
                    key_id=metadata.key_id,
                    endpoint=request.url.path,
                    method=request.method,
                    response_code=response_status,
                    duration_ms=duration_ms,
                    ip_address=client_ip,
                    user_agent=user_agent
                )
                
                logger.info(
                    f"API call tracked: {metadata.key_id} "
                    f"{request.method} {request.url.path} "
                    f"{response_status} {duration_ms:.1f}ms"
                )
            
            await send(message)
        
        await self.app(scope, receive, send_with_tracking)


# ============================================================================
# RESPONSE HEADERS HELPER
# ============================================================================

class RateLimitHeaders:
    """Helper to add rate limit headers to responses"""
    
    @staticmethod
    def get_headers(metadata: APIKeyMetadata) -> Dict[str, str]:
        """Get rate limit headers for response"""
        remaining = max(0, metadata.requests_limit - metadata.requests_this_month)
        
        return {
            "X-RateLimit-Limit": str(metadata.requests_limit),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(metadata.requests_limit),
            "X-RateLimit-Tier": metadata.tier.value,
        }


# ============================================================================
# ROUTE WRAPPER FOR EASY INTEGRATION
# ============================================================================

def require_api_key(
    required_mode: Optional[KeyMode] = None,
    track_usage: bool = True
):
    """
    Decorator for route handlers that require API key
    
    Usage:
        @app.post("/invoke")
        @require_api_key(required_mode=KeyMode.INVOKE_ONLY)
        async def invoke(request: Request):
            key = request.scope["api_key_metadata"]
            return {"key_id": key.key_id}
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            # API key should be validated by middleware
            metadata = request.scope.get("api_key_metadata")
            
            if not metadata:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Missing or invalid API key"
                )
            
            # Check mode
            if required_mode and required_mode != KeyMode.FULL_ACCESS:
                if metadata.mode != required_mode and metadata.mode != KeyMode.FULL_ACCESS:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"This endpoint requires {required_mode} mode"
                    )
            
            # Call route handler
            response = await func(request, *args, **kwargs)
            
            # Add rate limit headers
            if isinstance(response, dict):
                # Convert to Response if needed to add headers
                pass
            
            return response
        
        return wrapper
    
    return decorator


# ============================================================================
# EXAMPLE FASTAPI APP WITH AUTHENTICATION
# ============================================================================

def create_authenticated_app(
    key_store: APIKeyStore,
    tracker: UsageTracker
) -> FastAPI:
    """Create FastAPI app with API key authentication"""
    
    app = FastAPI(title="Orion Gateway - Authenticated")
    
    # Initialize dependencies
    deps = APIKeyDependencies(key_store, tracker)
    
    # Add middleware for request tracking
    app.add_middleware(
        APIKeyMiddleware,
        key_store=key_store,
        tracker=tracker
    )
    
    # ========================================================================
    # PUBLIC ENDPOINTS (No auth required)
    # ========================================================================
    
    @app.get("/")
    async def root():
        """API info"""
        return {
            "name": "Orion Gateway",
            "version": "2.0.0",
            "auth": "x-api-key header required for most endpoints"
        }
    
    # ========================================================================
    # PROTECTED ENDPOINTS (Auth required)
    # ========================================================================
    
    @app.get("/health")
    async def health(
        key: Optional[APIKeyMetadata] = Depends(deps.optional_api_key)
    ):
        """Health check (optional API key for tracking)"""
        return {
            "status": "healthy",
            "tracked": key is not None
        }
    
    @app.post("/invoke")
    async def invoke(
        request: Request,
        key: APIKeyMetadata = Depends(deps.verify_rate_limit)
    ):
        """
        Main inference endpoint
        Requires: x-api-key header with INVOKE_ONLY or FULL_ACCESS mode
        """
        body = await request.json()
        
        return {
            "request_id": "req_123",
            "status": "success",
            "result": {
                "answer": "Result would go here"
            },
            "usage": {
                "tier": key.tier.value,
                "requests_used": key.requests_this_month,
                "requests_limit": key.requests_limit
            }
        }
    
    @app.get("/usage")
    async def usage_endpoint(
        key: APIKeyMetadata = Depends(deps.verify_api_key)
    ):
        """
        Get current usage for API key
        Requires: x-api-key header with READ_ONLY or FULL_ACCESS mode
        """
        report = tracker.get_usage_report(key.key_id)
        
        return {
            "status": "success",
            "usage": report
        }
    
    # ========================================================================
    # ADMIN ENDPOINTS (Full access only)
    # ========================================================================
    
    @app.post("/admin/keys/create")
    async def admin_create_key(
        request: Request,
        key: APIKeyMetadata = Depends(deps.verify_api_key)
    ):
        """
        Admin: Create new API key
        Requires: x-api-key header with FULL_ACCESS mode
        """
        # Check if key has FULL_ACCESS
        if key.mode != KeyMode.FULL_ACCESS:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only FULL_ACCESS keys can create new keys"
            )
        
        body = await request.json()
        
        # Create new key
        new_key, metadata = key_store.create_key(
            name=body.get("name", "Unnamed Key"),
            tier=body.get("tier", "starter"),
            mode=body.get("mode", "invoke_only"),
            created_by=key.key_id,
            environment=body.get("environment", "production")
        )
        
        return {
            "status": "success",
            "key": new_key,  # Only returned once!
            "key_id": metadata.key_id,
            "metadata": metadata.to_dict()
        }
    
    @app.get("/admin/keys/list")
    async def admin_list_keys(
        key: APIKeyMetadata = Depends(deps.verify_api_key)
    ):
        """
        Admin: List all API keys
        Requires: x-api-key header with FULL_ACCESS mode
        """
        if key.mode != KeyMode.FULL_ACCESS:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        
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
        Admin: Revoke an API key
        Requires: x-api-key header with FULL_ACCESS mode
        """
        if key.mode != KeyMode.FULL_ACCESS:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        
        success = key_store.revoke_key(key_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Key {key_id} not found"
            )
        
        return {
            "status": "success",
            "message": f"Key {key_id} revoked"
        }
    
    @app.get("/admin/stats")
    async def admin_stats(
        key: APIKeyMetadata = Depends(deps.verify_api_key)
    ):
        """
        Admin: View statistics
        Requires: x-api-key header with FULL_ACCESS mode
        """
        if key.mode != KeyMode.FULL_ACCESS:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        
        return {
            "status": "success",
            "stats": {
                "total_keys": len(key_store.keys),
                "active_keys": sum(1 for k in key_store.keys.values() if k.status.value == "active"),
                "total_requests": sum(len(v) for v in key_store.usage.values())
            }
        }
    
    return app


if __name__ == "__main__":
    import uvicorn
    from api_key_manager import init_api_key_system, KeyTier, KeyMode
    
    # Initialize
    key_store, tracker = init_api_key_system()
    
    # Create test keys
    test_key, metadata = key_store.create_key(
        name="Test Invoke Key",
        tier=KeyTier.PRO,
        mode=KeyMode.INVOKE_ONLY,
        created_by="system"
    )
    print(f"Test invoke key: {test_key}")
    
    admin_key, _ = key_store.create_key(
        name="Admin Key",
        tier=KeyTier.ENTERPRISE,
        mode=KeyMode.FULL_ACCESS,
        created_by="system"
    )
    print(f"Admin key: {admin_key}")
    
    # Create app
    app = create_authenticated_app(key_store, tracker)
    
    print("\nStarting authenticated Orion Gateway on http://127.0.0.1:8001")
    print("Test with:")
    print(f'  curl -H "x-api-key: {test_key}" http://127.0.0.1:8001/invoke')
    
    uvicorn.run(app, host="127.0.0.1", port=8001)
