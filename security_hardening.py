"""
DREDGE Gateway Security Hardening Module (FIXED - Graceful slowapi fallback)

Implements comprehensive security fixes for dredgeoriongateway.com
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, Callable
import time
import secrets

from fastapi import FastAPI, Request, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# Try to import slowapi, but continue without if not available
try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    HAS_SLOWAPI = True
except ImportError:
    HAS_SLOWAPI = False
    logger.warning("[Security] slowapi not available - rate limiting disabled")

# ============================================================================
# SECURITY CONFIGURATION
# ============================================================================

ALLOWED_ORIGINS = [
    "https://dredgeoriongateway.com",
    "https://www.dredgeoriongateway.com",
    "https://app.dredgeoriongateway.com",
    "http://localhost:3000",
    "http://localhost:8000",
]

API_KEY_PREFIX = "drg_"
API_KEY_LENGTH = 32
MAX_REQUEST_SIZE = 1_000_000

RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_WINDOW = 60
REQUEST_TIMEOUT = 30

# ============================================================================
# RATE LIMITER (Graceful Fallback)
# ============================================================================

class DummyLimiter:
    """Dummy limiter for when slowapi is not available"""
    pass

if HAS_SLOWAPI:
    limiter = Limiter(key_func=get_remote_address)
else:
    limiter = DummyLimiter()


# ============================================================================
# SECURITY HEADERS MIDDLEWARE
# ============================================================================

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Any:
        response = await call_next(request)
        
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' data:; "
            "connect-src 'self' https:; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'"
        )
        response.headers["Server"] = "DREDGE"
        
        return response


# ============================================================================
# REQUEST VALIDATION MIDDLEWARE
# ============================================================================

class RequestValidationMiddleware(BaseHTTPMiddleware):
    """Validate and sanitize incoming requests"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Any:
        if request.headers.get("content-length"):
            try:
                content_length = int(request.headers["content-length"])
                if content_length > MAX_REQUEST_SIZE:
                    return JSONResponse(
                        status_code=413,
                        content={"error": "Request payload too large"}
                    )
            except ValueError:
                pass
        
        method = request.method
        path = request.url.path
        client_ip = request.client.host if request.client else "unknown"
        
        logger.info(f"[Request] {method} {path} from {client_ip}")
        
        try:
            response = await call_next(request)
        except Exception as e:
            logger.error(f"[Request Error] {method} {path}: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error"}
            )
        
        return response


# ============================================================================
# CORS RESTRICTIONS
# ============================================================================

def setup_cors_security(app: FastAPI):
    """Setup CORS with restricted origins"""
    from fastapi.middleware.cors import CORSMiddleware
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Accept", "Accept-Language", "Content-Language", "Content-Type", "Authorization", "X-API-Key"],
        max_age=3600,
        expose_headers=["X-Total-Count", "X-Page-Count"],
    )


# ============================================================================
# API KEY MANAGEMENT
# ============================================================================

class APIKeyManager:
    """Manage API keys securely"""
    
    def __init__(self):
        self.keys: Dict[str, Dict[str, Any]] = {}
        self.rate_limits: Dict[str, Dict[str, Any]] = {}
    
    def generate_key(self) -> str:
        """Generate secure API key"""
        random_bytes = secrets.token_hex(16)
        return f"{API_KEY_PREFIX}{random_bytes}"
    
    def validate_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key"""
        if not api_key or not api_key.startswith(API_KEY_PREFIX):
            return None
        
        key_data = self.keys.get(api_key)
        if not key_data:
            return None
        
        if not key_data.get("enabled", True):
            return None
        
        if "expires_at" in key_data:
            if datetime.utcnow() > datetime.fromisoformat(key_data["expires_at"]):
                return None
        
        return key_data
    
    def check_rate_limit(self, api_key: str) -> bool:
        """Check rate limit for API key"""
        if api_key not in self.rate_limits:
            self.rate_limits[api_key] = {
                "requests": 0,
                "window_start": time.time(),
                "limit": 1000
            }
        
        limit_data = self.rate_limits[api_key]
        window_elapsed = time.time() - limit_data["window_start"]
        
        if window_elapsed >= 3600:
            limit_data["requests"] = 0
            limit_data["window_start"] = time.time()
            window_elapsed = 0
        
        limit_data["requests"] += 1
        
        return limit_data["requests"] <= limit_data["limit"]


api_key_manager = APIKeyManager()


# ============================================================================
# SETUP SECURITY
# ============================================================================

def setup_security(app: FastAPI):
    """Setup all security measures on app"""
    
    logger.info("[Security] Initializing security hardening...")
    
    app.add_middleware(SecurityHeadersMiddleware)
    logger.info("[Security] Security headers middleware added")
    
    app.add_middleware(RequestValidationMiddleware)
    logger.info("[Security] Request validation middleware added")
    
    setup_cors_security(app)
    logger.info("[Security] CORS restrictions applied")
    
    if HAS_SLOWAPI:
        app.state.limiter = limiter
        logger.info("[Security] Rate limiting enabled")
    else:
        logger.warning("[Security] Rate limiting disabled (slowapi not available)")
    
    logger.info("[Security] All security measures initialized")


__all__ = [
    'setup_security',
    'APIKeyManager',
    'SecurityHeadersMiddleware',
    'RequestValidationMiddleware',
    'limiter',
    'api_key_manager',
]
