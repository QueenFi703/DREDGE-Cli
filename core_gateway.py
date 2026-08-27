"""
DREDGE CORE GATEWAY - Unified ASGI Application Spine

Architecture:
  The Core Gateway is a single FastAPI application that serves as the
  execution spine for the entire DREDGE platform. All functionality is
  provided through a modular adapter system.

Adapters:
  - Studio Adapter: Web UI and dashboard
  - Auth Adapter: API key management and authentication
  - Health Adapter: System health and monitoring
  - Admin Adapter: Administrative operations
  - Gordon Adapter: Gordon AI Agent integration (NEW)

Mounting:
  Each adapter is independently mounted on the core gateway via FastAPI
  include_router() mechanism, allowing for clean separation of concerns.

Benefits:
  1. Single entry point for all services
  2. Unified authentication/middleware
  3. Modular and extensible
  4. Easy to enable/disable adapters
  5. Clear dependency injection
  6. Better testing (each adapter isolated)
  7. Safe mounting with graceful degradation
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CORE GATEWAY APPLICATION
# ============================================================================

app = FastAPI(
    title="DREDGE Core Gateway",
    description="Unified ASGI application spine with modular adapters",
    version="2.0.0",
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
# ADAPTER REGISTRATION SYSTEM
# ============================================================================

class AdapterRegistry:
    """Registry for managing DREDGE adapters"""
    
    def __init__(self):
        self.adapters: Dict[str, Dict[str, Any]] = {}
    
    def register(self, name: str, enabled: bool = True, description: str = ""):
        """Register an adapter"""
        self.adapters[name] = {
            "name": name,
            "enabled": enabled,
            "description": description
        }
        logger.info(f"[Adapter] Registered: {name} ({'enabled' if enabled else 'disabled'})")
    
    def list_adapters(self) -> Dict[str, Dict[str, Any]]:
        """List all registered adapters"""
        return self.adapters
    
    def is_enabled(self, name: str) -> bool:
        """Check if adapter is enabled"""
        return self.adapters.get(name, {}).get("enabled", False)


# Create adapter registry
adapter_registry = AdapterRegistry()

# ============================================================================
# CORE ROUTES
# ============================================================================

@app.get("/", tags=["Core"])
async def root() -> Dict[str, Any]:
    """Core gateway entry point"""
    return {
        "service": "DREDGE Core Gateway",
        "version": "2.0.0",
        "architecture": "Unified ASGI Spine with Modular Adapters",
        "status": "operational",
        "adapters": adapter_registry.list_adapters(),
        "documentation": "/swagger",
        "health": "/health"
    }


@app.get("/health", tags=["Core"])
async def health() -> Dict[str, str]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "dredge-core-gateway",
        "version": "2.0.0"
    }


@app.get("/adapters", tags=["Core"])
async def list_adapters() -> Dict[str, Any]:
    """List all registered adapters"""
    return {
        "status": "success",
        "adapters": adapter_registry.list_adapters()
    }


@app.get("/status", tags=["Core"])
async def gateway_status() -> Dict[str, Any]:
    """Get gateway status and adapter information"""
    enabled = [name for name, adapter in adapter_registry.list_adapters().items() 
               if adapter["enabled"]]
    disabled = [name for name, adapter in adapter_registry.list_adapters().items() 
                if not adapter["enabled"]]
    
    return {
        "status": "operational",
        "gateway": "DREDGE Core Gateway",
        "version": "2.0.0",
        "adapters": {
            "enabled": enabled,
            "disabled": disabled,
            "total": len(adapter_registry.list_adapters())
        }
    }


# ============================================================================
# ADAPTER 1: HEALTH ADAPTER
# ============================================================================

def create_health_adapter():
    """Create health monitoring adapter"""
    from fastapi import APIRouter
    
    router = APIRouter(prefix="/health", tags=["Health"])
    
    @router.get("/detailed")
    async def health_detailed() -> Dict[str, Any]:
        """Detailed health information"""
        return {
            "status": "healthy",
            "service": "dredge-core-gateway",
            "version": "2.0.0",
            "timestamp": __import__("time").time(),
            "adapters": {
                name: adapter["enabled"]
                for name, adapter in adapter_registry.list_adapters().items()
            }
        }
    
    @router.get("/readiness")
    async def readiness() -> Dict[str, str]:
        """Kubernetes readiness probe"""
        return {"status": "ready"}
    
    @router.get("/liveness")
    async def liveness() -> Dict[str, str]:
        """Kubernetes liveness probe"""
        return {"status": "alive"}
    
    return router


# ============================================================================
# ADAPTER 2: STUDIO ADAPTER (DREDGE Studio UI)
# ============================================================================

def create_studio_adapter():
    """Create DREDGE Studio UI adapter"""
    from fastapi import APIRouter, HTTPException
    from fastapi.responses import FileResponse
    from pathlib import Path
    
    router = APIRouter(prefix="/studio", tags=["Studio"])
    
    @router.get("/dashboard")
    async def dashboard():
        """DREDGE Studio dashboard"""
        static_dir = Path(__file__).parent / 'dredge-cli-repo' / 'src' / 'dredge' / 'static'
        html_file = static_dir / 'dashboard_combined.html'
        
        if html_file.exists():
            return FileResponse(str(html_file), media_type='text/html')
        
        raise HTTPException(status_code=404, detail="Dashboard not found")
    
    @router.get("/advanced")
    async def advanced():
        """Advanced features dashboard"""
        static_dir = Path(__file__).parent / 'dredge-cli-repo' / 'src' / 'dredge' / 'static'
        html_file = static_dir / 'advanced_dashboard_new.html'
        
        if html_file.exists():
            return FileResponse(str(html_file), media_type='text/html')
        
        raise HTTPException(status_code=404, detail="Advanced dashboard not found")
    
    @router.get("/status")
    async def studio_status() -> Dict[str, Any]:
        """Studio status and capabilities"""
        return {
            "status": "operational",
            "adapter": "studio",
            "features": [
                "Dashboard",
                "Advanced Features",
                "Insight Lifting",
                "Model Management",
                "Pipeline Visualization"
            ]
        }
    
    return router


# ============================================================================
# ADAPTER 3: AUTH ADAPTER (API Key Management)
# ============================================================================

def create_auth_adapter():
    """Create authentication and API key adapter"""
    from fastapi import APIRouter, Header, HTTPException
    from pydantic import BaseModel
    
    router = APIRouter(prefix="/auth", tags=["Auth"])
    
    class APIKeyRequest(BaseModel):
        name: str
        tier: str = "starter"
        mode: str = "invoke_only"
    
    @router.get("/keys")
    async def list_keys(x_api_key: str = Header(None)) -> Dict[str, Any]:
        """List API keys"""
        if not x_api_key:
            raise HTTPException(status_code=401, detail="API key required")
        
        return {
            "status": "success",
            "keys": []
        }
    
    @router.post("/keys/create")
    async def create_key(request: APIKeyRequest, x_api_key: str = Header(None)) -> Dict[str, Any]:
        """Create new API key"""
        if not x_api_key:
            raise HTTPException(status_code=401, detail="Admin key required")
        
        return {
            "status": "success",
            "key": "dg_" + "x" * 40,
            "message": "API key created"
        }
    
    @router.get("/status")
    async def auth_status() -> Dict[str, Any]:
        """Auth adapter status"""
        return {
            "status": "operational",
            "adapter": "auth",
            "features": [
                "API Key Management",
                "Rate Limiting",
                "Usage Tracking",
                "Access Control"
            ]
        }
    
    return router


# ============================================================================
# ADAPTER 4: ADMIN ADAPTER (Administrative Operations)
# ============================================================================

def create_admin_adapter():
    """Create admin operations adapter"""
    from fastapi import APIRouter, Header, HTTPException
    
    router = APIRouter(prefix="/admin", tags=["Admin"])
    
    @router.get("/info")
    async def admin_info(x_api_key: str = Header(None)) -> Dict[str, Any]:
        """Admin information"""
        if not x_api_key:
            raise HTTPException(status_code=401, detail="Admin key required")
        
        return {
            "status": "success",
            "admin": "true",
            "adapters": adapter_registry.list_adapters(),
            "gateway": "DREDGE Core Gateway v2.0.0"
        }
    
    @router.get("/config")
    async def admin_config(x_api_key: str = Header(None)) -> Dict[str, Any]:
        """Gateway configuration"""
        if not x_api_key:
            raise HTTPException(status_code=401, detail="Admin key required")
        
        return {
            "status": "success",
            "gateway": "DREDGE Core Gateway",
            "architecture": "Unified ASGI Spine with Modular Adapters",
            "adapters": adapter_registry.list_adapters()
        }
    
    return router


# ============================================================================
# ADAPTER MOUNTING (Safe with error handling)
# ============================================================================

def mount_adapters():
    """Mount all adapters to the core gateway with safe error handling"""
    
    adapters_to_mount = [
        ("Studio", create_studio_adapter, "DREDGE Studio Web UI"),
        ("Auth", create_auth_adapter, "API Key Management & Authentication"),
        ("Health", create_health_adapter, "System Health & Monitoring"),
        ("Admin", create_admin_adapter, "Administrative Operations"),
    ]
    
    # Try to import and mount Gordon adapter (new, optional)
    try:
        from gordon_adapter import create_gordon_adapter
        adapters_to_mount.append(
            ("Gordon", create_gordon_adapter, "Gordon AI Agent Integration")
        )
    except Exception as e:
        logger.warning(f"[Mount] Gordon adapter not available: {e}")
    
    # Mount each adapter safely
    for adapter_name, adapter_creator, description in adapters_to_mount:
        try:
            router = adapter_creator()
            app.include_router(router)
            adapter_registry.register(adapter_name, enabled=True, description=description)
            logger.info(f"[Mount] {adapter_name} adapter mounted successfully")
        except Exception as e:
            logger.warning(f"[Mount] {adapter_name} adapter failed (non-critical): {e}")
            adapter_registry.register(adapter_name, enabled=False, 
                                     description=f"{description} (failed)")


# ============================================================================
# APPLICATION STARTUP
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize gateway on startup"""
    print("\n" + "=" * 80)
    print("  DREDGE CORE GATEWAY - ASGI Application Spine")
    print("=" * 80)
    print()
    print("Architecture: Unified ASGI Spine with Modular Adapters")
    print()
    print("Mounting adapters...")
    print()
    
    mount_adapters()
    
    print()
    print("Gateway Status:")
    print(f"  - Service: DREDGE Core Gateway")
    print(f"  - Version: 2.0.0")
    print(f"  - Framework: FastAPI + Uvicorn")
    print(f"  - Adapters: {len([a for a in adapter_registry.list_adapters().values() if a['enabled']])} enabled")
    print()
    print("Access Points:")
    print("  - Root:      http://127.0.0.1:8000/")
    print("  - Health:    http://127.0.0.1:8000/health")
    print("  - Swagger:   http://127.0.0.1:8000/swagger")
    print("  - ReDoc:     http://127.0.0.1:8000/redoc")
    print("  - Status:    http://127.0.0.1:8000/status")
    print("  - Adapters:  http://127.0.0.1:8000/adapters")
    print("  - Gordon:    http://127.0.0.1:8000/gordon/capabilities")
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
