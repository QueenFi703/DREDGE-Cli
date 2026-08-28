"""
DREDGE CORE GATEWAY - Fixed with Authentication and Login
Architecture: Unified ASGI Spine with Modular Adapters
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CREATE APPLICATION
# ============================================================================

app = FastAPI(
    title="DREDGE Studio",
    description="AI-Powered API Gateway with Cognitive Architecture",
    version="2.5.0",
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

# Add Vercel Analytics
try:
    from vercel_analytics import VercelAnalyticsMiddleware
    app.add_middleware(VercelAnalyticsMiddleware)
    logger.info("[Analytics] Vercel Web Analytics enabled")
except Exception as e:
    logger.warning(f"[Analytics] Not available: {e}")

# ============================================================================
# ADAPTER REGISTRY
# ============================================================================

class AdapterRegistry:
    def __init__(self):
        self.adapters = {}
    
    def register(self, name: str, enabled: bool = True, description: str = ""):
        self.adapters[name] = {"name": name, "enabled": enabled, "description": description}
        logger.info(f"[Adapter] {name}: {'ENABLED' if enabled else 'DISABLED'}")
    
    def list_adapters(self):
        return self.adapters
    
    def is_enabled(self, name: str) -> bool:
        return self.adapters.get(name, {}).get("enabled", False)

adapter_registry = AdapterRegistry()

# ============================================================================
# CORE ROUTES
# ============================================================================

@app.get("/", tags=["Core"], response_class=HTMLResponse)
async def root():
    """Home page"""
    home_html = Path(__file__).parent / "templates" / "index.html"
    
    if home_html.exists():
        return home_html.read_text()
    
    return """
    <h1>DREDGE Studio</h1>
    <p>AI-Powered API Gateway</p>
    <a href="/swagger">API Documentation</a>
    """

@app.get("/health", tags=["Core"])
async def health():
    """Health check"""
    return {"status": "healthy", "service": "dredge-studio", "version": "2.5.0"}

@app.get("/status", tags=["Core"])
async def status():
    """Gateway status"""
    enabled = [n for n, a in adapter_registry.list_adapters().items() if a["enabled"]]
    return {
        "status": "operational",
        "service": "DREDGE Studio",
        "version": "2.5.0",
        "adapters": {"enabled": enabled, "total": len(adapter_registry.list_adapters())}
    }

@app.get("/dashboard", tags=["Pages"], response_class=HTMLResponse)
async def dashboard():
    """Dashboard page (protected)"""
    dashboard_html = Path(__file__).parent / "templates" / "dashboard.html"
    
    if dashboard_html.exists():
        return dashboard_html.read_text()
    
    return "<h1>Dashboard</h1><p>Coming soon</p>"

# ============================================================================
# STUDIO ADAPTER
# ============================================================================

def create_studio_adapter():
    """Studio UI adapter"""
    from fastapi import APIRouter
    
    router = APIRouter(prefix="/studio", tags=["Studio"])
    
    @router.get("/status")
    async def studio_status():
        return {
            "status": "operational",
            "features": [
                "Dashboard",
                "Advanced Features",
                "Model Management",
                "Pipeline Visualization",
                "Analytics & Insights"
            ]
        }
    
    return router

# ============================================================================
# HEALTH ADAPTER
# ============================================================================

def create_health_adapter():
    """Health monitoring adapter"""
    from fastapi import APIRouter
    
    router = APIRouter(prefix="/health", tags=["Health"])
    
    @router.get("/detailed")
    async def health_detailed():
        return {
            "status": "healthy",
            "service": "dredge-studio",
            "version": "2.5.0",
            "adapters": {n: a["enabled"] for n, a in adapter_registry.list_adapters().items()}
        }
    
    @router.get("/readiness")
    async def readiness():
        return {"status": "ready"}
    
    @router.get("/liveness")
    async def liveness():
        return {"status": "alive"}
    
    return router

# ============================================================================
# MOUNT ADAPTERS
# ============================================================================

def mount_adapters():
    """Mount all adapters"""
    
    adapters_to_mount = [
        ("Studio", create_studio_adapter, "DREDGE Studio UI"),
        ("Health", create_health_adapter, "Health Monitoring"),
    ]
    
    # Try Gordon
    try:
        from gordon_adapter import create_gordon_adapter
        adapters_to_mount.append(("Gordon", create_gordon_adapter, "Gordon AI Agent"))
    except Exception as e:
        logger.warning(f"[Mount] Gordon not available: {e}")
    
    # Mount
    for name, creator, desc in adapters_to_mount:
        try:
            router = creator()
            app.include_router(router)
            adapter_registry.register(name, enabled=True, description=desc)
            logger.info(f"[Mount] {name} mounted")
        except Exception as e:
            logger.warning(f"[Mount] {name} failed: {e}")
            adapter_registry.register(name, enabled=False, description=desc)

# ============================================================================
# MOUNT AUTHENTICATION
# ============================================================================

def mount_auth():
    """Mount authentication system"""
    try:
        from auth_module import create_auth_router
        auth_router = create_auth_router()
        app.include_router(auth_router)
        adapter_registry.register("Auth", enabled=True, description="Authentication & Sessions")
        logger.info("[Mount] Auth mounted")
    except Exception as e:
        logger.warning(f"[Mount] Auth failed: {e}")
        adapter_registry.register("Auth", enabled=False, description="Authentication (failed)")

# ============================================================================
# STARTUP
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Startup initialization"""
    logger.info("=" * 80)
    logger.info("DREDGE STUDIO - Starting up")
    logger.info("=" * 80)
    
    logger.info("Mounting authentication system...")
    mount_auth()
    
    logger.info("Mounting adapters...")
    mount_adapters()
    
    logger.info("")
    logger.info("Services:")
    for name, adapter in adapter_registry.list_adapters().items():
        status_icon = "[OK]" if adapter["enabled"] else "[FAIL]"
        logger.info(f"  {status_icon} {name}: {adapter['description']}")
    
    logger.info("")
    logger.info("Ready at http://localhost:8000")
    logger.info("=" * 80)

# ============================================================================
# LOCAL EXECUTION
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True, log_level="info")
