"""
DREDGE INTEGRATED GATEWAY - Three-Layer Cognitive Architecture

Complete System Integration:
  Layer 1: GPT Sol (Reasoning Engine)
  Layer 2: Tresh (Decision/Orchestration Layer)
  Layer 3: DREDGE (Application/Execution Layer)
  Hub: Cognitive Nervous System

Plus:
  - Comprehensive Security Hardening
  - Rate Limiting & CORS Restrictions
  - Telemetry & Monitoring
  - Full API Documentation
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
# IMPORT COGNITIVE LAYERS
# ============================================================================

try:
    from gpt_sol_reasoning_engine import gpt_sol_engine, setup_gpt_sol
    HAS_GPT_SOL = True
    logger.info("[Init] GPT Sol Reasoning Engine loaded")
except Exception as e:
    logger.warning(f"[Init] GPT Sol not available: {e}")
    HAS_GPT_SOL = False
    gpt_sol_engine = None

try:
    from tresh_decision_layer import tresh_engine, setup_tresh
    HAS_TRESH = True
    logger.info("[Init] Tresh Decision Layer loaded")
except Exception as e:
    logger.warning(f"[Init] Tresh not available: {e}")
    HAS_TRESH = False
    tresh_engine = None

try:
    from cognitive_nervous_system import setup_cognitive_nervous_system, cognitive_nervous_system
    HAS_NERVOUS_SYSTEM = True
    logger.info("[Init] Cognitive Nervous System loaded")
except Exception as e:
    logger.warning(f"[Init] Cognitive Nervous System not available: {e}")
    HAS_NERVOUS_SYSTEM = False

try:
    from security_hardening import setup_security
    HAS_SECURITY = True
    logger.info("[Init] Security Hardening loaded")
except Exception as e:
    logger.warning(f"[Init] Security Hardening not available: {e}")
    HAS_SECURITY = False

# ============================================================================
# CORE GATEWAY APPLICATION
# ============================================================================

app = FastAPI(
    title="DREDGE Integrated Gateway",
    description="Three-Layer Cognitive Architecture with Security Hardening",
    version="3.0.0",
    docs_url="/swagger",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add basic CORS (will be replaced by security hardening)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Vercel Web Analytics middleware
try:
    from vercel_analytics import VercelAnalyticsMiddleware
    app.add_middleware(VercelAnalyticsMiddleware)
    logger.info("[Analytics] Vercel Web Analytics middleware enabled")
except Exception as e:
    logger.warning(f"[Analytics] Vercel Analytics not available: {e}")

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


adapter_registry = AdapterRegistry()

# ============================================================================
# CORE ROUTES
# ============================================================================

@app.get("/", tags=["Core"])
async def root() -> Dict[str, Any]:
    """Core gateway entry point"""
    return {
        "service": "DREDGE Integrated Gateway",
        "version": "3.0.0",
        "architecture": "Three-Layer Cognitive with Security",
        "status": "operational",
        "layers": {
            "gpt_sol": "enabled" if HAS_GPT_SOL else "disabled",
            "tresh": "enabled" if HAS_TRESH else "disabled",
            "nervous_system": "enabled" if HAS_NERVOUS_SYSTEM else "disabled",
            "security": "enabled" if HAS_SECURITY else "disabled"
        },
        "adapters": adapter_registry.list_adapters(),
        "documentation": "/swagger"
    }


@app.get("/health", tags=["Core"])
async def health() -> Dict[str, str]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "dredge-integrated-gateway",
        "version": "3.0.0",
        "cognitive_layers": "operational" if (HAS_GPT_SOL and HAS_TRESH) else "limited"
    }


@app.get("/status", tags=["Core"])
async def gateway_status() -> Dict[str, Any]:
    """Get gateway status"""
    return {
        "status": "operational",
        "gateway": "DREDGE Integrated Gateway v3.0.0",
        "architecture": "Three-Layer Cognitive Architecture",
        "layers": {
            "gpt_sol": {
                "status": "operational" if HAS_GPT_SOL else "unavailable",
                "description": "Advanced reasoning engine"
            },
            "tresh": {
                "status": "operational" if HAS_TRESH else "unavailable",
                "description": "Strategic decision layer"
            },
            "dredge": {
                "status": "operational",
                "description": "Application execution layer"
            },
            "nervous_system": {
                "status": "operational" if HAS_NERVOUS_SYSTEM else "unavailable",
                "description": "Integration hub"
            }
        },
        "security": {
            "status": "enabled" if HAS_SECURITY else "limited",
            "hardening": "applied" if HAS_SECURITY else "pending"
        }
    }


@app.get("/architecture", tags=["Core"])
async def architecture() -> Dict[str, Any]:
    """Get architecture information"""
    return {
        "name": "DREDGE Integrated Gateway",
        "version": "3.0.0",
        "model": "Three-Layer Cognitive Architecture",
        "layers": [
            {
                "name": "GPT Sol",
                "position": 1,
                "role": "Advanced Reasoning",
                "capabilities": [
                    "Multi-modal reasoning",
                    "Ethical analysis",
                    "Strategic forecasting",
                    "Pattern recognition"
                ],
                "status": "operational" if HAS_GPT_SOL else "unavailable"
            },
            {
                "name": "Tresh",
                "position": 2,
                "role": "Decision & Orchestration",
                "capabilities": [
                    "Strategic decisions",
                    "Agent orchestration",
                    "Performance learning",
                    "Strategy adaptation"
                ],
                "status": "operational" if HAS_TRESH else "unavailable"
            },
            {
                "name": "DREDGE",
                "position": 3,
                "role": "Application Execution",
                "capabilities": [
                    "Plan execution",
                    "Resource management",
                    "Performance telemetry",
                    "Error handling"
                ],
                "status": "operational"
            }
        ],
        "integration": "Cognitive Nervous System" if HAS_NERVOUS_SYSTEM else "basic",
        "security": "Hardened" if HAS_SECURITY else "standard"
    }


# ============================================================================
# ADAPTER MOUNTING
# ============================================================================

def mount_adapters():
    """Mount all adapters and cognitive layers"""
    
    adapters_to_mount = []
    
    # Health adapter
    def create_health_adapter():
        from fastapi import APIRouter
        router = APIRouter(prefix="/health", tags=["Health"])
        
        @router.get("/detailed")
        async def health_detailed() -> Dict[str, Any]:
            return {
                "status": "healthy",
                "service": "dredge-integrated-gateway",
                "version": "3.0.0",
                "layers": {
                    "gpt_sol": "operational" if HAS_GPT_SOL else "unavailable",
                    "tresh": "operational" if HAS_TRESH else "unavailable",
                    "dredge": "operational",
                    "nervous_system": "operational" if HAS_NERVOUS_SYSTEM else "unavailable"
                }
            }
        
        @router.get("/readiness")
        async def readiness():
            return {"status": "ready"}
        
        @router.get("/liveness")
        async def liveness():
            return {"status": "alive"}
        
        return router
    
    adapters_to_mount.append(("Health", create_health_adapter, "Health Monitoring"))
    
    # Mount adapters
    for adapter_name, adapter_creator, description in adapters_to_mount:
        try:
            router = adapter_creator()
            app.include_router(router)
            adapter_registry.register(adapter_name, enabled=True, description=description)
            logger.info(f"[Mount] {adapter_name} adapter mounted")
        except Exception as e:
            logger.warning(f"[Mount] {adapter_name} failed: {e}")
            adapter_registry.register(adapter_name, enabled=False, description=description)
    
    # Mount cognitive layers
    if HAS_GPT_SOL:
        try:
            setup_gpt_sol(app)
            adapter_registry.register("GPT-Sol", enabled=True, description="Reasoning Engine")
            logger.info("[Mount] GPT Sol mounted")
        except Exception as e:
            logger.warning(f"[Mount] GPT Sol failed: {e}")
    
    if HAS_TRESH:
        try:
            setup_tresh(app, gpt_sol_engine)
            adapter_registry.register("Tresh", enabled=True, description="Decision Layer")
            logger.info("[Mount] Tresh mounted")
        except Exception as e:
            logger.warning(f"[Mount] Tresh failed: {e}")
    
    if HAS_NERVOUS_SYSTEM:
        try:
            setup_cognitive_nervous_system(app, gpt_sol_engine, tresh_engine)
            adapter_registry.register("Nervous-System", enabled=True, description="Integration Hub")
            logger.info("[Mount] Cognitive Nervous System mounted")
        except Exception as e:
            logger.warning(f"[Mount] Nervous System failed: {e}")
    
    # Try to mount Gordon adapter
    try:
        from gordon_adapter import create_gordon_adapter
        router = create_gordon_adapter()
        app.include_router(router)
        adapter_registry.register("Gordon", enabled=True, description="AI Agent Integration")
        logger.info("[Mount] Gordon adapter mounted")
    except Exception as e:
        logger.warning(f"[Mount] Gordon adapter not available: {e}")
    
    # Apply security hardening (LAST, after all routes are mounted)
    if HAS_SECURITY:
        try:
            setup_security(app)
            adapter_registry.register("Security", enabled=True, description="Security Hardening")
            logger.info("[Mount] Security hardening applied")
        except Exception as e:
            logger.warning(f"[Mount] Security hardening failed: {e}")


# ============================================================================
# STARTUP EVENT
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize gateway on startup"""
    print("\n" + "=" * 80)
    print("  DREDGE INTEGRATED GATEWAY")
    print("  Three-Layer Cognitive Architecture")
    print("=" * 80)
    print()
    print("Mounting adapters and cognitive layers...")
    print()
    
    mount_adapters()
    
    print()
    print("Gateway Status:")
    print(f"  ✓ GPT Sol Reasoning Engine: {'ENABLED' if HAS_GPT_SOL else 'DISABLED'}")
    print(f"  ✓ Tresh Decision Layer: {'ENABLED' if HAS_TRESH else 'DISABLED'}")
    print(f"  ✓ DREDGE Application Layer: ENABLED")
    print(f"  ✓ Cognitive Nervous System: {'ENABLED' if HAS_NERVOUS_SYSTEM else 'DISABLED'}")
    print(f"  ✓ Security Hardening: {'ENABLED' if HAS_SECURITY else 'DISABLED'}")
    print()
    print("Access Points:")
    print("  - Root:      http://127.0.0.1:8000/")
    print("  - Health:    http://127.0.0.1:8000/health")
    print("  - Status:    http://127.0.0.1:8000/status")
    print("  - Architecture: http://127.0.0.1:8000/architecture")
    print("  - Swagger:   http://127.0.0.1:8000/swagger")
    print("  - GPT Sol:   http://127.0.0.1:8000/gpt-sol/analyze (if enabled)")
    print("  - Tresh:     http://127.0.0.1:8000/tresh/decide (if enabled)")
    print("  - Nervous System: http://127.0.0.1:8000/nervous-system/request (if enabled)")
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
