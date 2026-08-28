"""
DREDGE INTEGRATED GATEWAY - Three-Layer Cognitive Architecture
Complete System Integration with Security Hardening (FIXED)
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from typing import Dict, Any

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
    from security_hardening import setup_security, setup_cors_security
    HAS_SECURITY = True
    logger.info("[Init] Security Hardening loaded")
except Exception as e:
    logger.warning(f"[Init] Security Hardening not available: {e}")
    HAS_SECURITY = False

# ============================================================================
# CREATE APP
# ============================================================================

app = FastAPI(
    title="DREDGE Integrated Gateway",
    description="Three-Layer Cognitive Architecture with Security Hardening",
    version="3.0.0",
    docs_url="/swagger",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# ============================================================================
# SETUP SECURITY FIRST (before routes mounted)
# ============================================================================

if HAS_SECURITY:
    try:
        setup_cors_security(app)
        logger.info("[Init] Security CORS setup applied")
    except Exception as e:
        logger.warning(f"[Init] Security CORS setup failed: {e}")
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info("[Init] Fallback CORS applied")

# Add Vercel Analytics
try:
    from vercel_analytics import VercelAnalyticsMiddleware
    app.add_middleware(VercelAnalyticsMiddleware)
    logger.info("[Analytics] Vercel Analytics enabled")
except Exception as e:
    logger.warning(f"[Analytics] Not available: {e}")

# ============================================================================
# CORE ROUTES
# ============================================================================

@app.get("/", tags=["Core"])
async def root() -> Dict[str, Any]:
    """Gateway entry point"""
    return {
        "service": "DREDGE Integrated Gateway",
        "version": "3.0.0",
        "status": "operational",
        "layers": {
            "gpt_sol": "enabled" if HAS_GPT_SOL else "disabled",
            "tresh": "enabled" if HAS_TRESH else "disabled",
            "nervous_system": "enabled" if HAS_NERVOUS_SYSTEM else "disabled",
            "security": "enabled" if HAS_SECURITY else "disabled"
        },
        "documentation": "/swagger"
    }

@app.get("/health", tags=["Core"])
async def health() -> Dict[str, str]:
    """Health check"""
    return {
        "status": "healthy",
        "service": "dredge-gateway",
        "version": "3.0.0"
    }

@app.get("/status", tags=["Core"])
async def gateway_status() -> Dict[str, Any]:
    """Gateway status"""
    return {
        "status": "operational",
        "layers": {
            "gpt_sol": "operational" if HAS_GPT_SOL else "unavailable",
            "tresh": "operational" if HAS_TRESH else "unavailable",
            "dredge": "operational",
            "nervous_system": "operational" if HAS_NERVOUS_SYSTEM else "unavailable"
        },
        "security": "enabled" if HAS_SECURITY else "limited"
    }

@app.get("/architecture", tags=["Core"])
async def architecture() -> Dict[str, Any]:
    """Architecture information"""
    return {
        "name": "DREDGE Integrated Gateway",
        "version": "3.0.0",
        "layers": [
            {
                "name": "GPT Sol",
                "role": "Reasoning",
                "status": "operational" if HAS_GPT_SOL else "unavailable"
            },
            {
                "name": "Tresh",
                "role": "Decision",
                "status": "operational" if HAS_TRESH else "unavailable"
            },
            {
                "name": "DREDGE",
                "role": "Execution",
                "status": "operational"
            }
        ]
    }

# ============================================================================
# MOUNT COGNITIVE LAYERS
# ============================================================================

def mount_layers():
    """Mount all cognitive layers"""
    
    if HAS_GPT_SOL:
        try:
            setup_gpt_sol(app)
            logger.info("[Mount] GPT Sol mounted")
        except Exception as e:
            logger.warning(f"[Mount] GPT Sol failed: {e}")
    
    if HAS_TRESH:
        try:
            setup_tresh(app, gpt_sol_engine)
            logger.info("[Mount] Tresh mounted")
        except Exception as e:
            logger.warning(f"[Mount] Tresh failed: {e}")
    
    if HAS_NERVOUS_SYSTEM:
        try:
            setup_cognitive_nervous_system(app, gpt_sol_engine, tresh_engine)
            logger.info("[Mount] Nervous System mounted")
        except Exception as e:
            logger.warning(f"[Mount] Nervous System failed: {e}")
    
    # Mount Gordon adapter
    try:
        from gordon_adapter import create_gordon_adapter
        router = create_gordon_adapter()
        app.include_router(router)
        logger.info("[Mount] Gordon adapter mounted")
    except Exception as e:
        logger.warning(f"[Mount] Gordon adapter not available: {e}")

# ============================================================================
# STARTUP EVENT (no unicode characters!)
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize gateway on startup"""
    logger.info("=" * 80)
    logger.info("DREDGE INTEGRATED GATEWAY - Startup")
    logger.info("=" * 80)
    
    logger.info("Mounting cognitive layers...")
    mount_layers()
    
    logger.info("")
    logger.info("Gateway Status:")
    logger.info(f"  - GPT Sol: {'ENABLED' if HAS_GPT_SOL else 'DISABLED'}")
    logger.info(f"  - Tresh: {'ENABLED' if HAS_TRESH else 'DISABLED'}")
    logger.info(f"  - DREDGE: ENABLED")
    logger.info(f"  - Nervous System: {'ENABLED' if HAS_NERVOUS_SYSTEM else 'DISABLED'}")
    logger.info(f"  - Security: {'ENABLED' if HAS_SECURITY else 'DISABLED'}")
    logger.info("")
    logger.info("Access Points:")
    logger.info("  - http://localhost:8000/ (root)")
    logger.info("  - http://localhost:8000/health (health check)")
    logger.info("  - http://localhost:8000/swagger (API docs)")
    logger.info("=" * 80)

# ============================================================================
# APPLY MIDDLEWARE SECURITY (after routes mounted)
# ============================================================================

@app.on_event("startup")
async def apply_security_middleware():
    """Apply security middleware after routes are loaded"""
    if HAS_SECURITY:
        try:
            from security_hardening import SecurityHeadersMiddleware, RequestValidationMiddleware
            app.add_middleware(SecurityHeadersMiddleware)
            app.add_middleware(RequestValidationMiddleware)
            logger.info("[Security] Middleware applied after startup")
        except Exception as e:
            logger.warning(f"[Security] Middleware setup failed: {e}")

# ============================================================================
# LOCAL EXECUTION
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True, log_level="info")
