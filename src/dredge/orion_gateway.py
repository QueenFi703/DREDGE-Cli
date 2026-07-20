"""
ORION GATEWAY API — Production-grade reasoning + routing layer

A FastAPI-based gateway that:
  • Authenticates requests via API keys
  • Manages tier-based quotas (free/pro/enterprise)
  • Routes to DREDGE reasoning engines
  • Logs usage for billing
  • Supports multiple inference modes (standard, deep, transform, analyze)

Stack: FastAPI + Pydantic + SQLAlchemy (PostgreSQL/SQLite)
"""
import os
import httpx
import time
import uuid
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# MODELS & SCHEMAS
# =============================================================================


class InvokeRequest(BaseModel):
    """User request to invoke a reasoning operation."""
    input: str = Field(..., min_length=1, max_length=10000)
    mode: str = Field("standard", pattern="^(standard|deep|transform|analyze)$")
    context: Optional[Dict[str, Any]] = None


class InvokeResponse(BaseModel):
    """Response from a reasoning invocation."""
    request_id: str
    timestamp: float
    status: str
    mode: str
    result: Dict[str, Any]
    usage: Dict[str, int]
    tier: str


class UsageResponse(BaseModel):
    """Current usage statistics for an API key."""
    api_key: str
    tier: str
    requests_this_month: int
    requests_limit: int
    remaining: int
    usage_percent: float
    reset_date: str


# =============================================================================
# IN-MEMORY STATE (Replace with PostgreSQL for production)
# =============================================================================

# API Key → Organization mapping
ORGANIZATIONS = {
    "demo-free-key": {
        "id": "org-free-001",
        "tier": "free",
        "name": "Demo Free",
        "requests_limit": 100,
    },
    "demo-pro-key": {
        "id": "org-pro-001",
        "tier": "pro",
        "name": "Demo Pro",
        "requests_limit": 10000,
    },
    "demo-enterprise-key": {
        "id": "org-enterprise-001",
        "tier": "enterprise",
        "name": "Demo Enterprise",
        "requests_limit": 1000000,
    },
}

# Track requests per organization per month
USAGE_TRACKER: Dict[str, int] = {}


# =============================================================================
# REASONING ENGINES (Mock implementations)
# =============================================================================


def route_to_reasoning_engine(payload: str, mode: str, context: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Route to appropriate reasoning engine based on mode.

    In production, this routes to:
    - Intent Shaper (Dolly) → "standard" mode
    - Deep Reasoning Engine (Quasimoto) → "deep" mode
    - Context Weave Engine (CWE) → "transform" mode
    - Analysis Engine → "analyze" mode
    """

    if mode == "transform":
        return {
            "engine": "context_weave",
            "operation": "rewrite",
            "output": f"[TRANSFORMED]: {payload}",
            "confidence": 0.92,
        }

    if mode == "deep":
        return {
            "engine": "quasimoto_dre",
            "reasoning_depth": 4,
            "analysis": f"[DEEP REASONING]: Analyzed '{payload}' across 4 reasoning layers",
            "confidence": 0.87,
            "layers": [
                {"layer": 1, "insight": "Semantic decomposition"},
                {"layer": 2, "insight": "Intent extraction"},
                {"layer": 3, "insight": "Context weaving"},
                {"layer": 4, "insight": "Synthesis"},
            ],
        }

    if mode == "analyze":
        return {
            "engine": "analysis",
            "summary": f"[ANALYZED]: {payload}",
            "entities": ["entity1", "entity2"],
            "sentiment": "neutral",
            "score": 0.65,
        }

    # Standard mode (default)
    return {
        "engine": "intent_shaper",
        "response": f"[STANDARD]: {payload}",
        "processed": True,
    }


# =============================================================================
# API KEY VALIDATION & QUOTA MANAGEMENT
# =============================================================================


def verify_and_get_org(api_key: str) -> Dict[str, Any]:
    """Verify API key and return organization details."""
    if api_key not in ORGANIZATIONS:
        logger.warning("Invalid API key attempt")
        raise HTTPException(status_code=401, detail="Invalid API key")

    return ORGANIZATIONS[api_key]


def check_quota(org_id: str, tier: str, requests_limit: int) -> Dict[str, Any]:
    """Check if organization has remaining quota for this month."""

    # Calculate month key (YYYY-MM)
    now = datetime.utcnow()
    month_key = f"{org_id}:{now.strftime('%Y-%m')}"

    # Initialize if needed
    if month_key not in USAGE_TRACKER:
        USAGE_TRACKER[month_key] = 0

    current_usage = USAGE_TRACKER[month_key]
    remaining = requests_limit - current_usage
    usage_percent = (current_usage / requests_limit * 100) if requests_limit > 0 else 0

    # Calculate next reset (1st of next month)
    if now.month == 12:
        reset_date = now.replace(year=now.year + 1, month=1, day=1)
    else:
        reset_date = now.replace(month=now.month + 1, day=1)

    return {
        "current_usage": current_usage,
        "requests_limit": requests_limit,
        "remaining": max(0, remaining),
        "usage_percent": usage_percent,
        "reset_date": reset_date.isoformat(),
        "month_key": month_key,
    }


def increment_usage(month_key: str) -> None:
    """Increment usage counter for the month."""
    if month_key not in USAGE_TRACKER:
        USAGE_TRACKER[month_key] = 0
    USAGE_TRACKER[month_key] += 1


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="Orion Gateway API",
    description="Production reasoning + routing layer for DREDGE",
    version="1.0.0",
)

# -------------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------------

MCP_URL =os.getenv(
    "MCP_URL",
    "HTTP://127.0.0.1:3002"
)

# --------------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "dredge-orion-gateway",
        "timestamp": time.time(),
    }

# --------------------------------------------------------------------------------
# Quasimoto MCP Proxy
# Public Endpoint
#
# https://dredgeoriongateway.com/mcp
#
# --------------------------------------------------------------------------------
@app.get ("/mcp")
async def mcp_info():

    try:
        async with httpx.AsyncClient() as client:

             reponse = await client.get(
                f"{MCP_URL}/"
             )

             response.raise_for_status()

             return response.json()
    except Exception as  e:

        raise HTTPException(
            status_code=503,
            detail=f"MCP service unavailable: {str(e)}"
        )


class MCPRequest (BaseModel):
    method: str
    params: dict | None = None


@app.post("/mcp")
async def mcp_request(
    payload: MCPRequest
):

    try:

       async with httpx.AsyncClient() as client

           response = await client.post(
               f"{MCP_URL}/mcp",
               json=paypload.model_dump()
           )

           response.raise_for_status()

           return response.json()


    except Exception as e:

        raise HTTPException(
            status_code=503,
            detail=f"MCP request failed: {str(e)}"
        )



@app.post("/invoke", response_model=InvokeResponse)
def invoke(
    request: InvokeRequest,
    x_api_key: str = Header(..., description="API key for authentication"),
) -> InvokeResponse:
    """
    Main inference endpoint.

    Modes:
    - standard: Fast intent shaping (Dolly)
    - deep: Multi-layer reasoning (Quasimoto DRE)
    - transform: Context rewriting (CWE)
    - analyze: Detailed analysis
    """

    # Step 1: Authenticate
    org = verify_and_get_org(x_api_key)
    org_id = org["id"]
    tier = org["tier"]

    # Step 2: Check quota
    now = datetime.utcnow()
    month_key = f"{org_id}:{now.strftime('%Y-%m')}"
    quota = check_quota(org_id, tier, org["requests_limit"])

    if quota["remaining"] <= 0 and tier != "enterprise":
        logger.warning("Quota exceeded for %s", org_id)
        raise HTTPException(status_code=429, detail="Monthly quota exceeded")

    # Step 3: Process request
    result = route_to_reasoning_engine(request.input, request.mode, request.context)

    # Step 4: Track usage
    increment_usage(month_key)

    # Step 5: Return response
    return InvokeResponse(
        request_id=str(uuid.uuid4()),
        timestamp=time.time(),
        status="success",
        mode=request.mode,
        result=result,
        usage={
            "tokens_consumed": len(request.input.split()),
            "mode_cost": {"standard": 1, "deep": 5, "transform": 3, "analyze": 4}[request.mode],
        },
        tier=tier,
    )


@app.get("/usage", response_model=UsageResponse)
def get_usage(x_api_key: str = Header(...)) -> UsageResponse:
    """Get current usage statistics for an API key."""

    org = verify_and_get_org(x_api_key)
    org_id = org["id"]

    now = datetime.utcnow()
    month_key = f"{org_id}:{now.strftime('%Y-%m')}"
    _ = month_key
    quota = check_quota(org_id, org["tier"], org["requests_limit"])

    return UsageResponse(
        api_key=x_api_key[:20] + "***",  # Redact for security
        tier=org["tier"],
        requests_this_month=quota["current_usage"],
        requests_limit=quota["requests_limit"],
        remaining=quota["remaining"],
        usage_percent=quota["usage_percent"],
        reset_date=quota["reset_date"],
    )


@app.get("/admin/stats")
def admin_stats(x_api_key: str = Header(...)) -> Dict[str, Any]:
    """Admin statistics endpoint (requires valid API key)."""

    _ = verify_and_get_org(x_api_key)

    # Aggregate stats
    total_requests = sum(USAGE_TRACKER.values())
    num_orgs = len(ORGANIZATIONS)

    return {
        "total_requests_all_time": total_requests,
        "total_organizations": num_orgs,
        "usage_by_month": {k: v for k, v in USAGE_TRACKER.items()},
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Orion Gateway API starting...")
    logger.info("   Demo API keys available: %d", len(ORGANIZATIONS))


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 Orion Gateway API shutting down...")


def run_orion(host: str = "0.0.0.0", port: int = 8001, debug: bool = False):
    """Run the Orion Gateway server."""
    import uvicorn

    logger.info("Starting Orion Gateway on %s:%d", host, port)
    uvicorn.run(
        "dredge.orion_gateway:app",
        host="0.0.0.0",
        port=int (os.getenv("PORT", 8080)),
        log_level="info" if debug else "warning",
    )


if __name__ == "__main__":
    run_orion(debug=True)
