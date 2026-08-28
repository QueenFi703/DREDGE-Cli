"""
ORION GATEWAY API — Production-grade reasoning + routing layer
"""
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InvokeRequest(BaseModel):
    input: str = Field(..., min_length=1, max_length=10000)
    mode: str = Field("standard", pattern="^(standard|deep|transform|analyze)$")
    context: Optional[Dict[str, Any]] = None


class InvokeResponse(BaseModel):
    request_id: str
    timestamp: float
    status: str
    mode: str
    result: Dict[str, Any]
    usage: Dict[str, int]
    tier: str


class UsageResponse(BaseModel):
    api_key: str
    tier: str
    requests_this_month: int
    requests_limit: int
    remaining: int
    usage_percent: float
    reset_date: str


ORGANIZATIONS = {
    "demo-free-key": {"id": "org-free-001", "tier": "free", "name": "Demo Free", "requests_limit": 100},
    "demo-pro-key": {"id": "org-pro-001", "tier": "pro", "name": "Demo Pro", "requests_limit": 10000},
    "demo-enterprise-key": {"id": "org-enterprise-001", "tier": "enterprise", "name": "Demo Enterprise", "requests_limit": 1000000},
}
USAGE_TRACKER: Dict[str, int] = {}


def route_to_reasoning_engine(payload: str, mode: str, context: Optional[Dict] = None) -> Dict[str, Any]:
    if mode == "transform":
        return {"engine": "context_weave", "operation": "rewrite", "output": f"[TRANSFORMED]: {payload}", "confidence": 0.92}
    if mode == "deep":
        return {"engine": "quasimoto_dre", "reasoning_depth": 4, "analysis": f"[DEEP REASONING]: Analyzed '{payload}' across 4 reasoning layers", "confidence": 0.87, "layers": [{"layer": 1, "insight": "Semantic decomposition"}, {"layer": 2, "insight": "Intent extraction"}, {"layer": 3, "insight": "Context weaving"}, {"layer": 4, "insight": "Synthesis"}]}
    if mode == "analyze":
        return {"engine": "analysis", "summary": f"[ANALYZED]: {payload}", "entities": ["entity1", "entity2"], "sentiment": "neutral", "score": 0.65}
    return {"engine": "intent_shaper", "response": f"[STANDARD]: {payload}", "processed": True}


def verify_and_get_org(api_key: str) -> Dict[str, Any]:
    if api_key not in ORGANIZATIONS:
        logger.warning("Invalid API key attempt")
        raise HTTPException(status_code=401, detail="Invalid API key")
    return ORGANIZATIONS[api_key]


def check_quota(org_id: str, tier: str, requests_limit: int) -> Dict[str, Any]:
    now = datetime.utcnow()
    month_key = f"{org_id}:{now.strftime('%Y-%m')}"
    current_usage = USAGE_TRACKER.setdefault(month_key, 0)
    remaining = requests_limit - current_usage
    reset_date = now.replace(year=now.year + 1, month=1, day=1) if now.month == 12 else now.replace(month=now.month + 1, day=1)
    return {"current_usage": current_usage, "requests_limit": requests_limit, "remaining": max(0, remaining), "usage_percent": (current_usage / requests_limit * 100) if requests_limit else 0, "reset_date": reset_date.isoformat(), "month_key": month_key}


def increment_usage(month_key: str) -> None:
    USAGE_TRACKER[month_key] = USAGE_TRACKER.get(month_key, 0) + 1


app = FastAPI(title="Orion Gateway API", description="Production reasoning + routing layer for DREDGE", version="1.0.0")
MCP_URL = os.getenv("MCP_URL", "http://127.0.0.1:8001")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "dredge-orion-gateway", "timestamp": time.time()}


class MCPRequest(BaseModel):
    method: str
    params: Optional[dict] = None


@app.get("/mcp")
async def mcp_info():
    return {"status": "ok", "message": "MCP endpoint is live. Use POST for requests."}


@app.post("/mcp")
async def mcp_request(payload: MCPRequest):
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(f"{MCP_URL}/mcp", json=payload.model_dump())
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail=f"MCP request failed: {exc}") from exc


@app.post("/invoke", response_model=InvokeResponse)
def invoke(request: InvokeRequest, x_api_key: str = Header(..., description="API key for authentication")) -> InvokeResponse:
    org = verify_and_get_org(x_api_key)
    quota = check_quota(org["id"], org["tier"], org["requests_limit"])
    if quota["remaining"] <= 0 and org["tier"] != "enterprise":
        raise HTTPException(status_code=429, detail="Monthly quota exceeded")
    result = route_to_reasoning_engine(request.input, request.mode, request.context)
    increment_usage(quota["month_key"])
    return InvokeResponse(request_id=str(uuid.uuid4()), timestamp=time.time(), status="success", mode=request.mode, result=result, usage={"tokens_consumed": len(request.input.split()), "mode_cost": {"standard": 1, "deep": 5, "transform": 3, "analyze": 4}[request.mode]}, tier=org["tier"])


@app.get("/usage", response_model=UsageResponse)
def get_usage(x_api_key: str = Header(...)) -> UsageResponse:
    org = verify_and_get_org(x_api_key)
    quota = check_quota(org["id"], org["tier"], org["requests_limit"])
    return UsageResponse(api_key=x_api_key[:20] + "***", tier=org["tier"], requests_this_month=quota["current_usage"], requests_limit=quota["requests_limit"], remaining=quota["remaining"], usage_percent=quota["usage_percent"], reset_date=quota["reset_date"])


@app.get("/admin/stats")
def admin_stats(x_api_key: str = Header(...)) -> Dict[str, Any]:
    verify_and_get_org(x_api_key)
    return {"total_requests_all_time": sum(USAGE_TRACKER.values()), "total_organizations": len(ORGANIZATIONS), "usage_by_month": dict(USAGE_TRACKER), "timestamp": datetime.utcnow().isoformat()}


@app.on_event("startup")
async def startup_event():
    logger.info("Orion Gateway API starting...")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Orion Gateway API shutting down...")


def run_orion(host: str = "0.0.0.0", port: int = 8001, debug: bool = False):
    import uvicorn
    uvicorn.run("dredge.orion_gateway:app", host=host, port=int(os.getenv("PORT", port)), log_level="info" if debug else "warning")


if __name__ == "__main__":
    run_orion(debug=True)
