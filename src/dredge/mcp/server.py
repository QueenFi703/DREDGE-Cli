"""
DREDGE MCP Agent Server
FastAPI server that exposes DREDGE's AI-agent pipeline over an HTTP API so that
external systems (laptops, other Jetson boards, cloud services) can submit
inference workloads and receive results from the Jetson Thor GPU.

Architecture::

    DREDGE CLI
         │
    Event Bus
         │
    Agent Layer  (VisionAgent / PlannerAgent / ReasoningAgent)
         │
    JetsonThor Adapter
         │
    CUDA / TensorRT
         │
    Thor GPU

Endpoints:
    POST /compute          – raw model inference via JetsonThor
    POST /agent/vision     – VisionAgent forward pass
    POST /agent/planner    – PlannerAgent forward pass
    POST /agent/reasoning  – ReasoningAgent forward pass
    POST /pipeline         – full Vision → Planner → Reasoning pipeline
    GET  /health           – server health + device info
"""
import logging
from typing import Any, Dict, List

import torch
import torch.nn as nn

logger = logging.getLogger("DREDGE.mcp.server")

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    _FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FASTAPI_AVAILABLE = False
    FastAPI = None  # type: ignore[assignment,misc]
    HTTPException = None  # type: ignore[assignment]
    BaseModel = object  # type: ignore[assignment,misc]

from dredge.agents.planner_agent import PlannerAgent
from dredge.agents.reasoning_agent import ReasoningAgent
from dredge.agents.vision_agent import VisionAgent
from dredge.hardware.jetson_thor import JetsonThor


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class ComputeRequest(BaseModel):  # type: ignore[misc]
    """Payload for the generic /compute endpoint."""

    data: List[float]
    input_dim: int = 64
    output_dim: int = 32


class AgentRequest(BaseModel):  # type: ignore[misc]
    """Payload for single-agent endpoints."""

    data: List[float]


class PipelineRequest(BaseModel):  # type: ignore[misc]
    """Payload for the full Vision → Planner → Reasoning pipeline."""

    data: List[float]
    vision_output_dim: int = 32
    planner_num_actions: int = 8


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_mcp_agent_app(device: str = "auto") -> "FastAPI":
    """
    Create and configure the FastAPI MCP agent application.

    Args:
        device: Target device for the JetsonThor adapter
                (``'auto'``, ``'cuda'``, or ``'cpu'``).

    Returns:
        Configured :class:`fastapi.FastAPI` instance.

    Raises:
        ImportError: If ``fastapi`` is not installed.
    """
    if not _FASTAPI_AVAILABLE:
        raise ImportError(
            "fastapi is required for the MCP agent server. "
            "Install it with: pip install fastapi uvicorn"
        )

    _app = FastAPI(
        title="DREDGE MCP Agent Server",
        description=(
            "AI-agent inference server backed by a NVIDIA Jetson Thor (JetPack 7). "
            "Exposes DREDGE's Vision → Planner → Reasoning pipeline over HTTP."
        ),
        version="1.0.0",
    )

    thor = JetsonThor(device=device)
    vision_agent = VisionAgent()
    planner_agent = PlannerAgent()
    reasoning_agent = ReasoningAgent()

    # -----------------------------------------------------------------------
    # Health
    # -----------------------------------------------------------------------

    @_app.get("/health")
    def health() -> Dict[str, Any]:
        """Return server health status and device information."""
        return {
            "status": "healthy",
            "device_info": thor.device_info(),
            "agents": ["VisionAgent", "PlannerAgent", "ReasoningAgent"],
        }

    # -----------------------------------------------------------------------
    # Generic compute
    # -----------------------------------------------------------------------

    @_app.post("/compute")
    def compute(payload: ComputeRequest) -> Dict[str, Any]:
        """
        Run inference using a lightweight default MLP on the JetsonThor.

        The *data* list is treated as a flat feature vector.  A default
        ``nn.Linear`` model is constructed with the given dimensions.
        """
        try:
            data_tensor = torch.tensor(payload.data, dtype=torch.float32).unsqueeze(0)
            model = nn.Linear(payload.input_dim, payload.output_dim)
            result = thor.compute(model, data_tensor)
            return {"result": result.detach().cpu().tolist()}
        except Exception as exc:
            logger.exception("Error in /compute")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    # -----------------------------------------------------------------------
    # Individual agents
    # -----------------------------------------------------------------------

    @_app.post("/agent/vision")
    def agent_vision(payload: AgentRequest) -> Dict[str, Any]:
        """Run the VisionAgent on the supplied data vector."""
        try:
            tensor = torch.tensor(payload.data, dtype=torch.float32).unsqueeze(0)
            result = vision_agent.run(tensor)
            return {"result": result.detach().cpu().tolist(), "agent": "VisionAgent"}
        except Exception as exc:
            logger.exception("Error in /agent/vision")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @_app.post("/agent/planner")
    def agent_planner(payload: AgentRequest) -> Dict[str, Any]:
        """Run the PlannerAgent on the supplied embedding vector."""
        try:
            tensor = torch.tensor(payload.data, dtype=torch.float32).unsqueeze(0)
            result = planner_agent.run(tensor)
            return {"result": result.detach().cpu().tolist(), "agent": "PlannerAgent"}
        except Exception as exc:
            logger.exception("Error in /agent/planner")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @_app.post("/agent/reasoning")
    def agent_reasoning(payload: AgentRequest) -> Dict[str, Any]:
        """Run the ReasoningAgent on the supplied action-logit vector."""
        try:
            tensor = torch.tensor(payload.data, dtype=torch.float32).unsqueeze(0)
            result = reasoning_agent.run(tensor)
            return {"result": result.detach().cpu().tolist(), "agent": "ReasoningAgent"}
        except Exception as exc:
            logger.exception("Error in /agent/reasoning")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    # -----------------------------------------------------------------------
    # Full pipeline
    # -----------------------------------------------------------------------

    @_app.post("/pipeline")
    def pipeline(payload: PipelineRequest) -> Dict[str, Any]:
        """
        Run the full Vision → Planner → Reasoning pipeline.

        The input data passes through each agent in sequence:

        1. **VisionAgent** extracts feature embeddings.
        2. **PlannerAgent** maps embeddings to action logits.
        3. **ReasoningAgent** produces a final decision score.
        """
        try:
            # Build pipeline-specific agents with matching dimensions
            _vision = VisionAgent(
                input_dim=len(payload.data),
                output_dim=payload.vision_output_dim,
            )
            _planner = PlannerAgent(
                input_dim=payload.vision_output_dim,
                num_actions=payload.planner_num_actions,
            )
            _reasoning = ReasoningAgent(input_dim=payload.planner_num_actions)

            sensor_input = torch.tensor(payload.data, dtype=torch.float32).unsqueeze(0)

            embedding = _vision.run(sensor_input)
            action_logits = _planner.run(embedding)
            decision = _reasoning.run(action_logits)

            return {
                "embedding": embedding.detach().cpu().tolist(),
                "action_logits": action_logits.detach().cpu().tolist(),
                "decision": decision.detach().cpu().tolist(),
                "pipeline": "Vision → Planner → Reasoning",
            }
        except Exception as exc:
            logger.exception("Error in /pipeline")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return _app


# ---------------------------------------------------------------------------
# Module-level singleton app (device resolved at import time)
# ---------------------------------------------------------------------------

app: "FastAPI | None" = None
if _FASTAPI_AVAILABLE:
    app = create_mcp_agent_app()


# ---------------------------------------------------------------------------
# Server runner
# ---------------------------------------------------------------------------

def run_mcp_agent_server(
    host: str = "0.0.0.0",
    port: int = 3003,
    device: str = "auto",
    reload: bool = False,
) -> None:
    """
    Start the MCP agent server using Uvicorn.

    Args:
        host:   Bind address (default ``'0.0.0.0'``).
        port:   TCP port (default ``3003``).
        device: Target device for JetsonThor (``'auto'``, ``'cuda'``, ``'cpu'``).
        reload: Enable auto-reload for development.

    Raises:
        ImportError: If ``fastapi`` or ``uvicorn`` is not installed.
    """
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            "uvicorn is required to run the MCP agent server. "
            "Install it with: pip install uvicorn"
        ) from exc

    _server_app = create_mcp_agent_app(device=device)
    logger.info(
        "Starting DREDGE MCP Agent Server on http://%s:%d (device=%s)",
        host,
        port,
        device,
    )
    uvicorn.run(_server_app, host=host, port=port, reload=reload)
