"""
DREDGE Reasoning Agent
Applies high-level reasoning over action logits from the PlannerAgent,
producing a final scalar decision score on the Jetson Thor GPU.

Pipeline position::

    VisionAgent  →  PlannerAgent  →  [ReasoningAgent]  →  Action
"""
import logging
from typing import Any, Dict

import torch
import torch.nn as nn

from dredge.events.ai_event import AIEvent, dispatch

logger = logging.getLogger("DREDGE.agents.ReasoningAgent")


class _DefaultReasoningModel(nn.Module):
    """Lightweight default MLP used when no custom model is supplied."""

    def __init__(self, input_dim: int = 8, output_dim: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


class ReasoningAgent:
    """
    Applies high-level reasoning to produce a final decision score.

    Routes workloads to the Jetson Thor GPU via the DREDGE event bus.

    Args:
        model:      Optional custom ``nn.Module``.  Defaults to a small MLP.
        input_dim:  Dimensionality of the incoming action logits.
        output_dim: Number of output values (default: 1 scalar score).

    Example::

        agent = ReasoningAgent()
        decision = agent.run(action_logits_tensor)
    """

    def __init__(
        self,
        model: nn.Module | None = None,
        input_dim: int = 8,
        output_dim: int = 1,
    ) -> None:
        self.model: nn.Module = (
            model if model is not None else _DefaultReasoningModel(input_dim, output_dim)
        )
        logger.info("ReasoningAgent initialized")

    def run(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Run reasoning inference on *input_data*.

        Args:
            input_data: Action-logit tensor (output of PlannerAgent).

        Returns:
            Decision-score tensor.
        """
        logger.debug("ReasoningAgent dispatching AIEvent")
        event = AIEvent(model=self.model, input_data=input_data)
        return dispatch(event)

    def describe(self) -> Dict[str, Any]:
        """Return a description of this agent."""
        return {
            "agent": "ReasoningAgent",
            "model": type(self.model).__name__,
            "role": "high-level reasoning and final decision",
        }
