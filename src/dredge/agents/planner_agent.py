"""
DREDGE Planner Agent
Converts feature embeddings produced by the VisionAgent into a plan
(action logits) by running a planning network on the Jetson Thor GPU.

Pipeline position::

    VisionAgent  →  [PlannerAgent]  →  ReasoningAgent  →  Action
"""
import logging
from typing import Any, Dict

import torch
import torch.nn as nn

from dredge.events.ai_event import AIEvent, dispatch

logger = logging.getLogger("DREDGE.agents.PlannerAgent")


class _DefaultPlannerModel(nn.Module):
    """Lightweight default MLP used when no custom model is supplied."""

    def __init__(self, input_dim: int = 32, num_actions: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


class PlannerAgent:
    """
    Generates an action plan from feature embeddings.

    Routes workloads to the Jetson Thor GPU via the DREDGE event bus.

    Args:
        model:       Optional custom ``nn.Module``.  Defaults to a small MLP.
        input_dim:   Dimensionality of the incoming feature embeddings.
        num_actions: Number of discrete action logits to produce.

    Example::

        agent = PlannerAgent()
        action_logits = agent.run(embedding_tensor)
    """

    def __init__(
        self,
        model: nn.Module | None = None,
        input_dim: int = 32,
        num_actions: int = 8,
    ) -> None:
        self.model: nn.Module = (
            model if model is not None else _DefaultPlannerModel(input_dim, num_actions)
        )
        logger.info("PlannerAgent initialized")

    def run(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Run planning inference on *input_data*.

        Args:
            input_data: Feature-embedding tensor (output of VisionAgent).

        Returns:
            Action-logit tensor.
        """
        logger.debug("PlannerAgent dispatching AIEvent")
        event = AIEvent(model=self.model, input_data=input_data)
        return dispatch(event)

    def describe(self) -> Dict[str, Any]:
        """Return a description of this agent."""
        return {
            "agent": "PlannerAgent",
            "model": type(self.model).__name__,
            "role": "action planning from embeddings",
        }
