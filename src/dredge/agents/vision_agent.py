"""
DREDGE Vision Agent
Processes visual / sensor input data through a neural model running on
the Jetson Thor GPU, producing feature embeddings for downstream agents.

Pipeline position::

    [VisionAgent]  →  PlannerAgent  →  ReasoningAgent  →  Action
"""
import logging
from typing import Any, Dict

import torch
import torch.nn as nn

from dredge.events.ai_event import AIEvent, dispatch

logger = logging.getLogger("DREDGE.agents.VisionAgent")


class _DefaultVisionModel(nn.Module):
    """Lightweight default CNN used when no custom model is supplied."""

    def __init__(self, input_dim: int = 64, output_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


class VisionAgent:
    """
    Processes visual or sensor input and extracts feature embeddings.

    Routes workloads to the Jetson Thor GPU via the DREDGE event bus.

    Args:
        model:      Optional custom ``nn.Module``.  Defaults to a small MLP.
        input_dim:  Feature dimensionality expected by the default model.
        output_dim: Embedding size produced by the default model.

    Example::

        agent = VisionAgent()
        embedding = agent.run(sensor_tensor)
    """

    def __init__(
        self,
        model: nn.Module | None = None,
        input_dim: int = 64,
        output_dim: int = 32,
    ) -> None:
        self.model: nn.Module = (
            model if model is not None else _DefaultVisionModel(input_dim, output_dim)
        )
        logger.info("VisionAgent initialized")

    def run(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Run visual inference on *input_data*.

        Args:
            input_data: Input sensor/image tensor.

        Returns:
            Feature-embedding tensor.
        """
        logger.debug("VisionAgent dispatching AIEvent")
        event = AIEvent(model=self.model, input_data=input_data)
        return dispatch(event)

    def describe(self) -> Dict[str, Any]:
        """Return a description of this agent."""
        return {
            "agent": "VisionAgent",
            "model": type(self.model).__name__,
            "role": "visual feature extraction",
        }
