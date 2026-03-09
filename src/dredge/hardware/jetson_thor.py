"""
DREDGE Jetson Thor Hardware Adapter
Connects the DREDGE compute pipeline to a NVIDIA Jetson Thor running JetPack 7.

This adapter routes AI workloads to the Jetson Thor's GPU via CUDA, falling back
to CPU when CUDA is not available (e.g., development machines without a GPU).

Architecture::

    DREDGE CLI
         │
    Event Bus
         │
    JetsonThor Adapter
         │
    CUDA / TensorRT
         │
    GPU on Jetson Thor
"""
import logging
from typing import Any, Dict

import torch
import torch.nn as nn

logger = logging.getLogger("DREDGE.hardware.JetsonThor")


class JetsonThor:
    """
    Hardware adapter for the NVIDIA Jetson Thor (JetPack 7).

    Automatically detects CUDA availability and routes model computation to the
    Jetson GPU when present, falling back to CPU for development environments.

    Example::

        from dredge.hardware.jetson_thor import JetsonThor

        thor = JetsonThor()
        result = thor.compute(model, input_tensor)
    """

    def __init__(self, device: str = "auto") -> None:
        """
        Initialize the JetsonThor adapter.

        Args:
            device: Target device.  One of ``'auto'``, ``'cuda'``, or
                    ``'cpu'``.  ``'auto'`` selects CUDA when available,
                    otherwise CPU (matches JetPack 7 behaviour on the Thor).
        """
        if device == "auto":
            self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda":
            # Gracefully degrade when CUDA is not present (e.g. CI runners)
            if not torch.cuda.is_available():
                logger.warning(
                    "CUDA requested but not available; falling back to CPU. "
                    "On Jetson Thor verify JetPack 7 CUDA drivers are installed."
                )
                self.device = "cpu"
            else:
                self.device = "cuda"
        else:
            self.device = device

        logger.info("JetsonThor adapter initialized on device=%s", self.device)

    # ------------------------------------------------------------------
    # Core compute
    # ------------------------------------------------------------------

    def compute(self, model: nn.Module, data: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass on the Jetson Thor GPU (or CPU fallback).

        Moves both *model* and *data* to the configured device before
        inference so callers do not need to manage device placement manually.

        Args:
            model: A ``torch.nn.Module`` to run inference with.
            data:  Input tensor.

        Returns:
            Output tensor (on the same device as *data* was moved to).
        """
        model = model.to(self.device)
        data = data.to(self.device)
        return model(data)

    # ------------------------------------------------------------------
    # Device info
    # ------------------------------------------------------------------

    def device_info(self) -> Dict[str, Any]:
        """
        Return a dictionary describing the current compute device.

        Returns:
            Dictionary with keys ``device``, ``cuda_available``,
            ``cuda_device_name``, and ``cuda_version``.
        """
        info: Dict[str, Any] = {
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": None,
            "cuda_version": torch.version.cuda,
        }
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
        return info
