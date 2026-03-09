"""
DREDGE AI Event
Defines the AIEvent type and the central dispatch function that routes
AI inference workloads to the appropriate hardware adapter.

Event flow::

    DREDGE CLI
         │
    Event Bus  ←  AIEvent(model, input_data)
         │
    JetsonThor Adapter
         │
    CUDA / TensorRT
         │
    GPU on Jetson Thor
"""
import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger("DREDGE.events.AIEvent")


class AIEvent:
    """
    Encapsulates an AI inference workload for dispatch over the event bus.

    Attributes:
        model:      The ``torch.nn.Module`` to run inference with.
        input_data: Input tensor or data to feed into *model*.

    Example::

        from dredge.events.ai_event import AIEvent, dispatch

        event = AIEvent(model=my_model, input_data=my_tensor)
        result = dispatch(event)
    """

    def __init__(self, model: nn.Module, input_data: torch.Tensor) -> None:
        self.model = model
        self.input_data = input_data


def dispatch(event: Any) -> torch.Tensor:
    """
    Route an event to the appropriate hardware adapter and return the result.

    Currently handles :class:`AIEvent` by forwarding to
    :class:`~dredge.hardware.jetson_thor.JetsonThor`.

    Args:
        event: The event to dispatch.  Must be an :class:`AIEvent`.

    Returns:
        The output tensor produced by the model.

    Raises:
        TypeError: If *event* is not a recognised event type.
    """
    # Import lazily to avoid circular dependencies and to allow the hardware
    # adapter to be mocked in tests without importing torch at module load time.
    from dredge.hardware.jetson_thor import JetsonThor

    if isinstance(event, AIEvent):
        thor = JetsonThor()
        logger.info(
            "Dispatching AIEvent to JetsonThor (device=%s)", thor.device
        )
        return thor.compute(event.model, event.input_data)

    raise TypeError(
        f"dispatch() received an unknown event type: {type(event).__name__!r}. "
        "Expected AIEvent."
    )
