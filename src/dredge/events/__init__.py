"""
DREDGE Event Bus
Provides event classes and the central dispatcher for AI workload routing.
"""
from .ai_event import AIEvent, dispatch

__all__ = ["AIEvent", "dispatch"]
