"""
DREDGE Autonomous Agents
Specialised agents that run on Jetson Thor GPU compute via the DREDGE event bus.

Each agent wraps a specific AI workload and dispatches it through the
:class:`~dredge.events.ai_event.AIEvent` / :func:`~dredge.events.ai_event.dispatch`
pipeline so that all computation is automatically accelerated by the JetsonThor
hardware adapter.

Available agents::

    Vision → Analysis → Decision → Action
"""
from .planner_agent import PlannerAgent
from .reasoning_agent import ReasoningAgent
from .vision_agent import VisionAgent

__all__ = ["VisionAgent", "PlannerAgent", "ReasoningAgent"]
