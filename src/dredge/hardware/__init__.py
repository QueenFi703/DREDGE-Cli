"""
DREDGE Hardware Adapters
Provides hardware-specific adapters for AI acceleration (e.g. NVIDIA Jetson Thor).
"""
from .jetson_thor import JetsonThor

__all__ = ["JetsonThor"]
