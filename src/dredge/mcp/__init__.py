"""
DREDGE MCP Sub-package
Exposes the FastAPI-based AI-Agent MCP server for Jetson Thor.
"""
from .server import app, create_mcp_agent_app, run_mcp_agent_server

__all__ = ["app", "create_mcp_agent_app", "run_mcp_agent_server"]
