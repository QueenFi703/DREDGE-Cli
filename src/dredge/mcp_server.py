"""
MCP Server for DREDGE - Model Context Protocol Server
This server exposes tools and resources for AI agents via the Model Context Protocol.
"""
import sys
import logging
from mcp.server.fastmcp import FastMCP

# Configure logging to stderr to avoid interfering with MCP protocol
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP(
    name="dredge-mcp-server",
    host="0.0.0.0",
    port=3001
)


@mcp.tool()
def get_version() -> str:
    """Get the DREDGE package version."""
    from . import __version__
    return f"DREDGE version: {__version__}"


@mcp.tool()
def hello_world(name: str = "World") -> str:
    """
    Say hello to someone.
    
    Args:
        name: The name to greet (default: "World")
    
    Returns:
        A greeting message
    """
    return f"Hello, {name}! Welcome to DREDGE MCP Server."


@mcp.tool()
def get_server_info() -> dict:
    """
    Get information about the MCP server.
    
    Returns:
        Dictionary with server information
    """
    return {
        "name": "dredge-mcp-server",
        "port": 3001,
        "protocol": "Model Context Protocol (MCP)",
        "description": "DREDGE MCP Server for AI agent integration"
    }


def main():
    """Main entry point for the MCP server."""
    logger.info("Starting DREDGE MCP server on 0.0.0.0:3001...")
    logger.info("Available tools: get_version, hello_world, get_server_info")
    logger.info("Using SSE (Server-Sent Events) transport")
    try:
        mcp.run(transport="sse")
    except KeyboardInterrupt:
        logger.info("Shutting down MCP server...")
    except Exception as e:
        logger.error(f"Error running MCP server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
