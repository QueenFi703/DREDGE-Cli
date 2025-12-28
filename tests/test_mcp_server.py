"""Tests for the DREDGE MCP server."""
import subprocess
import time
import requests
import pytest


def test_mcp_server_import():
    """Test that the MCP server module can be imported."""
    from dredge import mcp_server
    assert hasattr(mcp_server, 'main')
    assert hasattr(mcp_server, 'mcp')


@pytest.mark.anyio
async def test_mcp_tools_registered():
    """Test that MCP tools are properly registered."""
    from dredge.mcp_server import mcp
    
    # Get list of tools
    tools = await mcp.list_tools()
    tool_names = [tool.name for tool in tools]
    
    # Verify our tools are registered
    assert 'get_version' in tool_names
    assert 'hello_world' in tool_names
    assert 'get_server_info' in tool_names


def test_tool_get_version():
    """Test the get_version tool."""
    from dredge.mcp_server import get_version
    import dredge
    
    result = get_version()
    assert f"DREDGE version: {dredge.__version__}" == result


def test_tool_hello_world():
    """Test the hello_world tool."""
    from dredge.mcp_server import hello_world
    
    # Test with default parameter
    result = hello_world()
    assert "Hello, World!" in result
    assert "DREDGE MCP Server" in result
    
    # Test with custom name
    result = hello_world("Alice")
    assert "Hello, Alice!" in result


def test_tool_get_server_info():
    """Test the get_server_info tool."""
    from dredge.mcp_server import get_server_info
    
    result = get_server_info()
    assert isinstance(result, dict)
    assert result['name'] == 'dredge-mcp-server'
    assert result['port'] == 3001
    assert 'Model Context Protocol' in result['protocol']


@pytest.mark.slow
def test_mcp_server_starts():
    """Test that the MCP server can start and listen on port 3001."""
    # Start the server in a subprocess
    process = subprocess.Popen(
        ['python', '-m', 'dredge.mcp_server'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    try:
        # Wait for server to start
        time.sleep(3)
        
        # Check that process is still running
        assert process.poll() is None, "Server process exited unexpectedly"
        
        # Try to connect to the SSE endpoint
        try:
            response = requests.get('http://localhost:3001/sse', stream=True, timeout=2)
            # SSE endpoints keep connection open, so we just check the status
            assert response.status_code == 200
            assert 'text/event-stream' in response.headers.get('content-type', '')
        except requests.exceptions.ReadTimeout:
            # Timeout is expected for SSE streams
            pass
        
    finally:
        # Clean up: terminate the server
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
