# Quick Start: MCP Server in GitHub Codespaces

## What Was Added

This repository now includes a Model Context Protocol (MCP) server that automatically starts when you open it in GitHub Codespaces.

### Files Added/Modified:

1. **`.devcontainer/devcontainer.json`** - Configures GitHub Codespaces to:
   - Forward port 3001 automatically
   - Install dependencies on container creation
   - Start the MCP server automatically on startup

2. **`src/dredge/mcp_server.py`** - The MCP server implementation with tools:
   - `get_version()` - Returns the DREDGE version
   - `hello_world(name)` - A greeting tool
   - `get_server_info()` - Returns server information

3. **`tests/test_mcp_server.py`** - Comprehensive test suite for the MCP server

4. **`requirements.txt` & `pyproject.toml`** - Updated with MCP dependencies:
   - `mcp[cli]>=0.9.0`
   - `fastmcp>=0.2.0`

5. **`MCP_SERVER.md`** - Detailed documentation about the MCP server

6. **`README.md`** - Updated to include MCP server information

## How to Use in GitHub Codespaces

### Automatic Setup

1. Open this repository in GitHub Codespaces
2. The MCP server will automatically:
   - Install dependencies
   - Start listening on port 3001
   - Forward the port (you'll get a notification)

### Manual Control

Start the server manually:
```bash
python -m dredge.mcp_server
```

Stop the server with `Ctrl+C`.

### Verify the Server is Running

```bash
# Check if the server is listening
curl http://localhost:3001/sse

# Or use the test suite
pytest tests/test_mcp_server.py
```

## Connecting AI Clients

Once the server is running in Codespaces:

1. Find the forwarded port URL in the "Ports" tab (usually something like `https://xxx-3001.app.github.dev`)
2. Configure your AI client (Claude Desktop, ChatGPT, etc.) to connect to:
   - **Endpoint**: `<forwarded-url>/sse` for SSE transport
   - **Protocol**: HTTP/SSE

## Available Tools

The MCP server exposes these tools to AI agents:

| Tool | Description | Parameters |
|------|-------------|------------|
| `get_version` | Get DREDGE version | None |
| `hello_world` | Greeting function | `name: str` (optional) |
| `get_server_info` | Get server info | None |

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run only MCP tests
pytest tests/test_mcp_server.py

# Run without the slow integration test
pytest tests/test_mcp_server.py -k "not test_mcp_server_starts"
```

## Troubleshooting

### Server won't start

Check if port 3001 is already in use:
```bash
lsof -i :3001
```

### Can't connect to the server

1. Verify the server is running: `ps aux | grep mcp_server`
2. Check the logs in the terminal
3. Ensure the port is forwarded in Codespaces

### Dependencies not installed

Reinstall:
```bash
pip install -e .
```

## Next Steps

To add more tools to the MCP server:

1. Edit `src/dredge/mcp_server.py`
2. Add a function decorated with `@mcp.tool()`
3. Document the function with a docstring
4. Add tests to `tests/test_mcp_server.py`

Example:
```python
@mcp.tool()
def my_tool(param: str) -> dict:
    """
    Description of what the tool does.
    
    Args:
        param: Description of the parameter
    
    Returns:
        Result as a dictionary
    """
    return {"result": param}
```

## Resources

- [MCP Documentation](https://modelcontextprotocol.io/)
- [FastMCP on GitHub](https://github.com/jlowin/fastmcp)
- [GitHub Codespaces Docs](https://docs.github.com/en/codespaces)
