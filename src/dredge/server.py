"""
DREDGE x Dolly Server
A lightweight web server for the DREDGE x Dolly integration.
Includes MCP tools for Quasimoto wave processing.
"""
import hashlib
import os
from functools import lru_cache
from flask import Flask, jsonify, request

from . import __version__
from . import mcp_server


@lru_cache(maxsize=1024)
def _compute_insight_hash(insight_text: str) -> str:
    """Compute SHA256 hash of insight text with caching for repeated insights."""
    return hashlib.sha256(insight_text.encode()).hexdigest()


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        """Root endpoint with API information."""
        return jsonify({
            "name": "DREDGE x Dolly",
            "version": __version__,
            "description": "GPU-CPU Lifter · Save · Files · Print",
            "endpoints": {
                "/": "API information",
                "/health": "Health check",
                "/lift": "Lift an insight (POST)",
                "/mcp/tools": "List available MCP tools",
                "/mcp/call": "Call an MCP tool (POST)",
            }
        })
    
    @app.route('/health')
    def health():
        """Health check endpoint."""
        return jsonify({"status": "healthy", "version": __version__})
    
    @app.route('/lift', methods=['POST'])
    def lift_insight():
        """
        Lift an insight with Dolly integration.
        
        Expected JSON payload:
        {
            "insight_text": "Your insight text here"
        }
        """
        data = request.get_json()
        
        if not data or 'insight_text' not in data:
            return jsonify({
                "error": "Missing required field: insight_text"
            }), 400
        
        insight_text = data['insight_text']
        
        # Optimized: Use cached hash computation for duplicate insights
        insight_id = _compute_insight_hash(insight_text)
        
        # Basic insight structure
        # Note: Full Dolly GPU integration would require PyTorch
        result = {
            "id": insight_id,
            "text": insight_text,
            "lifted": True,
            "message": "Insight processed (full GPU acceleration requires PyTorch/Dolly setup)"
        }
        
        return jsonify(result)
    
    @app.route('/mcp/tools')
    def mcp_tools():
        """List available MCP tools."""
        return jsonify(mcp_server.get_tool_info())
    
    @app.route('/mcp/call', methods=['POST'])
    def mcp_call():
        """
        Call an MCP tool.
        
        Expected JSON payload:
        {
            "tool": "tool_name",
            "parameters": {...}
        }
        """
        data = request.get_json()
        
        if not data or 'tool' not in data:
            return jsonify({
                "error": "Missing required field: tool"
            }), 400
        
        tool_name = data['tool']
        parameters = data.get('parameters', {})
        
        result = mcp_server.call_tool(tool_name, **parameters)
        return jsonify(result)
    
    return app


def run_server(host='0.0.0.0', port=3001, debug=False):
    """
    Run the DREDGE x Dolly server.
    
    Args:
        host: Host to bind to (default: 0.0.0.0 for codespaces)
        port: Port to listen on (default: 3001)
        debug: Enable debug mode (default: False)
    """
    app = create_app()
    print(f"🚀 Starting DREDGE x Dolly server on http://{host}:{port}")
    print(f"📡 API Version: {__version__}")
    print(f"🔧 Debug mode: {debug}")
    print(f"🌊 MCP Tools: {len(mcp_server.MCP_TOOLS)} available")
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_server()
