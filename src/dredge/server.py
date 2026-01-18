"""
DREDGE x Dolly Server
A lightweight web server for the DREDGE x Dolly integration.
"""
import hashlib
import os
import logging
from functools import lru_cache
from flask import Flask, jsonify, request

from . import __version__
from .config import load_config


def setup_logging(debug: bool = False):
    """Setup logging configuration."""
    config = load_config()
    log_config = config.get("logging", {})
    level = logging.DEBUG if debug else getattr(logging, log_config.get("level", "INFO"))
    log_format = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


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
    
    return app


def run_server(host='0.0.0.0', port=3001, debug=False):
    """
    Run the DREDGE x Dolly server.
    
    Args:
        host: Host to bind to (default: 0.0.0.0 for codespaces)
        port: Port to listen on (default: 3001)
        debug: Enable debug mode (default: False)
    """
    logger = setup_logging(debug)
    
    logger.info(f"Starting DREDGE x Dolly Server v{__version__}")
    logger.info(f"Host: {host}, Port: {port}, Debug: {debug}")
    
    app = create_app()
    
    print(f"🚀 Starting DREDGE x Dolly server on http://{host}:{port}")
    print(f"📡 API Version: {__version__}")
    print(f"🔧 Debug mode: {debug}")
    
    logger.info("Server ready. Press CTRL+C to stop.")
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_server()
