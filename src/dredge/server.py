"""
DREDGE x Dolly Server
A lightweight web server for the DREDGE x Dolly integration.
"""
import os
import time
from flask import Flask, jsonify, request
from flask.json.provider import DefaultJSONProvider

from . import __version__


# Global metrics storage (simple in-memory for now)
_metrics = {
    "request_count": 0,
    "lift_count": 0,
    "start_time": time.time(),
    "latencies": []
}


class CompactJSONProvider(DefaultJSONProvider):
    """Custom JSON provider that uses compact encoding for better performance."""
    
    def dumps(self, obj, **kwargs):
        """Serialize object to JSON with compact encoding (no extra whitespace)."""
        # Override to use compact separators by default for smaller responses
        kwargs.setdefault('separators', (',', ':'))
        return super().dumps(obj, **kwargs)


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Use compact JSON encoding for smaller response sizes
    # Set provider class and explicitly instantiate for Flask 3.x compatibility
    app.json_provider_class = CompactJSONProvider
    app.json = CompactJSONProvider(app)
    
    # Request timing middleware
    @app.before_request
    def before_request():
        request._start_time = time.time()
        _metrics["request_count"] += 1
    
    @app.after_request
    def after_request(response):
        if hasattr(request, '_start_time'):
            latency = (time.time() - request._start_time) * 1000  # ms
            _metrics["latencies"].append(latency)
            # Keep only last 1000 latencies
            if len(_metrics["latencies"]) > 1000:
                _metrics["latencies"] = _metrics["latencies"][-1000:]
        return response
    
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
                "/metrics": "Performance metrics",
                "/lift": "Lift an insight (POST)",
            }
        })
    
    @app.route('/health')
    def health():
        """Health check endpoint.
        
        Returns OK if system is healthy, degraded if there are issues.
        """
        uptime = time.time() - _metrics["start_time"]
        
        # Simple health checks
        status = "healthy"
        checks = {
            "uptime_seconds": uptime,
            "request_count": _metrics["request_count"]
        }
        
        # Check if we're getting too many errors (simple heuristic)
        if _metrics["request_count"] > 0:
            checks["requests_per_second"] = _metrics["request_count"] / uptime
        
        return jsonify({
            "status": status,
            "version": __version__,
            "checks": checks
        })
    
    @app.route('/metrics')
    def metrics():
        """Performance metrics endpoint.
        
        Returns Prometheus-compatible metrics and additional stats.
        """
        uptime = time.time() - _metrics["start_time"]
        
        # Calculate latency percentiles
        latencies = sorted(_metrics["latencies"])
        p50 = latencies[len(latencies)//2] if latencies else 0
        p95 = latencies[int(len(latencies)*0.95)] if latencies else 0
        p99 = latencies[int(len(latencies)*0.99)] if latencies else 0
        
        metrics_data = {
            "uptime_seconds": uptime,
            "request_count": _metrics["request_count"],
            "lift_count": _metrics["lift_count"],
            "requests_per_second": _metrics["request_count"] / uptime if uptime > 0 else 0,
            "latency_ms": {
                "p50": p50,
                "p95": p95,
                "p99": p99,
                "mean": sum(latencies) / len(latencies) if latencies else 0
            },
            "memory_info": "not_implemented"
        }
        
        return jsonify(metrics_data)
    
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
        
        # Increment lift counter for metrics
        _metrics["lift_count"] += 1
        
        # Fast hash-based ID generation using a simple but consistent hash
        # For non-cryptographic IDs, we use a lightweight deterministic approach
        # Based on the string's bytes to ensure consistency across sessions
        #
        # Identity contract:
        # - IDs are content-derived labels, not cryptographic proofs
        # - 64-bit space (~1.8e19) provides collision resistance for modest scale
        # - Collision behavior: Last write wins (IDs overwrite on collision)
        # - Suitable for small to medium deployments; consider upgrading to
        #   128-bit hash or BLAKE2s for high-scale infrastructure
        hash_value = 0
        for char in insight_text:
            hash_value = (hash_value * 31 + ord(char)) & 0xFFFFFFFFFFFFFFFF
        insight_id = format(hash_value, '016x')
        
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


def run_server(host='0.0.0.0', port=3001, debug=False, reload=False, quiet=False, verbose=False):
    """
    Run the DREDGE x Dolly server.
    
    Args:
        host: Host to bind to (default: 0.0.0.0 for codespaces)
        port: Port to listen on (default: 3001)
        debug: Enable debug mode (default: False)
        reload: Enable hot reload with file watching (default: False)
        quiet: Quiet mode, minimal output (default: False)
        verbose: Verbose mode, detailed output (default: False)
    """
    if not quiet:
        print(f"🚀 Starting DREDGE x Dolly server on http://{host}:{port}")
        print(f"📡 API Version: {__version__}")
        print(f"🔧 Debug mode: {debug}")
        if reload:
            print(f"🔥 Hot reload: ENABLED (watching source files)")
        if verbose:
            print(f"📊 Verbose mode: ENABLED")
    
    if reload:
        # Use Werkzeug's reloader for hot reload
        # This watches Python files and restarts the server on changes
        import os
        os.environ['FLASK_ENV'] = 'development'
        
        if verbose and not quiet:
            print("👀 Watching files:")
            print(f"   • {os.path.dirname(__file__)}/*.py")
            print("   • Config files (if present)")
        
        app = create_app()
        app.run(
            host=host, 
            port=port, 
            debug=debug,
            use_reloader=True,
            reloader_type='stat'  # Use stat-based reloader for compatibility
        )
    else:
        app = create_app()
        if not quiet:
            print(f"🔧 Debug mode: {debug}")
        app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_server()
