"""
Flask Legacy Adapters - Example WSGI Apps to mount in Hybrid Gateway

These are example Flask applications that demonstrate how to:
1. Create standalone Flask apps (WSGI)
2. Mount them in the FastAPI core (ASGI) via WSGIMiddleware
3. Create a single ASGI application for deployment

Each can be mounted at different paths within the gateway.
"""

from flask import Flask, jsonify, request
from typing import Dict, Any

# ============================================================================
# LEGACY ADAPTER 1: Dashboard Legacy
# ============================================================================

def create_dashboard_legacy() -> Flask:
    """
    Legacy dashboard adapter - Flask WSGI app
    Can be mounted at /legacy/dashboard or similar
    """
    app = Flask(__name__)
    
    @app.route('/', methods=['GET'])
    def index():
        """Dashboard index"""
        return jsonify({
            "adapter": "dashboard_legacy",
            "type": "Flask WSGI",
            "features": ["Dashboard", "Reports", "Analytics"]
        })
    
    @app.route('/data', methods=['GET'])
    def get_data():
        """Get dashboard data"""
        return jsonify({
            "data": [
                {"name": "Metric 1", "value": 100},
                {"name": "Metric 2", "value": 200},
                {"name": "Metric 3", "value": 300}
            ]
        })
    
    @app.route('/render', methods=['POST'])
    def render_dashboard():
        """Render dashboard"""
        data = request.get_json()
        return jsonify({
            "status": "rendered",
            "dashboard": data
        })
    
    return app


# ============================================================================
# LEGACY ADAPTER 2: API Legacy
# ============================================================================

def create_api_legacy() -> Flask:
    """
    Legacy API adapter - Flask WSGI app
    Can be mounted at /legacy/api or similar
    """
    app = Flask(__name__)
    
    @app.route('/status', methods=['GET'])
    def status():
        """API status"""
        return jsonify({
            "status": "online",
            "version": "1.0.0",
            "adapter": "api_legacy"
        })
    
    @app.route('/data', methods=['GET'])
    def get_data():
        """Get API data"""
        return jsonify({
            "data": {
                "items": [
                    {"id": 1, "name": "Item 1"},
                    {"id": 2, "name": "Item 2"}
                ]
            }
        })
    
    @app.route('/process', methods=['POST'])
    def process_data():
        """Process data"""
        data = request.get_json()
        return jsonify({
            "status": "processed",
            "input": data,
            "result": "success"
        })
    
    return app


# ============================================================================
# LEGACY ADAPTER 3: Auth Legacy
# ============================================================================

def create_auth_legacy() -> Flask:
    """
    Legacy auth adapter - Flask WSGI app
    Can be mounted at /legacy/auth or similar
    """
    app = Flask(__name__)
    
    @app.route('/status', methods=['GET'])
    def status():
        """Auth status"""
        return jsonify({
            "status": "operational",
            "adapter": "auth_legacy",
            "features": ["Login", "Logout", "Token Management"]
        })
    
    @app.route('/login', methods=['POST'])
    def login():
        """Login endpoint"""
        credentials = request.get_json()
        return jsonify({
            "status": "logged_in",
            "token": "legacy_token_xyz123",
            "user": credentials.get("username")
        })
    
    @app.route('/token/validate', methods=['POST'])
    def validate_token():
        """Validate token"""
        token = request.headers.get('X-Token')
        return jsonify({
            "valid": True,
            "token": token,
            "user": "user@example.com"
        })
    
    return app


# ============================================================================
# MOUNTING EXAMPLE IN HYBRID GATEWAY
# ============================================================================

"""
To use these adapters in hybrid_gateway.py:

from flask_legacy_adapters import create_dashboard_legacy, create_api_legacy, create_auth_legacy
from starlette.middleware.wsgi import WSGIMiddleware

# In mount_adapters() function:

# Mount Dashboard Legacy
dashboard = create_dashboard_legacy()
app.mount("/legacy/dashboard", WSGIMiddleware(dashboard))

# Mount API Legacy
api = create_api_legacy()
app.mount("/legacy/api", WSGIMiddleware(api))

# Mount Auth Legacy
auth = create_auth_legacy()
app.mount("/legacy/auth", WSGIMiddleware(auth))

Result:
  GET /legacy/dashboard/data → Flask dashboard
  POST /legacy/api/process → Flask API
  POST /legacy/auth/login → Flask auth
"""
