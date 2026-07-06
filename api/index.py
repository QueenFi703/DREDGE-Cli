"""
DREDGE Studio - Vercel API Entry Point
Gordon Integration + DREDGE Backend
"""

import sys
from pathlib import Path

# Add source to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from flask import Flask, jsonify
from dredge.advanced_features import dredge_advanced
from dredge.dependabot_alerts import (
    dependabot_bp,
    register_dependabot_alerts
)

# Initialize Flask app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Register blueprints
app.register_blueprint(dredge_advanced)
register_dependabot_alerts(app)

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Home endpoint"""
    return jsonify({
        "name": "DREDGE Studio",
        "version": "2.0.0",
        "status": "operational",
        "deployment": "vercel",
        "endpoints": {
            "dashboard": "/advanced",
            "health": "/health",
            "api": "/api/",
            "dependabot": "/api/dependabot/",
            "advanced": "/api/advanced/",
            "docs": "/docs"
        },
        "features": [
            "FiBot Security Intelligence",
            "Dependabot Alerts",
            "String Theory Computation",
            "DREDGE Pipeline",
            "Model Management",
            "MCP Operations",
            "Advanced Visualization"
        ]
    })


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "version": "2.0.0",
        "deployment": "vercel"
    })


@app.route('/api')
def api_root():
    """API root"""
    return jsonify({
        "api": "DREDGE Studio API v1",
        "version": "2.0.0",
        "endpoints": [
            "/api/advanced/",
            "/api/dependabot/",
            "/api/dredge/",
            "/health"
        ]
    })


@app.route('/api/dredge/status')
def dredge_status():
    """DREDGE status endpoint"""
    return jsonify({
        "status": "operational",
        "version": "2.0.0",
        "deployment": "vercel",
        "features": [
            "Model Management",
            "MCP Operations",
            "Insight Lifting",
            "DREDGE Pipeline",
            "Swift Toolchain",
            "Code Generation",
            "Dependabot Alerts",
            "Container Status",
            "API Testing",
            "Visualization"
        ]
    })


@app.route('/api/dredge/lift', methods=['POST'])
def lift_insight():
    """Lift insight endpoint"""
    from flask import request
    
    data = request.get_json() or {}
    insight = data.get('insight', '')
    
    if not insight:
        return jsonify({"error": "Missing insight parameter"}), 400
    
    return jsonify({
        "status": "lifted",
        "original": insight,
        "enhanced": f"[Enhanced via DREDGE] {insight}",
        "confidence": 0.89,
        "models_used": ["Quasimoto 4D", "String Theory 10D", "DREDGE Reasoner"]
    })


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found", "status": 404}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({"error": "Server error", "status": 500}), 500


# ============================================================================
# GORDON INTEGRATION
# ============================================================================

@app.route('/api/gordon/capabilities')
def gordon_capabilities():
    """Gordon integration capabilities"""
    return jsonify({
        "gordon_version": "1.0.0",
        "dredge_integration": "active",
        "capabilities": [
            {
                "name": "Security Analysis",
                "endpoint": "/api/dependabot/fibot/chat",
                "method": "POST",
                "description": "Analyze vulnerabilities with FiBot"
            },
            {
                "name": "Model Management",
                "endpoint": "/api/advanced/models/list",
                "method": "GET",
                "description": "List and manage AI models"
            },
            {
                "name": "String Theory",
                "endpoint": "/api/advanced/visualization/string-spectrum",
                "method": "POST",
                "description": "Compute string vibrational spectrum"
            },
            {
                "name": "Monitoring",
                "endpoint": "/api/advanced/containers/status",
                "method": "GET",
                "description": "Get system monitoring metrics"
            },
            {
                "name": "Recommendations",
                "endpoint": "/api/dependabot/recommendations",
                "method": "GET",
                "description": "Get FiBot recommendations"
            }
        ]
    })


# ============================================================================
# VERCEL HANDLER
# ============================================================================

# Vercel expects the app object for ASGI/WSGI
handler = app.wsgi_app
