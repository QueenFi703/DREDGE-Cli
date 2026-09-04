"""
DREDGE Studio - API Entry Point
Gordon Integration + DREDGE Backend + Nebius/NVIDIA Nemotron
"""

import sys
from pathlib import Path

# Add source to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from flask import Flask, jsonify, request
from dredge.advanced_features import dredge_advanced
from dredge.dependabot_alerts import register_dependabot_alerts
from dredge.nebius import (
    DEFAULT_MODEL,
    get_nebius_config,
    nebius_configured,
    dredge_reason,
    NebiusConfigurationError,
)

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

app.register_blueprint(dredge_advanced)
register_dependabot_alerts(app)


@app.route("/")
def index():
    return jsonify({
        "name": "DREDGE Studio",
        "version": "2.0.0",
        "status": "operational",
        "deployment": "nebius-ready",
        "endpoints": {
            "dashboard": "/advanced",
            "health": "/health",
            "api": "/api/",
            "dependabot": "/api/dependabot/",
            "advanced": "/api/advanced/",
            "nebius_status": "/api/dredge/nebius/status",
            "nebius_reason": "/api/dredge/nebius/reason",
        },
        "features": [
            "FiBot Security Intelligence",
            "Dependabot Alerts",
            "String Theory Computation",
            "DREDGE Pipeline",
            "Model Management",
            "MCP Operations",
            "NVIDIA Nemotron via Nebius Token Factory",
        ],
    })


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "version": "2.0.0",
        "nebius_configured": nebius_configured(),
    })


@app.route("/api")
def api_root():
    return jsonify({
        "api": "DREDGE Studio API v1",
        "version": "2.0.0",
        "endpoints": [
            "/api/advanced/",
            "/api/dependabot/",
            "/api/dredge/",
            "/api/dredge/nebius/status",
            "/api/dredge/nebius/reason",
            "/health",
        ],
    })


@app.route("/api/dredge/status")
def dredge_status():
    return jsonify({
        "status": "operational",
        "version": "2.0.0",
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
            "Visualization",
            "Nebius Token Factory",
            "NVIDIA Nemotron",
        ],
    })


@app.route("/api/dredge/lift", methods=["POST"])
def lift_insight():
    data = request.get_json(silent=True) or {}
    insight = data.get("insight", "")
    if not insight:
        return jsonify({"error": "Missing insight parameter"}), 400

    return jsonify({
        "status": "lifted",
        "original": insight,
        "enhanced": f"[Enhanced via DREDGE] {insight}",
        "confidence": 0.89,
        "models_used": ["Quasimoto 4D", "String Theory 10D", "DREDGE Reasoner"],
    })


@app.route("/api/dredge/nebius/status")
def nebius_status():
    """Expose integration health without ever returning the API key."""
    config = get_nebius_config()
    return jsonify({
        "provider": "Nebius Token Factory",
        "model": config["model"] if nebius_configured() else DEFAULT_MODEL,
        "base_url": config["base_url"],
        "configured": nebius_configured(),
        "nvidia_model": config["model"].startswith("nvidia/"),
    })


@app.route("/api/dredge/nebius/reason", methods=["POST"])
def nebius_reason():
    """Run a DREDGE reasoning task with NVIDIA Nemotron on Nebius."""
    data = request.get_json(silent=True) or {}
    prompt = str(data.get("prompt", "")).strip()
    context = data.get("context")

    if not prompt:
        return jsonify({"error": "Missing prompt parameter"}), 400
    if context is not None and not isinstance(context, str):
        return jsonify({"error": "context must be a string"}), 400

    try:
        response = dredge_reason(prompt, context=context)
    except NebiusConfigurationError as exc:
        return jsonify({"error": str(exc), "provider": "Nebius Token Factory"}), 503
    except Exception as exc:
        app.logger.exception("Nebius inference failed")
        return jsonify({"error": "Nebius inference failed", "detail": str(exc)}), 502

    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return jsonify({
        "provider": "Nebius Token Factory",
        "model": response.get("model", get_nebius_config()["model"]),
        "answer": message.get("content", ""),
        "usage": response.get("usage", {}),
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found", "status": 404}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({"error": "Server error", "status": 500}), 500


@app.route("/api/gordon/capabilities")
def gordon_capabilities():
    return jsonify({
        "gordon_version": "1.0.0",
        "dredge_integration": "active",
        "capabilities": [
            {
                "name": "Security Analysis",
                "endpoint": "/api/dependabot/fibot/chat",
                "method": "POST",
                "description": "Analyze vulnerabilities with FiBot",
            },
            {
                "name": "Model Management",
                "endpoint": "/api/advanced/models/list",
                "method": "GET",
                "description": "List and manage AI models",
            },
            {
                "name": "String Theory",
                "endpoint": "/api/advanced/visualization/string-spectrum",
                "method": "POST",
                "description": "Compute string vibrational spectrum",
            },
            {
                "name": "Monitoring",
                "endpoint": "/api/advanced/containers/status",
                "method": "GET",
                "description": "Get system monitoring metrics",
            },
            {
                "name": "Recommendations",
                "endpoint": "/api/dependabot/recommendations",
                "method": "GET",
                "description": "Get FiBot recommendations",
            },
            {
                "name": "Nemotron Reasoning",
                "endpoint": "/api/dredge/nebius/reason",
                "method": "POST",
                "description": "Run DREDGE reasoning with NVIDIA Nemotron via Nebius Token Factory",
            },
        ],
    })


handler = app.wsgi_app
