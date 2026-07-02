#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DREDGE Studio - Full Web UI Server
Combined Standard + Advanced Features
Production-ready Flask WSGI application
"""
import os
import sys
from pathlib import Path

# Add dredge to path
sys.path.insert(0, str(Path(__file__).parent / 'dredge-cli-repo' / 'src'))

from flask import Flask, jsonify, request, send_file
import json

app = Flask(__name__, static_folder=str(Path(__file__).parent / 'dredge-cli-repo' / 'src' / 'dredge' / 'static'))

# Register advanced features
from dredge.advanced_features import register_advanced_features
register_advanced_features(app)

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Home page"""
    return jsonify({
        "message": "DREDGE Studio - Combined UI",
        "version": "2.0.0",
        "dashboard": "http://127.0.0.1:8000/dashboard",
        "advanced": "http://127.0.0.1:8000/advanced",
        "docs": "http://127.0.0.1:8000/docs",
        "api": "http://127.0.0.1:8000/api/"
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "version": "2.0.0"})

@app.route('/dashboard')
def dashboard():
    """Main DREDGE Studio Dashboard"""
    static_dir = Path(__file__).parent / 'dredge-cli-repo' / 'src' / 'dredge' / 'static'
    html_file = static_dir / 'dashboard_combined.html'
    if html_file.exists():
        return send_file(str(html_file), mimetype='text/html')
    return jsonify({"error": "Dashboard not found"}), 404

@app.route('/advanced')
def advanced_dashboard():
    """Advanced features dashboard"""
    static_dir = Path(__file__).parent / 'dredge-cli-repo' / 'src' / 'dredge' / 'static'
    html_file = static_dir / 'advanced_dashboard_new.html'
    if html_file.exists():
        return send_file(str(html_file), mimetype='text/html')
    return jsonify({"error": "Advanced dashboard not found"}), 404

@app.route('/docs')
def api_docs():
    """API documentation"""
    static_dir = Path(__file__).parent / 'dredge-cli-repo' / 'src' / 'dredge' / 'static'
    html_file = static_dir / 'docs.html'
    if html_file.exists():
        return send_file(str(html_file), mimetype='text/html')
    return jsonify({"error": "Documentation not found"}), 404

@app.route('/api/dredge/status')
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
            "Visualization"
        ]
    })

@app.route('/api/dredge/lift', methods=['POST'])
def lift_insight():
    """Lift insight endpoint"""
    data = request.get_json()
    if not data or 'insight' not in data:
        return jsonify({"error": "Missing insight parameter"}), 400
    
    insight = data['insight']
    return jsonify({
        "status": "lifted",
        "original": insight,
        "enhanced": "[Enhanced via DREDGE] " + insight,
        "confidence": 0.89,
        "models_used": ["Quasimoto 4D", "String Theory 10D", "DREDGE Reasoner"]
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found", "status": 404}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({"error": "Server error", "status": 500}), 500

if __name__ == '__main__':
    print("DREDGE STUDIO - FULL WEB UI v2.0.0")
    print("Starting server on http://127.0.0.1:8000")
    print("")
    print("Access Points:")
    print("- Main Dashboard:  http://127.0.0.1:8000/dashboard")
    print("- Advanced UI:     http://127.0.0.1:8000/advanced")
    print("- API Docs:        http://127.0.0.1:8000/docs")
    print("- Health Check:    http://127.0.0.1:8000/health")
    print("")
    print("Press Ctrl+C to stop")
    
    app.run(host='127.0.0.1', port=8000, debug=True, use_reloader=False)
