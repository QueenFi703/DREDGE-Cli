#!/usr/bin/env python
"""
DREDGE Studio Advanced - Standalone server on port 8000
Serves 10 advanced features with no authentication
"""
import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dredge-cli-repo', 'src'))

from flask import Flask, jsonify, request, send_file
from pathlib import Path
import json

app = Flask(__name__, static_folder='dredge-cli-repo/src/dredge/static')

# Register advanced features
from dredge.advanced_features import register_advanced_features
register_advanced_features(app)

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "version": "1.0.0"})

@app.route('/')
def index():
    return jsonify({
        "name": "DREDGE Studio Advanced",
        "version": "1.0.0",
        "features": [
            "Model Management",
            "MCP Operations",
            "Insight Lifting",
            "Swift Toolchain",
            "Dependabot Alerts",
            "API Tester",
            "Container Status",
            "String Visualization",
            "Wave Function Plotter",
            "Code Generation"
        ],
        "dashboard": "http://127.0.0.1:8000/advanced",
        "api_base": "http://127.0.0.1:8000/api/advanced"
    })

@app.route('/advanced')
def advanced_dashboard():
    """Serve advanced dashboard"""
    html_path = Path('dredge-cli-repo/src/dredge/static/advanced_dashboard.html')
    if html_path.exists():
        return send_file(str(html_path), mimetype='text/html')
    return jsonify({"error": "Dashboard not found"}), 404

if __name__ == '__main__':
    print("Starting DREDGE Studio Advanced on http://127.0.0.1:8000")
    print("Dashboard: http://127.0.0.1:8000/advanced")
    print("API: http://127.0.0.1:8000/api/advanced/")
    print("Press Ctrl+C to stop")
    app.run(host='127.0.0.1', port=8000, debug=True, use_reloader=False)
