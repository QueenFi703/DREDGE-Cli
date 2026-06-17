#!/usr/bin/env python
"""
DREDGE WSGI Application Entry Point

Exposes Flask app for production servers (Gunicorn, uWSGI, etc.)
Compatible with Railway, Vercel, Heroku, and other platforms
"""

import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import and create app
from dredge.server import create_app

# Create Flask application
app = create_app()

# Configure for production
if __name__ != '__main__':
    # Running under WSGI server (Gunicorn, etc.)
    app.config['ENV'] = os.environ.get('FLASK_ENV', 'production')
    app.config['DEBUG'] = False
    app.config['TESTING'] = False


if __name__ == '__main__':
    # Direct execution (development)
    port = int(os.environ.get('PORT', 3001))
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print(f"Starting DREDGE on {host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug)
