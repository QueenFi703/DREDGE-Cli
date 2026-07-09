#!/usr/bin/env python
"""
DREDGE WSGI Application Entry Point

Exposes Flask app for production servers (Gunicorn, uWSGI, etc.)
Compatible with Railway, Vercel, Heroku, and other platforms
"""

import os
import sys
from pathlib import Path

# Add project root/src directory to Python path if needed
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"

if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

# Import your Flask application factory or app
from dredge.server import create_app

# Create the WSGI application
app = create_app()

# Configure for production
if __name__ == "__main__":
    # Direct execution (development)
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
