#!/usr/bin/env python3
"""
DREDGE Server Launcher - Loads .env and starts server
"""

import os
import sys
from pathlib import Path

# Load .env from current directory
from dotenv import load_dotenv
env_path = Path('.env')
if env_path.exists():
    load_dotenv(env_path)
    print(f"[+] Environment loaded from {env_path.absolute()}")

# Verify OAuth credentials
github_id = os.environ.get('GITHUB_CLIENT_ID', '').strip()
github_secret = os.environ.get('GITHUB_CLIENT_SECRET', '').strip()
secret_key = os.environ.get('SECRET_KEY', '').strip()

print("\n[Environment Status]")
print(f"  SECRET_KEY: {'SET' if secret_key else 'NOT SET'}")
print(f"  GitHub OAuth: {'CONFIGURED' if github_id and github_secret else 'NOT CONFIGURED'}")
print(f"  Google OAuth: {'CONFIGURED' if os.environ.get('GOOGLE_CLIENT_ID') and os.environ.get('GOOGLE_CLIENT_SECRET') else 'NOT CONFIGURED'}")
print()

# Now import and run server
from dredge.server import run_server

# Parse arguments
host = os.environ.get('FLASK_HOST', '0.0.0.0')
port = int(os.environ.get('FLASK_PORT', 3000))
debug = False

if '--debug' in sys.argv:
    debug = True

print(f"[+] Starting DREDGE Server on {host}:{port}")
run_server(host=host, port=port, debug=debug)
