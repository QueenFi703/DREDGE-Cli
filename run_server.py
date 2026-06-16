#!/usr/bin/env python3
"""
DREDGE Server Startup with .env Loading
Loads environment variables from .env file and starts the server
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    print(f"[+] Loading environment from: {env_path}")
    load_dotenv(env_path)
    print("[+] Environment variables loaded")
else:
    print(f"[-] .env file not found at: {env_path}")

# Verify credentials are loaded
github_id = os.environ.get("GITHUB_CLIENT_ID", "")
github_secret = os.environ.get("GITHUB_CLIENT_SECRET", "")
google_id = os.environ.get("GOOGLE_CLIENT_ID", "")
google_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "")
secret_key = os.environ.get("SECRET_KEY", "")

print("\n[OAuth Configuration Status]")
print(f"  GitHub: {'CONFIGURED' if github_id and github_secret else 'NOT CONFIGURED'}")
print(f"  Google: {'CONFIGURED' if google_id and google_secret else 'NOT CONFIGURED'}")
print(f"  SECRET_KEY: {'SET' if secret_key else 'NOT SET'}")
print()

# Now import and run server
from dredge.server import run_server

# Parse arguments
host = "0.0.0.0"
port = 3000
debug = False

if len(sys.argv) > 1:
    if "--port" in sys.argv:
        idx = sys.argv.index("--port")
        if idx + 1 < len(sys.argv):
            port = int(sys.argv[idx + 1])
    
    if "--host" in sys.argv:
        idx = sys.argv.index("--host")
        if idx + 1 < len(sys.argv):
            host = sys.argv[idx + 1]
    
    if "--debug" in sys.argv:
        debug = True

# Run server
run_server(host=host, port=port, debug=debug)
