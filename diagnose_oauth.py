#!/usr/bin/env python3
"""
Diagnose GitHub OAuth Configuration
"""

import os
import sys
from pathlib import Path

# Load .env
from dotenv import load_dotenv
env_path = Path('.env')
load_dotenv(env_path)

print("\n" + "="*80)
print("GITHUB OAUTH DIAGNOSTIC REPORT")
print("="*80 + "\n")

# Check environment
print("[1] Environment Variables:")
github_id = os.environ.get('GITHUB_CLIENT_ID', '')
github_secret = os.environ.get('GITHUB_CLIENT_SECRET', '')
secret_key = os.environ.get('SECRET_KEY', '')

print(f"  GITHUB_CLIENT_ID: {github_id[:20] if github_id else 'NOT SET'}...")
print(f"  GITHUB_CLIENT_SECRET: {github_secret[:20] if github_secret else 'NOT SET'}...")
print(f"  SECRET_KEY: {secret_key[:20] if secret_key else 'NOT SET'}...\n")

# Check Flask app
print("[2] Flask Application Setup:")
from flask import Flask
app = Flask(__name__)
app.secret_key = secret_key

print(f"  Flask app created: OK")
print(f"  Secret key set: OK\n")

# Check Flask-Login
print("[3] Flask-Login Setup:")
from flask_login import LoginManager
login_manager = LoginManager()
login_manager.init_app(app)
print(f"  LoginManager initialized: OK\n")

# Check OAuth
print("[4] Authlib OAuth Setup:")
from authlib.integrations.flask_client import OAuth

oauth = OAuth(app)
print(f"  OAuth object created: OK")
print(f"  OAuth attributes before register: {[a for a in dir(oauth) if not a.startswith('_')]}\n")

# Register GitHub
print("[5] GitHub OAuth Registration:")
if github_id and github_secret:
    try:
        oauth.register(
            name="github",
            client_id=github_id,
            client_secret=github_secret,
            access_token_url="https://github.com/login/oauth/access_token",
            authorize_url="https://github.com/login/oauth/authorize",
            api_base_url="https://api.github.com/",
            client_kwargs={"scope": "user:email"},
        )
        print(f"  GitHub registered: OK")
        print(f"  OAuth has 'github': {hasattr(oauth, 'github')}")
        if hasattr(oauth, 'github'):
            print(f"  GitHub client: {oauth.github}")
        else:
            print(f"  ERROR: oauth.github attribute NOT found!")
            print(f"  Available attributes: {[a for a in dir(oauth) if not a.startswith('_')]}")
    except Exception as e:
        print(f"  ERROR registering GitHub: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"  SKIP: GitHub credentials not set")

print("\n" + "="*80)
print("END DIAGNOSTIC REPORT")
print("="*80 + "\n")
