#!/usr/bin/env python3
"""Test OAuth routing directly"""

from dotenv import load_dotenv
from pathlib import Path
import os

# Load .env
load_dotenv(Path('.env'))

print("\n[Setup] Loading modules and initializing app")

from flask import Flask
from flask_login import LoginManager
from src.dredge.auth import init_auth, get_oauth

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY')

login_manager = LoginManager()
login_manager.init_app(app)

print("[Setup] Calling init_auth...")
init_auth(app)

print("\n[Test] After init_auth, checking get_oauth()")
oauth = get_oauth()
print(f"  oauth: {oauth}")
print(f"  oauth type: {type(oauth)}")
print(f"  Has github: {hasattr(oauth, 'github') if oauth else 'None'}")

if oauth and hasattr(oauth, 'github'):
    print(f"  oauth.github: {oauth.github}")

print("\n[Test] Simulating route call")
with app.test_request_context('/auth/github'):
    from src.dredge.auth import github_login
    print("  Calling github_login()...")
    try:
        result = github_login()
        print(f"  Result type: {type(result)}")
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  Exception: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
