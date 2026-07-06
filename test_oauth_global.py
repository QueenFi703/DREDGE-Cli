#!/usr/bin/env python3
"""Test OAuth global variable persistence"""

from dotenv import load_dotenv
from pathlib import Path
import os

# Load .env
load_dotenv(Path('.env'))

print("\n[Test 1] Import auth module")
from src.dredge.auth import oauth as oauth_initial, init_auth

print(f"  oauth initial: {oauth_initial}")

print("\n[Test 2] Create Flask app and call init_auth")
from flask import Flask
from flask_login import LoginManager

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY')

login_manager = LoginManager()
login_manager.init_app(app)

init_auth(app)

print("\n[Test 3] Check auth module after init_auth")
from src.dredge import auth

print(f"  auth.oauth: {auth.oauth}")
print(f"  auth.oauth is oauth_initial: {auth.oauth is oauth_initial}")

if auth.oauth:
    print(f"  Has github: {hasattr(auth.oauth, 'github')}")
    if hasattr(auth.oauth, 'github'):
        print(f"  auth.oauth.github: {auth.oauth.github}")
    else:
        print(f"  ERROR: auth.oauth has no github attribute!")
else:
    print(f"  ERROR: auth.oauth is None!")

print("\n[Test 4] Check in github_login function")
print(f"  oauth module global: {auth.oauth}")

print("\n" + "="*80)
if auth.oauth and hasattr(auth.oauth, 'github'):
    print("SUCCESS: GitHub OAuth properly configured!")
else:
    print("FAILURE: GitHub OAuth not properly configured!")
print("="*80 + "\n")
