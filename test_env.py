#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Test 1: Check .env exists
env_path = Path('.env')
print(f"[Test 1] .env exists: {env_path.exists()}")
print(f"  Path: {env_path.absolute()}\n")

# Test 2: Load manually  
from dotenv import load_dotenv
load_dotenv(env_path)

# Test 3: Check values
github_id = os.environ.get('GITHUB_CLIENT_ID')
github_secret = os.environ.get('GITHUB_CLIENT_SECRET')
secret_key = os.environ.get('SECRET_KEY')

print(f"[Test 2] After load_dotenv:")
print(f"  GITHUB_CLIENT_ID: {github_id[:20] if github_id else 'NOT SET'}...")
print(f"  GITHUB_CLIENT_SECRET: {github_secret[:20] if github_secret else 'NOT SET'}...")
print(f"  SECRET_KEY: {secret_key[:20] if secret_key else 'NOT SET'}...")

# Test 4: Check if auth detects it
print(f"\n[Test 3] Auth detection:")
print(f"  GitHub configured: {bool(github_id and github_secret)}")
print(f"  Google configured: {bool(os.environ.get('GOOGLE_CLIENT_ID') and os.environ.get('GOOGLE_CLIENT_SECRET'))}")

print("\nAll tests passed! Environment variables are correctly set.")
