#!/usr/bin/env python3
"""
DREDGE OAuth Setup - Configuration Helper (Non-Interactive Version)

This displays what needs to be configured for GitHub and Google OAuth.
"""

import os
import secrets
from pathlib import Path


def main():
    print("\n" + "="*80)
    print("  DREDGE OAUTH SETUP - CONFIGURATION GUIDE")
    print("="*80 + "\n")
    
    # Generate new SECRET_KEY
    secret_key = secrets.token_hex(32)
    
    print("STEP 0: SECRET KEY GENERATED")
    print("-" * 80)
    print(f"\nGenerated SECRET_KEY:\n  {secret_key}\n")
    
    print("\n" + "="*80)
    print("STEP 1: GITHUB OAUTH")
    print("="*80 + "\n")
    
    print("""
1. Visit: https://github.com/settings/developers
2. Click: "New OAuth App"
3. Fill in:
   - Application name: DREDGE Studio
   - Homepage URL: http://localhost:3000
   - Authorization callback URL: http://localhost:3000/auth/github/callback
4. Register and copy:
   - Client ID
   - Client Secret (click "Generate a new client secret")
""")
    
    print("\n" + "="*80)
    print("STEP 2: GOOGLE OAUTH")
    print("="*80 + "\n")
    
    print("""
1. Visit: https://console.developers.google.com/
2. Create new project: "DREDGE Studio"
3. Enable: "Google+ API"
4. Create OAuth 2.0 Client ID (Web Application):
   - Authorized JavaScript origins: http://localhost:3000
   - Authorized redirect URIs: http://localhost:3000/auth/google/callback
5. Copy:
   - Client ID (ends with .apps.googleusercontent.com)
   - Client Secret
""")
    
    print("\n" + "="*80)
    print("STEP 3: UPDATE .ENV FILE")
    print("="*80 + "\n")
    
    env_path = Path(".env")
    
    if env_path.exists():
        print(f"Current .env location: {env_path.absolute()}\n")
        print("Edit .env and add your credentials:\n")
    
    env_template = f"""# DREDGE Studio - OAuth Configuration

FLASK_ENV=production
SECRET_KEY={secret_key}

# Add your GitHub credentials here:
GITHUB_CLIENT_ID=<paste-your-github-client-id>
GITHUB_CLIENT_SECRET=<paste-your-github-client-secret>

# Add your Google credentials here:
GOOGLE_CLIENT_ID=<paste-your-google-client-id>
GOOGLE_CLIENT_SECRET=<paste-your-google-client-secret>

OAUTH_REDIRECT_BASE=http://localhost:3000
FLASK_HOST=0.0.0.0
FLASK_PORT=3000
"""
    
    print(env_template)
    
    print("\n" + "="*80)
    print("STEP 4: RESTART SERVER")
    print("="*80 + "\n")
    
    print("""
After updating .env:

1. Stop current service:
   stop_background_job job_1781560856_3

2. Start new service:
   python -m dredge.server

3. Visit: http://localhost:3000/auth/login

4. You should see OAuth buttons for GitHub and Google
""")
    
    print("\n" + "="*80)
    print("[+] OAuth Setup Guide Complete")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
