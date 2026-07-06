╔════════════════════════════════════════════════════════════════════════════╗
║              DREDGE OAUTH LOGIN CONFIGURATION GUIDE                         ║
║                    GitHub + Google Setup Instructions                       ║
╚════════════════════════════════════════════════════════════════════════════╝

✅ OAUTH SUPPORT AVAILABLE
===========================

DREDGE supports two OAuth providers:
  ✓ GitHub OAuth
  ✓ Google OAuth

Both are configured and ready to activate with credentials.


📋 STEP 1: GENERATE FLASK SECRET KEY
=====================================

Required for session management. Generate a secure key:

Command:
  python -c "import secrets; print(secrets.token_hex(32))"

Example output:
  a1b2c3d4e5f6... (64 character hex string)

Save this value - you'll need it in .env


🔑 STEP 2: CONFIGURE GITHUB OAUTH
==================================

2a. Create GitHub OAuth App
────────────────────────────

1. Go to: https://github.com/settings/developers
2. Click "New OAuth App"
3. Fill in the form:
   
   Application name:              DREDGE Studio
   Homepage URL:                  http://localhost:3000
   Application description:       AI-powered security & development platform
   Authorization callback URL:    http://localhost:3000/auth/github/callback

4. Click "Create OAuth App"
5. Copy the following:
   - Client ID
   - Client Secret (click "Generate a new client secret")

Keep these safe - add to .env file.

2b. Add to .env
─────────────

GITHUB_CLIENT_ID=<your-client-id>
GITHUB_CLIENT_SECRET=<your-client-secret>


🔑 STEP 3: CONFIGURE GOOGLE OAUTH
==================================

3a. Create Google OAuth Credentials
────────────────────────────────────

1. Go to: https://console.developers.google.com/
2. Create a new project or select existing:
   - Project name: DREDGE Studio
3. Enable the "Google+ API":
   - Go to "APIs & Services" → "Library"
   - Search for "Google+ API"
   - Click it → Click "Enable"

4. Create OAuth 2.0 Client:
   - Go to "APIs & Services" → "Credentials"
   - Click "Create Credentials" → "OAuth 2.0 Client ID"
   - Choose "Web Application"
   - Fill in details:
     Name: DREDGE Studio
   
5. Configure Authorized Redirect URIs:
   - Authorized JavaScript origins:
     http://localhost:3000
     https://your-domain.com (for production)
   
   - Authorized redirect URIs:
     http://localhost:3000/auth/google/callback
     https://your-domain.com/auth/google/callback (for production)

6. Click "Create"
7. Download JSON (optional) or copy:
   - Client ID
   - Client Secret

3b. Add to .env
──────────────

GOOGLE_CLIENT_ID=<your-client-id>.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=<your-client-secret>


📝 STEP 4: CREATE .env FILE
============================

Copy .env.example to .env:

  cp dredge-cli-repo/.env.example dredge-cli-repo/.env

Edit dredge-cli-repo/.env with your credentials:

  FLASK_ENV=production
  SECRET_KEY=a1b2c3d4e5f6... (from Step 1)
  GITHUB_CLIENT_ID=xxxxx
  GITHUB_CLIENT_SECRET=yyyyy
  GOOGLE_CLIENT_ID=zzzzz.apps.googleusercontent.com
  GOOGLE_CLIENT_SECRET=wwwww
  OAUTH_REDIRECT_BASE=http://localhost:3000


🚀 STEP 5: START SERVER WITH OAUTH
===================================

5a. Load Environment Variables
──────────────────────────────

Option A: Export from .env file
  cd dredge-cli-repo
  export $(cat .env | xargs)

Option B: Load with python-dotenv
  pip install python-dotenv
  # Then Python automatically loads .env

5b. Start DREDGE Server
──────────────────────

  python -m dredge.server --host 0.0.0.0 --port 3000

Expected output (if OAuth configured):
  Google OAuth provider registered.
  GitHub OAuth provider registered.
  Starting DREDGE x Dolly server on http://0.0.0.0:3000


✅ STEP 6: TEST OAUTH LOGIN
============================

6a. Access Login Page
────────────────────

Open in browser:
  http://localhost:3000/auth/login

You should see:
  - "Sign in with Google" button
  - "Sign in with GitHub" button

6b. Test GitHub Login
────────────────────

1. Click "Sign in with GitHub"
2. Authorize DREDGE Studio
3. You should be redirected back to dashboard
4. User profile should show GitHub info

6c. Test Google Login
────────────────────

1. Click "Sign in with Google"
2. Select your Google account
3. Grant permissions
4. You should be redirected back to dashboard
5. User profile should show Google info


🔐 STEP 7: VERIFY LOGIN WORKS
=============================

Check /auth/status endpoint:

  curl http://localhost:3000/auth/status

Unauthenticated response:
  {"authenticated": false}

After login, response should be:
  {
    "authenticated": true,
    "name": "Your Name",
    "email": "your.email@example.com",
    "provider": "github" or "google",
    "avatar": "https://..."
  }


🌐 STEP 8: PRODUCTION DEPLOYMENT
=================================

For Railway, Vercel, or other platforms:

1. Set environment variables in platform dashboard:
   FLASK_ENV=production
   SECRET_KEY=<your-secret>
   GITHUB_CLIENT_ID=<your-id>
   GITHUB_CLIENT_SECRET=<your-secret>
   GOOGLE_CLIENT_ID=<your-id>
   GOOGLE_CLIENT_SECRET=<your-secret>
   OAUTH_REDIRECT_BASE=https://your-domain.com

2. Update OAuth callback URLs in GitHub/Google:
   GitHub: https://your-domain.com/auth/github/callback
   Google: https://your-domain.com/auth/google/callback

3. Deploy DREDGE


🛡️ SECURITY BEST PRACTICES
============================

✓ DO:
  - Use unique, strong SECRET_KEY for each environment
  - Store secrets in environment variables (never in code)
  - Use HTTPS in production (OAuth requires secure transport)
  - Rotate secrets regularly
  - Use short-lived tokens
  - Implement CSRF protection

✗ DON'T:
  - Commit secrets to git
  - Use same secret for dev and prod
  - Share secrets publicly
  - Log sensitive information
  - Use weak SECRET_KEY


🔍 TROUBLESHOOTING
==================

"OAuth provider not configured"
  → Check environment variables are set
  → Verify SECRET_KEY is defined
  → Restart server after setting .env

"Redirect URI mismatch"
  → Verify callback URL matches in GitHub/Google settings
  → Check OAUTH_REDIRECT_BASE in .env
  → For local dev: use http://localhost:3000
  → For production: use https://your-domain.com

"401 Unauthorized"
  → Client ID/Secret may be wrong
  → Check they're pasted correctly (no extra spaces)
  → Regenerate if uncertain

"Session not persisting"
  → SECRET_KEY may not be set
  → Check FLASK_ENV=production
  → Restart server


📚 REFERENCES
=============

GitHub OAuth:       https://github.com/settings/developers
Google OAuth:       https://console.developers.google.com/
Authlib Docs:       https://docs.authlib.org/
Flask-Login Docs:   https://flask-login.readthedocs.io/


════════════════════════════════════════════════════════════════════════════════

                     ✅ OAUTH CONFIGURATION COMPLETE

         When environment variables are set, login will show both options:
              "Sign in with GitHub" | "Sign in with Google"

════════════════════════════════════════════════════════════════════════════════
