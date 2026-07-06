╔════════════════════════════════════════════════════════════════════════════╗
║         DREDGE OAUTH CONFIGURATION - GITHUB + GOOGLE LOGIN SETUP             ║
║                          Complete Guide & Troubleshooting                    ║
╚════════════════════════════════════════════════════════════════════════════╝

✅ QUICK START
==============

For interactive setup, run:

    python setup_oauth.py

This will guide you through creating GitHub and Google OAuth apps.


📋 MANUAL SETUP
================

If you prefer to do it manually, follow these steps:


╔════════════════════════════════════════════════════════════════════════════╗
║  STEP 1: CREATE GITHUB OAUTH APP                                          ║
╚════════════════════════════════════════════════════════════════════════════╝

1.1 Navigate to GitHub Developer Settings
─────────────────────────────────────────

   URL: https://github.com/settings/developers
   
   Or go to:
   - GitHub.com → Settings (top right profile)
   - Developer settings → OAuth Apps


1.2 Create New OAuth App
────────────────────────

   Click "New OAuth App" button


1.3 Fill Application Details
────────────────────────────

   Application name:
      DREDGE Studio
   
   Homepage URL:
      http://localhost:3000
   
   Application description (optional):
      AI-powered security and development platform
   
   Authorization callback URL (IMPORTANT!):
      http://localhost:3000/auth/github/callback


1.4 Register Application
────────────────────────

   Click "Register application"


1.5 Copy Credentials
───────────────────

   You'll see two important values:
   
   a) Client ID
      - Copy this to clipboard
   
   b) Client Secret
      - Click "Generate a new client secret"
      - Copy to clipboard
      - ⚠️  KEEP THIS SECRET! Don't share or commit to git


1.6 Save for Later
─────────────────

   Keep these safe - you'll need them in the .env file


╔════════════════════════════════════════════════════════════════════════════╗
║  STEP 2: CREATE GOOGLE OAUTH CREDENTIALS                                   ║
╚════════════════════════════════════════════════════════════════════════════╝

2.1 Navigate to Google Cloud Console
──────────────────────────────────────

   URL: https://console.developers.google.com/


2.2 Create New Project
──────────────────────

   Click "Select a project" at top
   Click "NEW PROJECT"
   
   Project name: DREDGE Studio
   Click "Create"
   
   (Wait for project to be created)


2.3 Enable Google+ API
──────────────────────

   In the search box, type "Google+ API"
   Click "Google+ API" in results
   Click "Enable" button
   
   (This gives your app permission to access user profiles)


2.4 Create OAuth 2.0 Client
──────────────────────────

   Go to: APIs & Services → Credentials
   Click "Create Credentials" → "OAuth 2.0 Client ID"
   
   Choose: "Web Application"
   
   Name (optional): DREDGE Studio Web Client
   
   Click "Create"


2.5 Configure Authorized URIs (IMPORTANT!)
──────────────────────────────────────────

   After creating, add these:
   
   Authorized JavaScript origins:
      http://localhost:3000
   
   Authorized redirect URIs:
      http://localhost:3000/auth/google/callback


2.6 Copy Credentials
───────────────────

   You'll see:
   
   a) Client ID
      - Looks like: 123456789.apps.googleusercontent.com
      - Copy this
   
   b) Client Secret
      - Random string
      - Copy this
      - ⚠️  KEEP THIS SECRET!


╔════════════════════════════════════════════════════════════════════════════╗
║  STEP 3: CONFIGURE ENVIRONMENT VARIABLES                                    ║
╚════════════════════════════════════════════════════════════════════════════╝

3.1 Create .env File
───────────────────

   In dredge-cli-repo/ directory:
   
   cp .env.example .env


3.2 Edit .env with Your Credentials
──────────────────────────────────

   Open .env in a text editor and fill in:


   # Generate a secret key (run: python -c "import secrets; print(secrets.token_hex(32))")
   SECRET_KEY=<generated-key-here>
   
   # GitHub credentials from Step 1
   GITHUB_CLIENT_ID=<your-github-client-id>
   GITHUB_CLIENT_SECRET=<your-github-client-secret>
   
   # Google credentials from Step 2
   GOOGLE_CLIENT_ID=<your-google-client-id>
   GOOGLE_CLIENT_SECRET=<your-google-client-secret>
   
   # Keep these as-is for local development
   OAUTH_REDIRECT_BASE=http://localhost:3000
   FLASK_ENV=production


3.3 Example .env File
────────────────────

   FLASK_ENV=production
   SECRET_KEY=a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6
   GITHUB_CLIENT_ID=Ov23liIxxxxxxxxxx
   GITHUB_CLIENT_SECRET=c6d1fxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   GOOGLE_CLIENT_ID=123456789-xxxxxxxxxxxxxxxxx.apps.googleusercontent.com
   GOOGLE_CLIENT_SECRET=GOCSPX-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   OAUTH_REDIRECT_BASE=http://localhost:3000
   FLASK_HOST=0.0.0.0
   FLASK_PORT=3000


3.4 Generate SECRET_KEY
──────────────────────

   Run this to generate a secure key:
   
   python -c "import secrets; print(secrets.token_hex(32))"
   
   Copy the output and paste into .env as SECRET_KEY


╔════════════════════════════════════════════════════════════════════════════╗
║  STEP 4: START DREDGE SERVER                                               ║
╚════════════════════════════════════════════════════════════════════════════╝

4.1 Load Environment Variables
──────────────────────────────

   cd dredge-cli-repo
   
   export $(cat .env | xargs)


4.2 Start DREDGE
───────────────

   python -m dredge.server
   
   or
   
   python -m dredge.server --host 0.0.0.0 --port 3000


4.3 Expected Output
──────────────────

   You should see:
   
   ✅ Google OAuth provider registered.
   ✅ GitHub OAuth provider registered.
   Starting DREDGE x Dolly server on http://0.0.0.0:3000
   API Version: 2.0.0
   Dashboard: http://localhost:3000/advanced
   API Endpoints: http://localhost:3000/api/advanced/
   Sign in at: http://localhost:3000/auth/login
   Server ready. Press CTRL+C to stop.


╔════════════════════════════════════════════════════════════════════════════╗
║  STEP 5: TEST OAUTH LOGIN                                                  ║
╚════════════════════════════════════════════════════════════════════════════╝

5.1 Open Login Page
──────────────────

   URL: http://localhost:3000/auth/login
   
   You should see:
   - 🐙 "Sign in with GitHub" button
   - 🔵 "Sign in with Google" button


5.2 Test GitHub Login
────────────────────

   1. Click "Sign in with GitHub"
   2. GitHub will ask for authorization
   3. Click "Authorize [your-app-name]"
   4. You should be redirected back to DREDGE dashboard


5.3 Test Google Login
───────────────────

   1. Click "Sign in with Google"
   2. Select your Google account
   3. Click "Allow" to grant permissions
   4. You should be redirected back to DREDGE dashboard


5.4 Verify User Profile
──────────────────────

   After login, you should see your name and email in the dashboard

   Or check: http://localhost:3000/auth/me
   
   Should return JSON like:
   {
     "id": "github:12345",
     "name": "Your Name",
     "email": "your.email@example.com",
     "provider": "github",
     "avatar": "https://..."
   }


╔════════════════════════════════════════════════════════════════════════════╗
║  TROUBLESHOOTING                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

Problem: "OAuth provider not configured"
──────────────────────────────────────────

   Cause: Environment variables not set
   
   Solution:
   1. Check .env file has all credentials
   2. Load environment: export $(cat .env | xargs)
   3. Verify: echo $GITHUB_CLIENT_ID
   4. Restart server


Problem: "Redirect URI mismatch"
──────────────────────────────

   Cause: Callback URL doesn't match settings
   
   Solution:
   1. Go to GitHub Settings → OAuth Apps
   2. Check "Authorization callback URL" = http://localhost:3000/auth/github/callback
   3. Same for Google: check "Authorized redirect URIs"
   4. If using different port (e.g., 8000), update both GitHub/Google AND .env
   5. Restart server


Problem: "Invalid Client ID"
────────────────────────────

   Cause: Credentials are wrong or expired
   
   Solution:
   1. Double-check you copied the credentials correctly (no extra spaces)
   2. If still not working, regenerate credentials:
      - GitHub: regenerate "Client Secret"
      - Google: delete and create new OAuth app
   3. Update .env with new credentials
   4. Restart server


Problem: "Sessions not persisting between requests"
───────────────────────────────────────────────────

   Cause: SECRET_KEY not set properly
   
   Solution:
   1. Check SECRET_KEY is in .env (and not empty)
   2. Verify SECRET_KEY is at least 32 characters
   3. Make sure you ran: export $(cat .env | xargs)
   4. Restart server


Problem: "Login page shows but buttons do nothing"
──────────────────────────────────────────────────

   Cause: JavaScript issue or network problem
   
   Solution:
   1. Check browser console for errors (F12)
   2. Verify server is running (http://localhost:3000/health)
   3. Check firewall isn't blocking localhost:3000
   4. Try different browser or incognito mode
   5. Restart server


Problem: "Can't reach http://localhost:3000"
────────────────────────────────────────────

   Cause: Server not running or wrong port
   
   Solution:
   1. Check if server is running: ps aux | grep dredge
   2. Check if port 3000 is available: lsof -i :3000
   3. If occupied, use different port: python -m dredge.server --port 8000
   4. Update OAUTH_REDIRECT_BASE in .env and GitHub/Google settings


Problem: "SSL/TLS certificate errors in production"
──────────────────────────────────────────────────

   Cause: Using HTTP instead of HTTPS
   
   Solution:
   1. For production, ALWAYS use HTTPS
   2. Update OAUTH_REDIRECT_BASE: https://your-domain.com
   3. Update GitHub callback URL: https://your-domain.com/auth/github/callback
   4. Update Google redirect URI: https://your-domain.com/auth/google/callback
   5. Get SSL certificate (Let's Encrypt, etc.)
   6. Configure your web server (Nginx, Apache) with SSL


╔════════════════════════════════════════════════════════════════════════════╗
║  PRODUCTION DEPLOYMENT                                                      ║
╚════════════════════════════════════════════════════════════════════════════╝

For Railway, Vercel, or other platforms:

1. Create new OAuth apps for your production domain
2. Set environment variables in platform dashboard:
   - FLASK_ENV=production
   - SECRET_KEY=<your-secret>
   - GITHUB_CLIENT_ID=<prod-client-id>
   - GITHUB_CLIENT_SECRET=<prod-client-secret>
   - GOOGLE_CLIENT_ID=<prod-client-id>
   - GOOGLE_CLIENT_SECRET=<prod-client-secret>
   - OAUTH_REDIRECT_BASE=https://your-domain.com

3. Update GitHub OAuth app callback to:
   https://your-domain.com/auth/github/callback

4. Update Google OAuth app redirect URI to:
   https://your-domain.com/auth/google/callback

5. Deploy!


════════════════════════════════════════════════════════════════════════════════

                    ✅ OAUTH CONFIGURATION COMPLETE

            You now have GitHub and Google login working!

    Visit http://localhost:3000/auth/login to test the login page

════════════════════════════════════════════════════════════════════════════════
