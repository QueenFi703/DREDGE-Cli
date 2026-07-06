#!/bin/bash
# DREDGE OAuth Configuration Setup
# Set up GitHub and Google OAuth credentials

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║              DREDGE OAuth Configuration Setup                             ║"
echo "║              GitHub + Google Login Configuration                          ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Required Variables
REDIRECT_URL="${OAUTH_REDIRECT_BASE:-http://localhost:3000}"
SECRET_KEY="${SECRET_KEY:-}"

echo "Current Configuration:"
echo "  Redirect URL: $REDIRECT_URL"
echo "  Secret Key:   ${SECRET_KEY:0:10}... ($([ -z "$SECRET_KEY" ] && echo "NOT SET" || echo "SET"))"
echo ""

# ============================================================================
# GitHub Setup Instructions
# ============================================================================

echo "═══════════════════════════════════════════════════════════════════════════"
echo "STEP 1: CONFIGURE GITHUB OAUTH"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""
echo "1. Go to: https://github.com/settings/developers"
echo "2. Click 'New OAuth App'"
echo "3. Fill in the form:"
echo "   Application name:    DREDGE Studio"
echo "   Homepage URL:        $REDIRECT_URL"
echo "   Authorization callback URL: $REDIRECT_URL/auth/github/callback"
echo ""
echo "4. Copy the Client ID and Client Secret"
echo "5. Set environment variables:"
echo ""
echo "   export GITHUB_CLIENT_ID='<your-client-id>'"
echo "   export GITHUB_CLIENT_SECRET='<your-client-secret>'"
echo ""

# ============================================================================
# Google Setup Instructions
# ============================================================================

echo "═══════════════════════════════════════════════════════════════════════════"
echo "STEP 2: CONFIGURE GOOGLE OAUTH"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""
echo "1. Go to: https://console.developers.google.com/"
echo "2. Create a new project or select existing"
echo "3. Enable 'Google+ API'"
echo "4. Go to 'Credentials' → 'Create Credentials' → 'OAuth 2.0 Client ID'"
echo "5. Choose 'Web Application'"
echo "6. Add Authorized JavaScript origins:"
echo "   $REDIRECT_URL"
echo ""
echo "7. Add Authorized redirect URIs:"
echo "   $REDIRECT_URL/auth/google/callback"
echo ""
echo "8. Copy the Client ID and Client Secret"
echo "9. Set environment variables:"
echo ""
echo "   export GOOGLE_CLIENT_ID='<your-client-id>'"
echo "   export GOOGLE_CLIENT_SECRET='<your-client-secret>'"
echo ""

# ============================================================================
# Flask Secret Key
# ============================================================================

echo "═══════════════════════════════════════════════════════════════════════════"
echo "STEP 3: SET FLASK SECRET KEY"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""
echo "Generate a secure secret key:"
echo "   python -c \"import secrets; print(secrets.token_hex(32))\""
echo ""
echo "Set the environment variable:"
echo "   export SECRET_KEY='<generated-secret-key>'"
echo ""

# ============================================================================
# Environment Setup
# ============================================================================

echo "═══════════════════════════════════════════════════════════════════════════"
echo "STEP 4: SET ALL ENVIRONMENT VARIABLES"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""
echo "Option A: Export in terminal (temporary)"
echo "  export FLASK_ENV=production"
echo "  export SECRET_KEY='your-secret-key'"
echo "  export GITHUB_CLIENT_ID='your-github-id'"
echo "  export GITHUB_CLIENT_SECRET='your-github-secret'"
echo "  export GOOGLE_CLIENT_ID='your-google-id'"
echo "  export GOOGLE_CLIENT_SECRET='your-google-secret'"
echo "  export OAUTH_REDIRECT_BASE='$REDIRECT_URL'"
echo ""
echo "Option B: Create .env file (persistent)"
echo "  Create .env in dredge-cli-repo/ with:"
echo ""

cat > /tmp/env_template.txt << 'EOF'
# DREDGE OAuth Configuration
FLASK_ENV=production
SECRET_KEY=<your-secret-key-here>

# GitHub OAuth
GITHUB_CLIENT_ID=<your-github-client-id>
GITHUB_CLIENT_SECRET=<your-github-client-secret>

# Google OAuth
GOOGLE_CLIENT_ID=<your-google-client-id>
GOOGLE_CLIENT_SECRET=<your-google-client-secret>

# OAuth Redirect Base
OAUTH_REDIRECT_BASE=http://localhost:3000
EOF

cat /tmp/env_template.txt

echo ""
echo "  Then run:"
echo "    export $(cat .env | xargs)"
echo "    python -m dredge.server"
echo ""

# ============================================================================
# Verify Configuration
# ============================================================================

echo "═══════════════════════════════════════════════════════════════════════════"
echo "STEP 5: VERIFY CONFIGURATION"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""
echo "Check that all variables are set:"
echo ""
echo "  echo 'SECRET_KEY:' \$SECRET_KEY"
echo "  echo 'GITHUB_CLIENT_ID:' \$GITHUB_CLIENT_ID"
echo "  echo 'GITHUB_CLIENT_SECRET:' \$GITHUB_CLIENT_SECRET"
echo "  echo 'GOOGLE_CLIENT_ID:' \$GOOGLE_CLIENT_ID"
echo "  echo 'GOOGLE_CLIENT_SECRET:' \$GOOGLE_CLIENT_SECRET"
echo ""

# ============================================================================
# Start Server
# ============================================================================

echo "═══════════════════════════════════════════════════════════════════════════"
echo "STEP 6: START DREDGE SERVER"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""
echo "Start the DREDGE service:"
echo "  cd dredge-cli-repo"
echo "  python -m dredge.server --host 0.0.0.0 --port 3000"
echo ""
echo "Expected output:"
echo "  Google OAuth provider registered."
echo "  GitHub OAuth provider registered."
echo "  Starting DREDGE x Dolly server on http://0.0.0.0:3000"
echo ""

# ============================================================================
# Test Login
# ============================================================================

echo "═══════════════════════════════════════════════════════════════════════════"
echo "STEP 7: TEST LOGIN"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""
echo "1. Open in browser:"
echo "   http://localhost:3000/auth/login"
echo ""
echo "2. You should see:"
echo "   - Sign in with Google button"
echo "   - Sign in with GitHub button"
echo ""
echo "3. Click a button and complete OAuth flow"
echo "4. You should be redirected to dashboard after login"
echo ""

echo "═══════════════════════════════════════════════════════════════════════════"
echo "✅ Setup Complete"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""
