#!/bin/bash
# Production deployment script for Railway

set -e

echo "=================================="
echo "DREDGE Auth Gateway - Production Deployment"
echo "=================================="

# Check prerequisites
echo "✓ Checking prerequisites..."
command -v railway >/dev/null 2>&1 || { echo "Railway CLI required"; exit 1; }
command -v git >/dev/null 2>&1 || { echo "Git required"; exit 1; }

# Get current branch
BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "✓ Current branch: $BRANCH"

# Ensure clean working directory
if [[ ! -z $(git status -s) ]]; then
    echo "✓ Committing changes..."
    git add -A
    git commit -m "chore: Production deployment $(date +%Y-%m-%d)"
fi

# Push to GitHub
echo "✓ Pushing to GitHub..."
git push origin $BRANCH

# Login to Railway
echo "✓ Authenticating with Railway..."
railway login

# Set up Railway project (if not already done)
echo "✓ Initializing Railway project..."
railway init --name dredge-auth-gateway || true

# Set environment variables in Railway
echo "✓ Setting environment variables..."
railway variables set PYTHONUNBUFFERED true
railway variables set FLASK_ENV production
railway variables set AUTH_STORAGE_PATH /data/api_keys.json
railway variables set LOG_LEVEL INFO
railway variables set ENABLE_RATE_LIMITING true

# Deploy
echo "✓ Deploying to Railway..."
railway up --build

echo ""
echo "=================================="
echo "Deployment Complete!"
echo "=================================="
echo ""
echo "Your application is now deployed!"
echo ""
echo "Get your URL:"
echo "  railway domains"
echo ""
echo "View logs:"
echo "  railway logs"
echo ""
echo "View status:"
echo "  railway status"
echo ""

# Get the deployed URL
DEPLOYED_URL=$(railway domains 2>/dev/null | head -1 || echo "https://your-railway-url")

echo "Access your gateway at:"
echo "  $DEPLOYED_URL"
echo ""
echo "API Documentation:"
echo "  $DEPLOYED_URL/docs"
echo ""
echo "Health Check:"
echo "  curl $DEPLOYED_URL/health"
echo ""
echo "Test Protected Endpoint:"
echo "  curl -H 'x-api-key: YOUR_KEY' $DEPLOYED_URL/orion/health"
echo ""
