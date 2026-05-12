#!/usr/bin/env bash
# =============================================================================
# ORION QUICK START — Production Gateway Setup
# =============================================================================

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

echo "🚀 ORION Gateway Quick Start"
echo "═════════════════════════════════════════════════════════════════════"
echo ""

# Step 1: Check dependencies
echo "📦 Step 1: Checking dependencies..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Install from https://www.python.org/"
    exit 1
fi
echo "✅ Python 3 found: $(python3 --version)"

if ! command -v pip &> /dev/null; then
    echo "❌ pip not found."
    exit 1
fi
echo "✅ pip found"
echo ""

# Step 2: Install Python dependencies
echo "📦 Step 2: Installing dependencies..."
pip install fastapi uvicorn pydantic click
echo "✅ Dependencies installed"
echo ""

# Step 3: Create .env file
echo "⚙️  Step 3: Setting up environment..."
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    cat > "$PROJECT_ROOT/.env" << 'EOF'
# Orion Gateway Configuration
ORION_HOST=0.0.0.0
ORION_PORT=3001
ORION_DEBUG=true

# Placeholder for local development
DATABASE_URL=postgresql://user:password@localhost:5432/orion_db
REDIS_URL=redis://localhost:6379/0
STRIPE_SECRET_KEY=sk_test_placeholder
JWT_SECRET=dev-secret-key-change-in-production
EOF
    echo "✅ Created .env file"
else
    echo "ℹ️  .env file already exists"
fi
echo ""

# Step 4: Display test API keys
echo "🔑 Step 4: Available test API keys..."
echo ""
echo "   Free Tier (100 requests/month):"
echo "   $ export API_KEY='demo-free-key'"
echo ""
echo "   Pro Tier (10,000 requests/month):"
echo "   $ export API_KEY='demo-pro-key'"
echo ""
echo "   Enterprise Tier (unlimited):"
echo "   $ export API_KEY='demo-enterprise-key'"
echo ""

# Step 5: Show how to start
echo "═════════════════════════════════════════════════════════════════════"
echo ""
echo "🎉 Ready to go!"
echo ""
echo "To start the Orion Gateway API server, run:"
echo ""
echo "   python -m dredge.orion_gateway"
echo ""
echo "Or with options:"
echo ""
echo "   python -c \"from dredge.orion_gateway import run_orion; run_orion(debug=True)\""
echo ""
echo "═════════════════════════════════════════════════════════════════════"
echo ""
echo "📖 Next steps:"
echo ""
echo "1. Start the API server:"
echo "   python -m dredge.orion_gateway"
echo ""
echo "2. In another terminal, test it:"
echo "   curl -X POST http://localhost:3001/invoke \\"
echo "     -H 'x-api-key: demo-pro-key' \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"input\":\"Hello\",\"mode\":\"standard\"}'"
echo ""
echo "3. Check usage:"
echo "   curl http://localhost:3001/usage \\"
echo "     -H 'x-api-key: demo-pro-key'"
echo ""
echo "4. See documentation:"
echo "   cat docs/ORION_GATEWAY.md"
echo ""
echo "═════════════════════════════════════════════════════════════════════"
