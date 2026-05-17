#!/usr/bin/env bash
# =============================================================================
# ORION QUICK START — Production Gateway Setup
# =============================================================================

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
VENV_DIR="$PROJECT_ROOT/.venv"

_echo_pkg_hint() {
    echo ""
    echo "💡 OS-specific Python/pip setup hints:"
    case "$(uname -s)" in
        Darwin)
            echo "   macOS: brew install python"
            ;;
        Linux)
            if command -v apt-get &> /dev/null; then
                echo "   Debian/Ubuntu: sudo apt-get update && sudo apt-get install -y python3 python3-pip python3-venv"
            elif command -v dnf &> /dev/null; then
                echo "   Fedora/RHEL: sudo dnf install -y python3 python3-pip"
            elif command -v yum &> /dev/null; then
                echo "   RHEL/CentOS: sudo yum install -y python3 python3-pip"
            elif command -v pacman &> /dev/null; then
                echo "   Arch: sudo pacman -S python python-pip"
            else
                echo "   Linux: install python3, pip, and venv via your distro package manager"
            fi
            ;;
        MINGW*|MSYS*|CYGWIN*)
            echo "   Windows: install Python from https://www.python.org/downloads/windows/"
            ;;
        *)
            echo "   Install Python 3 + pip using your operating system package manager"
            ;;
    esac
}

echo "🚀 ORION Gateway Quick Start"
echo "═════════════════════════════════════════════════════════════════════"
echo ""

# Step 1: Check dependencies
echo "📦 Step 1: Checking dependencies..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found."
    _echo_pkg_hint
    exit 1
fi
echo "✅ Python 3 found: $(python3 --version)"

if ! python3 -m pip --version &> /dev/null; then
    echo "❌ pip for Python 3 not found."
    _echo_pkg_hint
    exit 1
fi
echo "✅ pip found: $(python3 -m pip --version | awk '{print $1, $2}')"
echo ""

# Step 2: Create and activate virtualenv
echo "📦 Step 2: Setting up virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "✅ Created virtual environment at $VENV_DIR"
else
    echo "ℹ️  Virtual environment already exists at $VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo "✅ Virtual environment activated"
echo ""

# Step 3: Install Python dependencies into virtualenv
echo "📦 Step 3: Installing dependencies into virtual environment..."
python3 -m pip install --upgrade pip
python3 -m pip install fastapi uvicorn pydantic click
echo "✅ Dependencies installed"
echo ""

# Step 4: Create .env file
echo "⚙️  Step 4: Setting up environment..."
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    cat > "$PROJECT_ROOT/.env" << 'ENVEOF'
# Orion Gateway Configuration
ORION_HOST=0.0.0.0
ORION_PORT=3001
ORION_DEBUG=true

# Placeholder for local development
DATABASE_URL=postgresql://user:password@localhost:5432/orion_db
REDIS_URL=redis://localhost:6379/0
STRIPE_SECRET_KEY=sk_test_placeholder
JWT_SECRET=dev-secret-key-change-in-production
ENVEOF
    echo "✅ Created .env file"
else
    echo "ℹ️  .env file already exists"
fi
echo ""

# Step 5: Display test API keys
echo "🔑 Step 5: Available test API keys..."
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

# Step 6: Show how to start
echo "═════════════════════════════════════════════════════════════════════"
echo ""
echo "🎉 Ready to go!"
echo ""
echo "To start the Orion Gateway API server, run:"
echo ""
echo "   source .venv/bin/activate"
echo "   python -m dredge.orion_gateway"
echo ""
echo "Or with options:"
echo ""
echo "   source .venv/bin/activate"
echo "   python -c \"from dredge.orion_gateway import run_orion; run_orion(debug=True)\""
echo ""
echo "═════════════════════════════════════════════════════════════════════"
echo ""
echo "📖 Next steps:"
echo ""
echo "1. Start the API server:"
echo "   source .venv/bin/activate"
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
