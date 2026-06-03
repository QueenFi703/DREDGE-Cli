#!/bin/bash
# Start Interactive DREDGE Environment

set -e

echo "🚀 Starting Interactive DREDGE Environment..."

# Check if .dockerignore exists and optimize it
if [ ! -f .dockerignore ]; then
	echo "⚠️  Creating .dockerignore to optimize Docker build context"
	cat > .dockerignore << 'EOF'
.git
.gitignore
.vs
.vscode
.idea
.DS_Store
*.swp
*.swo
*~
__pycache__
*.pyc
.pytest_cache
.coverage
htmlcov
.tox
dist
build
*.egg-info
node_modules
.yarn
.next
.nuxt
coverage
.env.local
.env.*.local
venv
env
ENV
docs
examples
tests
EOF
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q -r requirements-interactive.txt

# Check for dredge-dev Docker image
if ! docker image inspect dredge-dev:latest > /dev/null 2>&1; then
	echo "🐳 Building dredge-dev Docker image..."
	docker build --target dev -t dredge-dev:latest .
fi

# Start services
echo "🎯 Starting services with docker-compose..."
docker-compose -f docker-compose.interactive.yml up -d

# Wait for services
echo "⏳ Waiting for services to start..."
sleep 3

# Display status
echo ""
echo "✅ Interactive DREDGE is running!"
echo ""
echo "📊 Services:"
echo "  - API Server:    http://localhost:8000"
echo "  - Web UI:        http://localhost:8000"
echo "  - Database:      localhost:5432"
echo "  - Redis Cache:   localhost:6379"
echo ""
echo "🔍 View logs:"
echo "  docker-compose -f docker-compose.interactive.yml logs -f"
echo ""
echo "🛑 Stop services:"
echo "  docker-compose -f docker-compose.interactive.yml down"
echo ""
