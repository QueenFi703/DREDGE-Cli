# Start Interactive DREDGE Environment (Windows)

Write-Host "🚀 Starting Interactive DREDGE Environment..." -ForegroundColor Green

# Create .dockerignore if it doesn't exist
if (-not (Test-Path .dockerignore)) {
	Write-Host "⚠️  Creating .dockerignore to optimize Docker build context" -ForegroundColor Yellow
	@'
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
'@ | Out-File -Encoding utf8 .dockerignore
}

# Install dependencies
Write-Host "📦 Installing dependencies..." -ForegroundColor Cyan
pip install -q -r requirements-interactive.txt

# Check for dredge-dev Docker image
Write-Host "🐳 Checking for dredge-dev Docker image..." -ForegroundColor Cyan
$imageExists = docker image inspect dredge-dev:latest 2>$null
if ($null -eq $imageExists) {
	Write-Host "Building dredge-dev Docker image..." -ForegroundColor Yellow
	docker build --target dev -t dredge-dev:latest .
}

# Start services
Write-Host "🎯 Starting services with docker-compose..." -ForegroundColor Cyan
docker-compose -f docker-compose.interactive.yml up -d

# Wait for services
Write-Host "⏳ Waiting for services to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

# Display status
Write-Host ""
Write-Host "✅ Interactive DREDGE is running!" -ForegroundColor Green
Write-Host ""
Write-Host "📊 Services:" -ForegroundColor Cyan
Write-Host "  - API Server:    http://localhost:8000"
Write-Host "  - Web UI:        http://localhost:8000"
Write-Host "  - Database:      localhost:5432"
Write-Host "  - Redis Cache:   localhost:6379"
Write-Host ""
Write-Host "🔍 View logs:" -ForegroundColor Cyan
Write-Host "  docker-compose -f docker-compose.interactive.yml logs -f"
Write-Host ""
Write-Host "🛑 Stop services:" -ForegroundColor Cyan
Write-Host "  docker-compose -f docker-compose.interactive.yml down"
Write-Host ""

# Open in browser
Write-Host "Opening Web UI in browser..." -ForegroundColor Yellow
Start-Process "http://localhost:8000"
