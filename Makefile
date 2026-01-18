# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DREDGE-Cli Makefile
# Primary entrypoint for local development workflows
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

.PHONY: help install-all install-python install-dev build-swift run-swift \
	serve mcp test-all test-python test-swift lint-all lint-python lint-swift \
	format-all format-python format-swift docker-build-cpu docker-build-gpu \
	docker-up-cpu docker-up-gpu docker-down clean install-hooks health info \
	version-info config-init config-show

.DEFAULT_GOAL := help

# ─────────────────────────────────────────────────────────────────────
# Help
# ─────────────────────────────────────────────────────────────────────

help: ## Show this help message
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "DREDGE-Cli Makefile"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "Available targets:"
	@echo ""
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Quick start:"
	@echo "  make install-all   # Install Python + Swift"
	@echo "  make test-all      # Run all tests"
	@echo "  make serve         # Start DREDGE server (port 3001)"
	@echo "  make health        # Check system health"
	@echo ""

# ─────────────────────────────────────────────────────────────────────
# Installation
# ─────────────────────────────────────────────────────────────────────

install-all: install-python build-swift ## Install all dependencies (Python + Swift)
	@echo "✅ All dependencies installed"

install-python: ## Install Python dependencies
	@echo "📦 Installing Python dependencies..."
	pip install -r requirements.txt
	pip install -e .
	@echo "✅ Python dependencies installed"

install-dev: install-python ## Install Python dependencies with development tools
	@echo "🔧 Installing development tools..."
	pip install pytest pytest-cov black ruff mypy pre-commit
	@echo "✅ Development tools installed"

install-hooks: ## Install pre-commit hooks
	@echo "🪝 Installing pre-commit hooks..."
	pip install pre-commit
	pre-commit install
	@echo "✅ Pre-commit hooks installed"

# ─────────────────────────────────────────────────────────────────────
# Swift
# ─────────────────────────────────────────────────────────────────────

build-swift: ## Build Swift CLI
	@echo "🔨 Building Swift CLI..."
	swift build --configuration release
	@echo "✅ Swift CLI built"

run-swift: ## Run Swift CLI
	@echo "🚀 Running Swift CLI..."
	swift run dredge-cli

# ─────────────────────────────────────────────────────────────────────
# Servers
# ─────────────────────────────────────────────────────────────────────

serve: ## Start DREDGE x Dolly server (port 3001)
	@echo "🚀 Starting DREDGE server on port 3001..."
	dredge-cli serve --host 0.0.0.0 --port 3001 --debug

mcp: ## Start MCP server with Quasimoto (port 3002)
	@echo "🚀 Starting MCP server on port 3002..."
	dredge-cli mcp --host 0.0.0.0 --port 3002 --debug

# ─────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────

health: ## Run health check on system and dependencies
	@echo "🏥 Running health check..."
	dredge-cli health

info: ## Show system information
	@echo "ℹ️  System Information:"
	@echo ""
	dredge-cli info

version-info: ## Show detailed version information
	@echo "📋 Version Information:"
	@echo ""
	dredge-cli --version-info

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

config-init: ## Initialize default configuration file
	@echo "⚙️  Initializing configuration..."
	dredge-cli config init

config-show: ## Show current configuration
	@echo "⚙️  Current Configuration:"
	@echo ""
	dredge-cli config show

# ─────────────────────────────────────────────────────────────────────
# Testing
# ─────────────────────────────────────────────────────────────────────

test-all: test-python test-swift ## Run all tests (Python + Swift)
	@echo "✅ All tests passed"

test-python: ## Run Python tests with coverage
	@echo "🧪 Running Python tests..."
	pytest tests/ -v --cov=src/dredge --cov-report=term-missing

test-swift: ## Run Swift tests
	@echo "🧪 Running Swift tests..."
	swift test

# ─────────────────────────────────────────────────────────────────────
# Linting
# ─────────────────────────────────────────────────────────────────────

lint-all: lint-python lint-swift ## Run all linters (Python + Swift)
	@echo "✅ All linting passed"

lint-python: ## Run Python linters (ruff, mypy)
	@echo "🔍 Linting Python code..."
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports || true

lint-swift: ## Run Swift linters (if installed)
	@echo "🔍 Linting Swift code..."
	@command -v swiftlint >/dev/null 2>&1 && swiftlint || echo "⚠️  SwiftLint not installed, skipping..."

# ─────────────────────────────────────────────────────────────────────
# Formatting
# ─────────────────────────────────────────────────────────────────────

format-all: format-python format-swift ## Format all code (Python + Swift)
	@echo "✅ All code formatted"

format-python: ## Format Python code with black
	@echo "✨ Formatting Python code..."
	black src/ tests/

format-swift: ## Format Swift code (if swiftformat installed)
	@echo "✨ Formatting Swift code..."
	@command -v swiftformat >/dev/null 2>&1 && swiftformat swift/ || echo "⚠️  SwiftFormat not installed, skipping..."

# ─────────────────────────────────────────────────────────────────────
# Docker
# ─────────────────────────────────────────────────────────────────────

docker-build-cpu: ## Build CPU Docker image
	@echo "🐳 Building CPU Docker image..."
	docker compose build dredge-server

docker-build-gpu: ## Build GPU Docker image
	@echo "🐳 Building GPU Docker image..."
	docker compose build quasimoto-mcp

docker-up-cpu: ## Start CPU container (Flask server, port 3001)
	@echo "🐳 Starting CPU container..."
	docker compose up dredge-server

docker-up-gpu: ## Start GPU container (MCP server, port 3002)
	@echo "🐳 Starting GPU container..."
	docker compose up quasimoto-mcp

docker-down: ## Stop all Docker containers
	@echo "🐳 Stopping Docker containers..."
	docker compose down

# ─────────────────────────────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────────────────────────────

clean: ## Clean build artifacts
	@echo "🧹 Cleaning build artifacts..."
	rm -rf build/ dist/ *.egg-info/
	rm -rf .build/ .swiftpm/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✅ Clean complete"
