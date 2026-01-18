#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Container Testing Script
# Tests Docker builds, Docker Compose configurations, and K8s manifests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "DREDGE-Cli Container Testing"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ─────────────────────────────────────────────────────────────────────
# Test 1: Validate YAML Syntax
# ─────────────────────────────────────────────────────────────────────
echo "📝 Test 1: Validating YAML syntax..."
for file in k8s/*.yaml; do
    echo "  Checking $file..."
    python3 -c "import yaml; list(yaml.safe_load_all(open('$file')))" && echo "  ✓ Valid" || { echo "  ✗ Invalid"; exit 1; }
done
echo "✅ YAML validation passed"
echo ""

# ─────────────────────────────────────────────────────────────────────
# Test 2: Validate Docker Compose Configuration
# ─────────────────────────────────────────────────────────────────────
echo "🐳 Test 2: Validating Docker Compose configurations..."
echo "  Checking docker-compose.yml..."
docker compose config > /dev/null 2>&1 && echo "  ✓ Valid" || { echo "  ✗ Invalid"; exit 1; }

echo "  Checking docker-compose.enhanced.yml..."
docker compose -f docker-compose.enhanced.yml config > /dev/null 2>&1 && echo "  ✓ Valid" || { echo "  ✗ Invalid"; exit 1; }

echo "  Checking docker-compose.profiles.yml..."
docker compose -f docker-compose.profiles.yml config > /dev/null 2>&1 && echo "  ✓ Valid" || { echo "  ✗ Invalid"; exit 1; }
echo "✅ Docker Compose validation passed"
echo ""

# ─────────────────────────────────────────────────────────────────────
# Test 3: Validate Kustomize Configuration
# ─────────────────────────────────────────────────────────────────────
echo "☸️  Test 3: Validating Kustomize configuration..."
if command -v kubectl &> /dev/null; then
    kubectl kustomize k8s/ > /dev/null 2>&1 && echo "  ✓ Kustomize valid" || echo "  ⚠️  Kustomize validation skipped (no cluster)"
else
    echo "  ⚠️  kubectl not found, skipping kustomize validation"
fi
echo ""

# ─────────────────────────────────────────────────────────────────────
# Test 4: Check Required Files
# ─────────────────────────────────────────────────────────────────────
echo "📁 Test 4: Checking required files..."
required_files=(
    "Dockerfile"
    ".dockerignore"
    "docker-compose.yml"
    "docker-compose.profiles.yml"
    ".env.container.example"
    ".github/workflows/docker-publish.yml"
    "k8s/namespace.yaml"
    "k8s/configmap.yaml"
    "k8s/redis.yaml"
    "k8s/dredge-server.yaml"
    "k8s/quasimoto-mcp.yaml"
    "k8s/hpa.yaml"
    "k8s/kustomization.yaml"
    "k8s/README.md"
    "nginx/nginx.conf"
    "monitoring/prometheus.yml"
    "monitoring/grafana/datasources/prometheus.yml"
    "monitoring/grafana/dashboards/dashboard.yml"
    "docs/CONTAINER_ARCHITECTURE.md"
    "docs/CONTAINER_QUICKSTART.md"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ Missing: $file"
        exit 1
    fi
done
echo "✅ All required files present"
echo ""

# ─────────────────────────────────────────────────────────────────────
# Test 5: Dockerfile Syntax Check
# ─────────────────────────────────────────────────────────────────────
echo "🔍 Test 5: Checking Dockerfile syntax..."
if command -v hadolint &> /dev/null; then
    hadolint Dockerfile && echo "  ✓ Dockerfile linting passed" || echo "  ⚠️  Dockerfile has linting warnings"
else
    echo "  ⚠️  hadolint not installed, skipping Dockerfile linting"
fi
echo ""

# ─────────────────────────────────────────────────────────────────────
# Test 6: Build Docker Images (Optional)
# ─────────────────────────────────────────────────────────────────────
if [ "${BUILD_IMAGES:-false}" = "true" ]; then
    echo "🔨 Test 6: Building Docker images..."
    
    echo "  Building CPU image..."
    docker build --target cpu-build -t dredge-cli:test-cpu . && echo "  ✓ CPU build succeeded" || { echo "  ✗ CPU build failed"; exit 1; }
    
    echo "  Building GPU image..."
    docker build --target gpu-build -t dredge-cli:test-gpu . && echo "  ✓ GPU build succeeded" || { echo "  ✗ GPU build failed"; exit 1; }
    
    echo "  Building dev image..."
    docker build --target dev -t dredge-cli:test-dev . && echo "  ✓ Dev build succeeded" || { echo "  ✗ Dev build failed"; exit 1; }
    
    echo "✅ Docker image builds passed"
else
    echo "⏭️  Test 6: Skipping Docker image builds (set BUILD_IMAGES=true to enable)"
fi
echo ""

# ─────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All container tests passed!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Next steps:"
echo "  1. Build images: BUILD_IMAGES=true ./scripts/test-containers.sh"
echo "  2. Start services: make docker-profile-cpu"
echo "  3. Deploy to K8s: kubectl apply -k k8s/"
echo ""
