# Docker Buildx Bake Configuration for DREDGE
# Usage: docker buildx bake -f docker-bake.hcl [target]

variable "REGISTRY" {
  default = "docker.io"
}

variable "NAMESPACE" {
  default = "dredge"
}

variable "VERSION" {
  default = "latest"
}

# Base image target (Python 3.12 + DREDGE CLI)
target "base" {
  dockerfile = "Dockerfile"
  target     = "base"
  tags = [
	"${REGISTRY}/${NAMESPACE}/dredge-base:${VERSION}",
	"${REGISTRY}/${NAMESPACE}/dredge-base:latest"
  ]
  args = {
	BUILDKIT_INLINE_CACHE = "1"
  }
}

# CPU build target (PyTorch CPU)
target "cpu-build" {
  dockerfile = "Dockerfile"
  target     = "cpu-build"
  tags = [
	"${REGISTRY}/${NAMESPACE}/dredge-cpu:${VERSION}",
	"${REGISTRY}/${NAMESPACE}/dredge-cpu:latest"
  ]
  depends_on = ["base"]
}

# GPU build target (Python 3.10 + PyTorch GPU)
target "gpu-build" {
  dockerfile = "Dockerfile"
  target     = "gpu-build"
  tags = [
	"${REGISTRY}/${NAMESPACE}/dredge-gpu:${VERSION}",
	"${REGISTRY}/${NAMESPACE}/dredge-gpu:latest"
  ]
}

# Development target (Swift + Testing tools)
target "dev" {
  dockerfile = "Dockerfile"
  target     = "dev"
  tags = [
	"${REGISTRY}/${NAMESPACE}/dredge-dev:${VERSION}",
	"${REGISTRY}/${NAMESPACE}/dredge-dev:latest"
  ]
  depends_on = ["base"]
}

# Production target (Minimal image)
target "prod" {
  dockerfile = "Dockerfile"
  target     = "prod"
  tags = [
	"${REGISTRY}/${NAMESPACE}/dredge:${VERSION}",
	"${REGISTRY}/${NAMESPACE}/dredge:latest"
  ]
  depends_on = ["base"]
}

# Group for local development (no push)
group "dev" {
  targets = ["prod", "dev", "cpu-build"]
}

# Group for CI/CD with multi-platform + provenance + SBOM
group "release" {
  targets = ["prod-release", "dev-release", "gpu-release"]
}

# Production release (multi-platform)
target "prod-release" {
  inherits = ["prod"]
  platforms = [
	"linux/amd64",
	"linux/arm64",
	"linux/arm/v7"
  ]
  provenance = "mode=max"
  sbom       = true
}

# Development release (multi-platform)
target "dev-release" {
  inherits = ["dev"]
  platforms = [
	"linux/amd64",
	"linux/arm64"
  ]
  provenance = "mode=max"
  sbom       = true
}

# GPU release (amd64 only)
target "gpu-release" {
  inherits = ["gpu-build"]
  platforms = [
	"linux/amd64"
  ]
  provenance = "mode=max"
  sbom       = true
}
