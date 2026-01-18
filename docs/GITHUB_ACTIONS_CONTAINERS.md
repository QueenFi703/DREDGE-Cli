# GitHub Actions Container Workflows

This document describes the automated container workflows for DREDGE-Cli.

## Overview

We have three main GitHub Actions workflows for container management:

1. **Docker Image CI/CD** (`docker-publish.yml`) - Builds and publishes images to GitHub Container Registry
2. **Docker Build Test** (`docker-test.yml`) - Validates Docker builds on pull requests
3. **Python CI** (`ci-python.yml`) - Runs Python tests and linting

## Docker Image CI/CD Workflow

### Triggers

The workflow runs automatically on:

- **Push to `main` branch** - Builds and publishes images tagged as `latest-{cpu,gpu,dev}`
- **Push to `develop` branch** - Builds and publishes images tagged as `develop-{cpu,gpu,dev}`
- **Version tags** (`v*`) - Builds and publishes images with semantic version tags
- **Manual trigger** - Can be triggered manually via GitHub UI

### What it Does

1. **Builds three image variants**:
   - `cpu-build` → `ghcr.io/queenfi703/dredge-cli:*-cpu`
   - `gpu-build` → `ghcr.io/queenfi703/dredge-cli:*-gpu`
   - `dev` → `ghcr.io/queenfi703/dredge-cli:*-dev`

2. **Multi-architecture support**:
   - CPU and dev images: `linux/amd64`, `linux/arm64`
   - GPU images: `linux/amd64` only (NVIDIA CUDA)

3. **Publishes to GitHub Container Registry** (ghcr.io)

4. **Runs security scanning** with Trivy

5. **Tests images** by starting containers and checking health endpoints

6. **Generates attestations** for supply chain security

### Image Tags

When you push to branches or create tags, images are automatically tagged:

| Git Action | Image Tags |
|------------|------------|
| Push to `main` | `latest-cpu`, `latest-gpu`, `latest-dev` |
| Push to `develop` | `develop-cpu`, `develop-gpu`, `develop-dev` |
| Tag `v1.2.3` | `v1.2.3-cpu`, `v1.2-cpu`, `v1-cpu` (and gpu, dev variants) |
| PR #42 | `pr-42-cpu`, `pr-42-gpu`, `pr-42-dev` (not pushed, build only) |

### Usage

#### Automatic Builds

Simply push to `main` or `develop`, or create a tag:

```bash
# Push to main (triggers build)
git checkout main
git push origin main

# Create a version tag (triggers build)
git tag v0.3.0
git push origin v0.3.0
```

#### Manual Trigger

1. Go to GitHub Actions tab
2. Select "Docker Image CI/CD" workflow
3. Click "Run workflow"
4. Select branch and run

#### Pull Published Images

After the workflow completes, images are available at:

```bash
# Pull latest CPU image
docker pull ghcr.io/queenfi703/dredge-cli:latest-cpu

# Pull specific version
docker pull ghcr.io/queenfi703/dredge-cli:v0.3.0-cpu

# Pull GPU image
docker pull ghcr.io/queenfi703/dredge-cli:latest-gpu

# Run directly
docker run -p 3001:3001 ghcr.io/queenfi703/dredge-cli:latest-cpu
```

## Docker Build Test Workflow

### Triggers

The workflow runs automatically on:

- **Pull requests** to `main` or `develop` branches
- Only when Docker-related files change:
  - `Dockerfile`
  - `docker-compose*.yml`
  - `.dockerignore`
  - `requirements.txt`
  - `pyproject.toml`
  - `src/**`
  - `.github/workflows/docker-*.yml`

### What it Does

1. **Validates Docker builds** without pushing to registry
2. **Tests images** by running containers and checking health
3. **Scans for vulnerabilities** with Trivy
4. **Validates Docker Compose** configurations
5. **Validates Kubernetes** manifests

### Usage

This workflow runs automatically on PRs. Check the Actions tab to see results.

To test locally before creating a PR:

```bash
# Run the container test script
./scripts/test-containers.sh

# Build images locally
docker build --target cpu-build -t dredge-cli:test-cpu .
docker build --target gpu-build -t dredge-cli:test-gpu .
docker build --target dev -t dredge-cli:test-dev .
```

## Permissions

### Required Repository Secrets

- `GITHUB_TOKEN` - Automatically provided by GitHub Actions
  - Used for pushing to GHCR
  - No manual setup required

### Required Permissions

The workflows require these permissions (already configured):

- `contents: read` - Read repository contents
- `packages: write` - Push to GitHub Container Registry
- `security-events: write` - Upload security scan results
- `id-token: write` - Generate attestations

### Making Images Public

By default, images pushed to GHCR are private. To make them public:

1. Go to https://github.com/users/QueenFi703/packages/container/dredge-cli/settings
2. Scroll to "Danger Zone"
3. Click "Change visibility"
4. Select "Public"
5. Confirm

## Caching

The workflows use GitHub Actions cache to speed up builds:

- Docker layer caching via `type=gha`
- Separate cache scopes for each target (`cpu-build`, `gpu-build`, `dev`)
- Caches are automatically cleaned up after 7 days of inactivity

## Security Scanning

### Trivy Scanner

All images are scanned with Trivy for:
- OS vulnerabilities
- Language-specific vulnerabilities (Python)
- Critical and high severity issues

Results are:
- Displayed in workflow logs
- Uploaded to GitHub Security tab (on push to main/develop)

### Viewing Security Results

1. Go to repository "Security" tab
2. Click "Code scanning"
3. View Trivy findings

## Troubleshooting

### Build Failures

If builds fail, check:

1. **Workflow logs** - Click on failed job for details
2. **Docker build logs** - Look for pip install errors, missing files
3. **Test locally** - Run `docker build --target cpu-build .`

Common issues:
- Missing files in `.dockerignore`
- Dependency conflicts in `requirements.txt`
- Network timeouts (temporary, re-run workflow)

### Image Pull Failures

If you can't pull images:

1. **Authenticate with GHCR**:
```bash
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin
```

2. **Check image visibility** - May be private by default

3. **Check image exists** - Visit https://github.com/QueenFi703/DREDGE-Cli/pkgs/container/dredge-cli

### Test Failures

If image tests fail:

1. **Check health endpoint** - Container may start but fail health check
2. **Check logs** - `docker logs CONTAINER_NAME`
3. **Test locally** - Run container and curl health endpoint manually

## Best Practices

### For Contributors

1. **Test Docker changes locally** before creating PR:
   ```bash
   ./scripts/test-containers.sh
   ```

2. **Update documentation** if adding new container features

3. **Check workflow results** - Ensure all checks pass before merging

### For Maintainers

1. **Review security scan results** before releasing

2. **Tag releases properly**:
   ```bash
   git tag -a v0.3.0 -m "Release v0.3.0"
   git push origin v0.3.0
   ```

3. **Monitor image sizes** - Keep images lean

4. **Clean up old images** - Remove unused tags from GHCR

## Monitoring

### View Workflow Status

- **All workflows**: https://github.com/QueenFi703/DREDGE-Cli/actions
- **Specific workflow**: Click on workflow name
- **Specific run**: Click on run to see logs

### View Published Images

- **All packages**: https://github.com/QueenFi703?tab=packages
- **DREDGE-Cli images**: https://github.com/users/QueenFi703/packages/container/package/dredge-cli

### Workflow Badges

Add to README.md:

```markdown
[![Docker Image CI/CD](https://github.com/QueenFi703/DREDGE-Cli/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/QueenFi703/DREDGE-Cli/actions/workflows/docker-publish.yml)
```

## Advanced Usage

### Custom Build Arguments

To pass build arguments, modify the workflow:

```yaml
- name: Build and push Docker image
  uses: docker/build-push-action@v5
  with:
    build-args: |
      PYTHON_VERSION=3.11
      TORCH_VERSION=2.1.0
```

### Multi-stage Build Optimization

The Dockerfile uses multi-stage builds:
- `base` - Common Python environment
- `cpu-build` - CPU-only build
- `gpu-build` - GPU-enabled build
- `dev` - Development tools

### Platform-Specific Builds

To add more platforms:

```yaml
platforms: linux/amd64,linux/arm64,linux/arm/v7
```

Note: GPU images are amd64-only due to NVIDIA CUDA requirements.

## Resources

- [GitHub Container Registry docs](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)
- [Docker build-push-action](https://github.com/docker/build-push-action)
- [Trivy vulnerability scanner](https://github.com/aquasecurity/trivy)
- [Docker best practices](https://docs.docker.com/develop/dev-best-practices/)
