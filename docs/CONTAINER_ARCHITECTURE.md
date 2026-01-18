# Container Architecture Guide

This document describes the comprehensive container architecture for DREDGE-Cli, including Docker, Kubernetes, and GitHub Container Registry integration.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Docker Setup](#docker-setup)
- [Docker Compose Profiles](#docker-compose-profiles)
- [Kubernetes Deployment](#kubernetes-deployment)
- [GitHub Container Registry](#github-container-registry)
- [Monitoring & Observability](#monitoring--observability)
- [Security Considerations](#security-considerations)
- [Production Deployment](#production-deployment)

## Architecture Overview

### Components

1. **DREDGE Server** - Flask-based API server (CPU-optimized)
   - Port: 3001
   - Target: `cpu-build`
   - Horizontally scalable

2. **Quasimoto MCP Server** - PyTorch-based MCP server (GPU-enabled)
   - Port: 3002
   - Target: `gpu-build`
   - CUDA 11.8 support

3. **Redis Cache** - Distributed caching layer
   - Port: 6379
   - Persistent storage with AOF

4. **Worker Nodes** - Background task processing
   - CPU-optimized
   - Horizontally scalable

5. **Nginx Reverse Proxy** - Load balancing and SSL termination
   - Ports: 80 (HTTP), 443 (HTTPS)
   - Rate limiting enabled

6. **Prometheus** - Metrics collection
   - Port: 9091

7. **Grafana** - Metrics visualization
   - Port: 3000

### Container Images

All images are published to GitHub Container Registry (ghcr.io):

- `ghcr.io/queenfi703/dredge-cli:latest-cpu` - CPU-only build
- `ghcr.io/queenfi703/dredge-cli:latest-gpu` - GPU-enabled build
- `ghcr.io/queenfi703/dredge-cli:latest-dev` - Development build with testing tools

Multi-architecture support:
- CPU images: `linux/amd64`, `linux/arm64`
- GPU images: `linux/amd64` (NVIDIA CUDA)

## Docker Setup

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- (For GPU) NVIDIA Docker runtime

### Basic Commands

```bash
# CPU-only deployment
make docker-up-cpu
docker-compose --profile cpu up -d

# GPU deployment
make docker-up-gpu
docker-compose --profile gpu up -d

# Full stack with monitoring
docker-compose --profile full up -d

# Stop services
docker-compose down
```

### Building Images

```bash
# Build CPU image
docker build --target cpu-build -t dredge-cli:cpu .

# Build GPU image
docker build --target gpu-build -t dredge-cli:gpu .

# Build dev image
docker build --target dev -t dredge-cli:dev .

# Build with buildx (multi-arch)
docker buildx build --platform linux/amd64,linux/arm64 \
  --target cpu-build -t dredge-cli:cpu .
```

## Docker Compose Profiles

The `docker-compose.profiles.yml` file provides multiple deployment profiles:

### Available Profiles

1. **cpu** - DREDGE server + Redis (CPU-only)
   ```bash
   docker-compose -f docker-compose.profiles.yml --profile cpu up -d
   ```

2. **gpu** - Quasimoto MCP + Redis (GPU-enabled)
   ```bash
   docker-compose -f docker-compose.profiles.yml --profile gpu up -d
   ```

3. **full** - Complete stack (all services)
   ```bash
   docker-compose -f docker-compose.profiles.yml --profile full up -d
   ```

4. **workers** - Background workers only
   ```bash
   docker-compose -f docker-compose.profiles.yml --profile workers up -d
   ```

5. **monitoring** - Prometheus + Grafana
   ```bash
   docker-compose -f docker-compose.profiles.yml --profile monitoring up -d
   ```

6. **proxy** - Nginx reverse proxy
   ```bash
   docker-compose -f docker-compose.profiles.yml --profile proxy up -d
   ```

### Environment Variables

Create a `.env` file:

```env
# Versions
VERSION=latest

# Ports
DREDGE_PORT=3001
MCP_PORT=3002
REDIS_PORT=6379
NGINX_HTTP_PORT=80
NGINX_HTTPS_PORT=443
PROMETHEUS_PORT=9091
GRAFANA_PORT=3000
METRICS_PORT=9090

# Configuration
FLASK_ENV=production
CACHE_ENABLED=true
METRICS_ENABLED=true
REDIS_MAX_MEMORY=512mb

# GPU
CUDA_VISIBLE_DEVICES=0
GPU_COUNT=1
DEVICE=auto

# Workers
WORKER_CONCURRENCY=4
WORKER_REPLICAS=2

# Monitoring
GRAFANA_PASSWORD=your-secure-password
```

### Health Checks

All services include health checks:

```bash
# Check service health
docker-compose ps

# View health check logs
docker inspect --format='{{json .State.Health}}' dredge-flask

# Manual health checks
curl http://localhost:3001/health
curl http://localhost:3002/
redis-cli -h localhost -p 6379 ping
```

## Kubernetes Deployment

See [k8s/README.md](k8s/README.md) for comprehensive Kubernetes documentation.

### Quick Start

```bash
# Deploy with kubectl
kubectl apply -f k8s/

# Deploy with kustomize
kubectl apply -k k8s/

# Check status
kubectl get pods -n dredge
kubectl get svc -n dredge
kubectl get ingress -n dredge
```

### Key Features

- **Auto-scaling**: HPA scales DREDGE server (2-10 replicas)
- **GPU Support**: NVIDIA GPU operator integration
- **High Availability**: Multi-replica deployments
- **Ingress**: NGINX ingress with TLS support
- **Persistent Storage**: Redis data persistence

## GitHub Container Registry

### Automated Publishing

The `.github/workflows/docker-publish.yml` workflow automatically:

1. Builds images on every push to `main` or `develop`
2. Tags images with branch names and semantic versions
3. Publishes to GitHub Container Registry (ghcr.io)
4. Supports multi-architecture builds
5. Generates artifact attestations

### Image Tags

Images are tagged with:
- `latest-cpu`, `latest-gpu`, `latest-dev` - Latest builds from main
- `develop-cpu`, `develop-gpu`, `develop-dev` - Latest builds from develop
- `v1.2.3-cpu` - Version tags from git tags
- `v1.2-cpu`, `v1-cpu` - Major/minor version tags

### Pulling Images

```bash
# Pull latest CPU image
docker pull ghcr.io/queenfi703/dredge-cli:latest-cpu

# Pull specific version
docker pull ghcr.io/queenfi703/dredge-cli:v0.2.0-cpu

# Pull GPU image
docker pull ghcr.io/queenfi703/dredge-cli:latest-gpu
```

### Authentication

```bash
# Login to GHCR
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# For Kubernetes, create image pull secret
kubectl create secret docker-registry ghcr-secret \
  --docker-server=ghcr.io \
  --docker-username=USERNAME \
  --docker-password=$GITHUB_TOKEN \
  --namespace=dredge
```

## Monitoring & Observability

### Prometheus Metrics

Access Prometheus at `http://localhost:9091`

Available metrics:
- Application metrics from metrics-exporter (port 9090)
- Container metrics (CPU, memory, network)
- Custom DREDGE metrics

### Grafana Dashboards

Access Grafana at `http://localhost:3000`

Default credentials:
- Username: `admin`
- Password: Set via `GRAFANA_PASSWORD` env var

Pre-configured:
- Prometheus datasource
- DREDGE service dashboards (to be customized)

### Logs

```bash
# View logs
docker-compose logs -f dredge-server
docker-compose logs -f quasimoto-mcp

# Export logs
docker-compose logs --no-color > dredge-logs.txt

# Kubernetes logs
kubectl logs -n dredge deployment/dredge-server -f
kubectl logs -n dredge deployment/quasimoto-mcp -f
```

### Tracing

For distributed tracing, consider adding:
- Jaeger
- Zipkin
- OpenTelemetry

## Security Considerations

### Container Security

1. **Image Scanning**
   ```bash
   # Scan with Trivy
   trivy image ghcr.io/queenfi703/dredge-cli:latest-cpu
   
   # Scan with Grype
   grype ghcr.io/queenfi703/dredge-cli:latest-cpu
   ```

2. **Non-root Users**
   - All containers run as non-root users where possible
   - Use USER directive in Dockerfile

3. **Read-only Filesystems**
   ```yaml
   security_opt:
     - no-new-privileges:true
   read_only: true
   ```

4. **Resource Limits**
   - CPU and memory limits enforced
   - Prevents resource exhaustion

### Network Security

1. **Network Policies** (Kubernetes)
   - Restrict pod-to-pod communication
   - Deny ingress by default

2. **TLS/SSL**
   - Use HTTPS in production
   - Configure SSL certificates in nginx/ssl/

3. **Secrets Management**
   - Use Docker secrets or Kubernetes secrets
   - Never commit secrets to git

### Best Practices

- [ ] Keep images up-to-date
- [ ] Scan for vulnerabilities regularly
- [ ] Use minimal base images
- [ ] Implement least-privilege access
- [ ] Enable audit logging
- [ ] Rotate credentials regularly

## Production Deployment

### Checklist

#### Pre-deployment
- [ ] Review and test all configurations
- [ ] Set up SSL certificates
- [ ] Configure DNS records
- [ ] Set strong passwords for all services
- [ ] Enable authentication/authorization
- [ ] Configure backup strategy
- [ ] Set up monitoring and alerting

#### Deployment
- [ ] Use specific version tags, not `latest`
- [ ] Enable health checks
- [ ] Configure resource limits
- [ ] Set up auto-scaling (HPA)
- [ ] Enable persistent storage
- [ ] Configure ingress/load balancer

#### Post-deployment
- [ ] Verify all services are running
- [ ] Test health endpoints
- [ ] Monitor logs for errors
- [ ] Set up automated backups
- [ ] Document disaster recovery procedures
- [ ] Conduct security audit

### Cloud Platform Guides

#### AWS ECS/EKS
```bash
# Push to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_URL
docker tag dredge-cli:cpu $ECR_URL/dredge-cli:cpu
docker push $ECR_URL/dredge-cli:cpu
```

#### Google Cloud Run/GKE
```bash
# Push to GCR
gcloud auth configure-docker
docker tag dredge-cli:cpu gcr.io/PROJECT_ID/dredge-cli:cpu
docker push gcr.io/PROJECT_ID/dredge-cli:cpu
```

#### Azure Container Instances/AKS
```bash
# Push to ACR
az acr login --name REGISTRY_NAME
docker tag dredge-cli:cpu REGISTRY_NAME.azurecr.io/dredge-cli:cpu
docker push REGISTRY_NAME.azurecr.io/dredge-cli:cpu
```

### Performance Tuning

1. **Redis Optimization**
   - Adjust maxmemory based on workload
   - Use appropriate eviction policy
   - Enable persistence (AOF/RDB)

2. **Container Resources**
   - Profile actual resource usage
   - Set appropriate requests/limits
   - Use resource quotas

3. **Network Optimization**
   - Enable HTTP/2
   - Use connection pooling
   - Implement caching strategies

4. **GPU Optimization**
   - Batch inference requests
   - Use mixed precision training
   - Monitor GPU utilization

## Troubleshooting

### Common Issues

**Container won't start**
```bash
docker-compose logs SERVICE_NAME
docker inspect CONTAINER_ID
```

**GPU not detected**
```bash
# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Verify Docker daemon
cat /etc/docker/daemon.json
```

**Network connectivity issues**
```bash
# Test internal DNS
docker exec dredge-flask ping redis

# Check network
docker network inspect dredge_dredge-network
```

**High memory usage**
```bash
# Monitor resources
docker stats

# Adjust limits
docker update --memory 2g --memory-swap 2g CONTAINER_ID
```

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [GitHub Packages Documentation](https://docs.github.com/en/packages)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)

## Support

For issues and questions:
- GitHub Issues: https://github.com/QueenFi703/DREDGE-Cli/issues
- Documentation: See [README.md](../README.md) and [BUILD.md](../BUILD.md)
