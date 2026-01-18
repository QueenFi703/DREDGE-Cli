# Quick Start Guide: Container Deployment

This guide provides quick commands to get started with DREDGE-Cli using containers.

## Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- (Optional) NVIDIA Docker for GPU support

## Quick Deployment Options

### 1. CPU-Only (Flask Server)

Fastest way to run the DREDGE API server:

```bash
# Using Make
make docker-profile-cpu

# Or directly with Docker Compose
docker compose -f docker-compose.profiles.yml --profile cpu up -d

# Access the API
curl http://localhost:3001/health
```

### 2. GPU-Enabled (MCP Server)

Requires NVIDIA GPU and drivers:

```bash
# Using Make
make docker-profile-gpu

# Or directly with Docker Compose
docker compose -f docker-compose.profiles.yml --profile gpu up -d

# Access the MCP server
curl http://localhost:3002/
```

### 3. Full Stack (All Services)

Complete deployment with monitoring:

```bash
# Using Make
make docker-profile-full

# Access services
# - DREDGE API: http://localhost:3001
# - MCP Server: http://localhost:3002
# - Grafana: http://localhost:3000
# - Prometheus: http://localhost:9091
# - Metrics: http://localhost:9090
```

### 4. With Reverse Proxy

Production setup with Nginx:

```bash
# Start full stack with proxy
docker compose -f docker-compose.profiles.yml --profile full --profile proxy up -d

# Access through Nginx
curl http://localhost/health
curl http://localhost/mcp
```

## Configuration

### Environment Variables

Copy and customize the environment file:

```bash
cp .env.container.example .env
# Edit .env with your settings
```

Key variables:
- `DREDGE_PORT` - DREDGE server port (default: 3001)
- `MCP_PORT` - MCP server port (default: 3002)
- `FLASK_ENV` - Flask environment (production/development)
- `CUDA_VISIBLE_DEVICES` - GPU device selection
- `GRAFANA_PASSWORD` - Grafana admin password

## Common Commands

```bash
# View logs
make docker-logs
docker compose -f docker-compose.profiles.yml logs -f

# Check status
make docker-ps
docker compose -f docker-compose.profiles.yml ps

# Stop services
make docker-profile-down
docker compose -f docker-compose.profiles.yml down

# Restart services
docker compose -f docker-compose.profiles.yml restart

# View resource usage
make docker-stats
```

## Health Checks

```bash
# DREDGE Server
curl http://localhost:3001/health

# MCP Server
curl http://localhost:3002/

# Redis
docker exec dredge-redis redis-cli ping

# Nginx (if proxy profile enabled)
curl http://localhost/nginx-health
```

## Pulling Pre-built Images

Images are available on GitHub Container Registry:

```bash
# Pull CPU image
docker pull ghcr.io/queenfi703/dredge-cli:latest-cpu

# Pull GPU image
docker pull ghcr.io/queenfi703/dredge-cli:latest-gpu

# Run directly
docker run -p 3001:3001 ghcr.io/queenfi703/dredge-cli:latest-cpu
```

## Troubleshooting

### Container won't start

```bash
# Check logs
docker compose -f docker-compose.profiles.yml logs SERVICE_NAME

# Inspect container
docker inspect CONTAINER_ID
```

### GPU not detected

```bash
# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Check Docker daemon configuration
cat /etc/docker/daemon.json
```

### Port already in use

```bash
# Find process using port
lsof -i :3001
# or
netstat -tulpn | grep 3001

# Change port in .env file
echo "DREDGE_PORT=3003" >> .env
```

## Production Deployment

For production deployments, see:
- [Container Architecture Guide](docs/CONTAINER_ARCHITECTURE.md) - Comprehensive container documentation
- [Kubernetes Guide](k8s/README.md) - Kubernetes deployment instructions

## Next Steps

1. Explore the API: `curl http://localhost:3001/`
2. Test MCP operations: See [README.md](README.md) for examples
3. Set up monitoring: Access Grafana at `http://localhost:3000`
4. Scale services: See [Container Architecture Guide](docs/CONTAINER_ARCHITECTURE.md)

## Support

For issues and questions:
- GitHub Issues: https://github.com/QueenFi703/DREDGE-Cli/issues
- Documentation: [BUILD.md](BUILD.md), [README.md](README.md)
