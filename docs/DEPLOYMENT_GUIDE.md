# DREDGE Deployment Guide

## Quick Start

### Basic Deployment (Single Node)

```bash
# Clone repository
git clone https://github.com/QueenFi703/DREDGE-Cli.git
cd DREDGE-Cli

# Copy environment configuration
cp .env.example .env
# Edit .env with your settings

# Start services using Docker Compose
docker-compose -f docker-compose.enhanced.yml up -d

# Verify services are running
docker-compose -f docker-compose.enhanced.yml ps

# Check logs
docker-compose -f docker-compose.enhanced.yml logs -f
```

### Services Overview

After deployment, the following services will be available:

| Service | Port | Description |
|---------|------|-------------|
| DREDGE x Dolly Server | 3001 | Main API server with Dolly integration |
| Quasimoto MCP Server | 3002 | Model Context Protocol server (GPU-enabled) |
| Redis Cache | 6379 | Distributed caching layer |
| Metrics Exporter | 9090 | Prometheus-compatible metrics endpoint |
| Workers | N/A | Background task processors (2 instances) |

### Accessing Services

```bash
# Check DREDGE server health
curl http://localhost:3001/health

# List MCP capabilities
curl http://localhost:3002/

# Get metrics
curl http://localhost:3002/mcp \
  -H "Content-Type: application/json" \
  -d '{"operation": "get_metrics"}'

# Get cache statistics
curl http://localhost:3002/mcp \
  -H "Content-Type: application/json" \
  -d '{"operation": "get_cache_stats"}'
```

## Configuration

### Environment Variables

Edit `.env` file to configure:

#### Cache Settings
- `CACHE_ENABLED=true` - Enable/disable caching
- `CACHE_BACKEND=redis` - Backend: memory, file, or redis
- `CACHE_TTL=3600` - Time-to-live in seconds

#### GPU Settings
- `DEVICE=auto` - Device selection (auto, cpu, cuda, mps)
- `CUDA_VISIBLE_DEVICES=0` - GPU device ID for CUDA

#### Worker Settings
- `NUM_WORKERS=2` - Number of worker instances
- `WORKER_CONCURRENCY=4` - Concurrent tasks per worker

#### Monitoring
- `METRICS_ENABLED=true` - Enable metrics collection
- `LOG_LEVEL=INFO` - Logging verbosity

## Architecture

### Single-Node Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Load Balancer (Optional)            │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
┌────────────────┐      ┌────────────────┐
│ DREDGE Server  │      │  MCP Server    │
│   (Port 3001)  │      │  (Port 3002)   │
│   CPU-based    │      │  GPU-enabled   │
└────────┬───────┘      └────────┬───────┘
         │                       │
         │    ┌──────────────────┤
         │    │                  │
         ▼    ▼                  ▼
    ┌──────────────┐      ┌──────────────┐
    │    Redis     │      │   Workers    │
    │   (Cache)    │      │   (2 nodes)  │
    └──────────────┘      └──────────────┘
```

### Scale-Out Architecture

For high-load production deployments:

```
┌─────────────────────────────────────────────────────────┐
│              HAProxy / Nginx Load Balancer              │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┴────────────────┬────────────────┐
         │                            │                │
         ▼                            ▼                ▼
┌─────────────────┐         ┌─────────────────┐  ┌─────────────────┐
│  DREDGE Server  │         │  DREDGE Server  │  │  MCP Server     │
│    Instance 1   │         │    Instance 2   │  │   (GPU Node)    │
└────────┬────────┘         └────────┬────────┘  └────────┬────────┘
         │                            │                     │
         └──────────┬─────────────────┘                     │
                    │                                       │
                    ▼                                       ▼
         ┌────────────────────┐              ┌──────────────────────┐
         │  Redis Cluster     │              │   Worker Pool        │
         │  (3 nodes)         │              │   (4-8 workers)      │
         │  - Caching         │              │   - Task processing  │
         │  - Session store   │              │   - Load distribution│
         └────────────────────┘              └──────────────────────┘
                    │
                    ▼
         ┌────────────────────┐
         │  Metrics/Monitoring│
         │  - Prometheus      │
         │  - Grafana         │
         └────────────────────┘
```

## Scaling Strategies

### Horizontal Scaling

#### Add More Workers

Edit `docker-compose.enhanced.yml`:

```yaml
  dredge-worker-3:
    build:
      context: .
      target: cpu-build
    container_name: dredge-worker-3
    environment:
      - WORKER_ID=worker-3
      # ... other settings
```

Then restart:
```bash
docker-compose -f docker-compose.enhanced.yml up -d --scale dredge-worker-1=4
```

#### Add More MCP Server Instances

For GPU-heavy workloads, add more MCP servers with different GPU assignments:

```yaml
  quasimoto-mcp-2:
    # ... same as quasimoto-mcp
    environment:
      - CUDA_VISIBLE_DEVICES=1  # Use second GPU
    ports:
      - "3003:3002"  # Different external port
```

### Vertical Scaling

#### Resource Limits

Add resource constraints in `docker-compose.enhanced.yml`:

```yaml
services:
  quasimoto-mcp:
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
        reservations:
          cpus: '2.0'
          memory: 4G
```

## Load Testing

### Run Batch Pipeline

```bash
# Basic load test (100 tasks, 4 workers)
python -m dredge.batch_pipeline --tasks 100 --workers 4

# Large-scale test (1000 tasks, 8 workers)
python -m dredge.batch_pipeline --tasks 1000 --workers 8

# Test without caching
python -m dredge.batch_pipeline --tasks 100 --no-cache
```

### Expected Performance

#### With Caching Enabled
- **Throughput**: 20-50 tasks/sec (CPU-only)
- **Throughput**: 50-100 tasks/sec (with GPU)
- **Latency P50**: 20-50ms
- **Latency P95**: 100-200ms
- **Cache Hit Rate**: 15-20% (with 20% repeat queries)

#### Without Caching
- **Throughput**: 15-30 tasks/sec (CPU-only)
- **Latency P50**: 30-80ms
- **Latency P95**: 150-300ms

### Load Testing Tools

#### Apache Bench
```bash
# Test DREDGE server
ab -n 1000 -c 10 http://localhost:3001/health

# Test MCP server
ab -n 1000 -c 10 -T application/json \
   -p mcp_request.json \
   http://localhost:3002/mcp
```

#### wrk
```bash
# Install wrk
sudo apt-get install wrk

# Run load test
wrk -t4 -c100 -d30s http://localhost:3001/health
```

## Monitoring

### Metrics Endpoints

```bash
# Server metrics
curl http://localhost:3002/mcp \
  -H "Content-Type: application/json" \
  -d '{"operation": "get_metrics"}' | jq

# Cache statistics
curl http://localhost:3002/mcp \
  -H "Content-Type: application/json" \
  -d '{"operation": "get_cache_stats"}' | jq
```

### Prometheus Integration

Add to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'dredge-metrics'
    static_configs:
      - targets: ['localhost:9090']
```

### Log Monitoring

```bash
# Follow all logs
docker-compose -f docker-compose.enhanced.yml logs -f

# Follow specific service
docker-compose -f docker-compose.enhanced.yml logs -f quasimoto-mcp

# Show last 100 lines
docker-compose -f docker-compose.enhanced.yml logs --tail=100
```

## Production Hardening

### Security Checklist

- [ ] Enable authentication (API keys)
- [ ] Enable rate limiting
- [ ] Use HTTPS/TLS
- [ ] Set up firewall rules
- [ ] Use secrets management (Docker secrets, Vault)
- [ ] Enable audit logging
- [ ] Implement CORS policies
- [ ] Regular security updates

### High Availability

1. **Multiple Server Instances**: Run 2+ instances behind load balancer
2. **Redis Cluster**: Use Redis Sentinel or Cluster mode
3. **Health Checks**: Configure proper health check endpoints
4. **Graceful Shutdown**: Implement proper signal handling
5. **Circuit Breakers**: Add circuit breakers for external dependencies

### Backup and Recovery

```bash
# Backup Redis data
docker exec dredge-redis redis-cli BGSAVE

# Export Redis data
docker exec dredge-redis redis-cli --rdb /data/dump.rdb

# Restore from backup
docker cp dump.rdb dredge-redis:/data/
docker restart dredge-redis
```

## Troubleshooting

### Common Issues

#### GPU Not Available
```bash
# Check CUDA availability
docker exec quasimoto-gpu python -c "import torch; print(torch.cuda.is_available())"

# Check GPU visibility
docker exec quasimoto-gpu nvidia-smi
```

#### Redis Connection Issues
```bash
# Test Redis connection
docker exec dredge-redis redis-cli ping

# Check Redis logs
docker logs dredge-redis
```

#### High Memory Usage
```bash
# Check memory usage
docker stats

# Clear cache
curl http://localhost:3002/mcp \
  -H "Content-Type: application/json" \
  -d '{"operation": "clear_cache"}'
```

## Maintenance

### Update Deployment

```bash
# Pull latest changes
git pull origin main

# Rebuild images
docker-compose -f docker-compose.enhanced.yml build

# Restart services
docker-compose -f docker-compose.enhanced.yml up -d

# Remove old images
docker image prune -f
```

### Database Migrations

For future database integrations:
```bash
# Run migrations
docker-compose exec dredge-server python -m dredge.migrations upgrade

# Rollback
docker-compose exec dredge-server python -m dredge.migrations downgrade
```

## Support

For issues and questions:
- GitHub Issues: https://github.com/QueenFi703/DREDGE-Cli/issues
- Documentation: See `docs/` directory
- Examples: See `examples/` directory
