# Orion Gateway Deployment Guide

**Orion Gateway** is a production-grade API infrastructure layer for DREDGE. This guide covers deployment options, configuration, and best practices.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Local Development](#local-development)
4. [Container Deployment](#container-deployment)
5. [Production Deployment](#production-deployment)
6. [Configuration](#configuration)
7. [Monitoring & Logs](#monitoring--logs)
8. [Troubleshooting](#troubleshooting)

---

## Overview

Orion Gateway provides:

- ✅ RESTful API with authentication (API keys)
- ✅ Tier-based quota management (free/pro/enterprise)
- ✅ Usage tracking and analytics
- ✅ Multi-model reasoning orchestration
- ✅ Stripe-ready billing integration
- ✅ Rate limiting and DDoS protection
- ✅ Request logging and audit trails

### Supported Deployment Targets

- 🐳 Docker (local/staging)
- ☸️ Kubernetes (production)
- 🔧 AWS ECS (managed container)
- ☁️ Fly.io (serverless alternative)
- 🚀 Render (simple deployment)

---

## Prerequisites

### System Requirements

- Python 3.9+
- Docker 20.10+ (for container deployments)
- 2GB RAM minimum
- 10GB storage

### Required Services

- PostgreSQL 14+ (database)
- Redis 6+ (caching)
- Optional: Stripe account (billing)

---

## Local Development

### Quick Start

```bash
# Clone and enter repository
git clone https://github.com/QueenFi703/DREDGE-Cli.git
cd DREDGE-Cli

# Run quick start script
bash scripts/orion-quickstart.sh

# Or manual setup
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
pip install fastapi uvicorn sqlalchemy psycopg2-binary redis stripe click

# Start server
python -m dredge.orion_cli serve --debug
```

### First Request

```bash
curl -X POST http://localhost:3001/invoke \
  -H "x-api-key: demo-pro-key" \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello Orion","mode":"standard"}'
```

### Available Test Keys

- `demo-free-key` → Free tier (100 req/month)
- `demo-pro-key` → Pro tier (10k req/month)
- `demo-enterprise-key` → Enterprise (unlimited)

### Interactive Testing

```bash
python -m dredge.orion_cli demo --api-key demo-pro-key
```

---

## Container Deployment

### Docker Compose (Local Stack)

```bash
# Start full stack (API + Postgres + Redis)
docker-compose up -d

# Logs
docker-compose logs -f orion-api

# Stop
docker-compose down
```

### Docker Compose File

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: orion
      POSTGRES_USER: orion
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  orion-api:
    build: .
    environment:
      DATABASE_URL: postgresql://orion:${DB_PASSWORD}@postgres:5432/orion
      REDIS_URL: redis://redis:6379
      STRIPE_SECRET_KEY: ${STRIPE_SECRET_KEY}
    ports:
      - "3001:3001"
    depends_on:
      - postgres
      - redis

volumes:
  postgres_data:
```

### Build Docker Image

```bash
# Create Dockerfile
cat > Dockerfile <<'EOF'
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml requirements.txt ./

RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir \
      fastapi uvicorn sqlalchemy psycopg2-binary redis stripe

COPY . .

EXPOSE 3001

CMD ["python", "-m", "dredge.orion_cli", "serve", "--host", "0.0.0.0", "--port", "3001"]
EOF

# Build
docker build -t orion-gateway:latest .

# Run
docker run -e DATABASE_URL=postgresql://... \
           -e STRIPE_SECRET_KEY=sk_... \
           -p 3001:3001 \
           orion-gateway:latest
```

---

## Production Deployment

### AWS ECS (Recommended for $10K/month scale)

#### 1. Create ECR Repository

```bash
aws ecr create-repository --repository-name orion-gateway --region us-east-1
```

#### 2. Push Image

```bash
# Build and tag
docker build -t orion-gateway:latest .
docker tag orion-gateway:latest \
  <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/orion-gateway:latest

# Push
docker push <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/orion-gateway:latest
```

#### 3. Create RDS Database

```bash
aws rds create-db-instance \
  --db-instance-identifier orion-postgres \
  --engine postgres \
  --engine-version 15.4 \
  --db-instance-class db.t3.micro \
  --allocated-storage 20 \
  --master-username orion \
  --master-user-password <SECURE_PASSWORD> \
  --publicly-accessible false
```

#### 4. Create ElastiCache Redis

```bash
aws elasticache create-cache-cluster \
  --cache-cluster-id orion-redis \
  --engine redis \
  --cache-node-type cache.t3.micro \
  --engine-version 7.0
```

#### 5. Create ECS Cluster & Service

See `docs/terraform/main.tf` for complete IaC setup.

### Kubernetes Deployment

```bash
# Create namespace
kubectl create namespace orion

# Apply secrets
kubectl create secret generic orion-secrets \
  --from-literal=database-url=postgresql://... \
  --from-literal=redis-url=redis://... \
  --from-literal=stripe-secret-key=sk_... \
  -n orion

# Apply manifests
kubectl apply -f k8s/postgres.yaml
kubectl apply -f k8s/deployment.yaml

# Verify
kubectl get pods -n orion
kubectl get svc -n orion
```

### Fly.io (Quick Deployment)

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Launch app
fly launch --region lax --name orion-gateway

# Set secrets
fly secrets set DATABASE_URL=postgresql://...
fly secrets set STRIPE_SECRET_KEY=sk_...

# Deploy
fly deploy

# Monitor
fly logs
```

---

## Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@host:5432/orion

# Redis
REDIS_URL=redis://localhost:6379/0

# Stripe
STRIPE_SECRET_KEY=sk_live_xxx
STRIPE_PUBLISHABLE_KEY=pk_live_xxx

# API
DEBUG=false
LOG_LEVEL=INFO
WORKERS=4

# CORS
ALLOWED_ORIGINS=https://app.example.com

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60  # seconds
```

### .env Example

```bash
cp .env.example .env
# Edit .env with your values
```

### Tier Configuration

```python
# src/dredge/orion_config.py
STRIPE_PRODUCTS = {
    "free": {
        "name": "Orion Free",
        "price": 0,
        "requests_per_month": 100,
        "features": ["standard_mode"]
    },
    "pro": {
        "name": "Orion Pro",
        "price": 2900,  # $29/month in cents
        "requests_per_month": 10000,
        "features": ["all_modes", "analytics", "priority_support"]
    },
    "enterprise": {
        "name": "Orion Enterprise",
        "price": None,  # Custom pricing
        "requests_per_month": 1000000,
        "features": ["unlimited", "sso", "sla", "dedicated_support"]
    }
}

# Stripe product IDs (set after creation)
STRIPE_PRODUCT_IDS = {
    "pro": "prod_xxx",
    "enterprise": "prod_yyy"
}
```

---

## Monitoring & Logs

### Health Checks

```bash
# System health
curl http://localhost:3001/health

# Admin stats
curl http://localhost:3001/admin/stats
```

### Logging

```bash
# View logs (Docker)
docker-compose logs -f orion-api

# View logs (Kubernetes)
kubectl logs -f deployment/orion-api -n orion

# View logs (Fly.io)
fly logs
```

### Metrics to Track

- **Requests per second** (throughput)
- **Latency** (p50, p95, p99)
- **Quota utilization** by tier
- **Error rates** (4xx, 5xx)
- **Auth failures** (invalid keys)

---

## Pre-Launch Checklist

### ✅ Development

- [ ] Local development works (`make serve`)
- [ ] Tests pass (`pytest`)
- [ ] API key authentication working
- [ ] Quota limiting tested
- [ ] All modes (standard, deep, transform, analyze) functional
- [ ] Error handling tested

### ✅ Staging

- [ ] Deployed to staging environment
- [ ] Database migrations applied
- [ ] Environment variables configured
- [ ] SSL/TLS certificates valid
- [ ] Rate limiting configured
- [ ] Monitoring alerts set up
- [ ] Backup strategy implemented

### ✅ Production

- [ ] Deployed to production
- [ ] Database backed up
- [ ] Monitoring dashboard operational
- [ ] Log aggregation working
- [ ] Incident response plan documented
- [ ] On-call rotation established
- [ ] Customer support process ready

### ✅ Operations

- [ ] Stripe billing configured
- [ ] Invoice generation automated
- [ ] API documentation published
- [ ] SDK/client libraries available
- [ ] Status page setup (statuspage.io)
- [ ] Uptime monitoring (Pingdom/Uptimerobot)
- [ ] Error tracking (Sentry)

---

## Troubleshooting

### API Not Responding

```bash
# Check service running
ps aux | grep orion

# Check port in use
lsof -i :3001

# Restart service
python -m dredge.orion_cli serve
```

### Database Connection Error

```bash
# Test connection
psql postgresql://user:pass@host:5432/orion

# Check DATABASE_URL
echo $DATABASE_URL

# Verify credentials
aws rds describe-db-instances --query 'DBInstances[0].Endpoint'
```

### Redis Connection Error

```bash
# Test Redis
redis-cli ping

# Check REDIS_URL
echo $REDIS_URL

# Restart Redis
redis-server
```

### High Latency

```bash
# Check database query performance
EXPLAIN ANALYZE SELECT * FROM request_logs LIMIT 10;

# Check Redis memory
redis-cli INFO memory

# Scale horizontally
# - Add more API replicas
# - Enable caching
# - Optimize database indexes
```

### Quota Not Enforcing

```bash
# Check quota logic in orion_gateway.py
# Verify organization tier in database
# Check request_logs table for count
SELECT COUNT(*) FROM request_logs 
  WHERE org_id = 'xxx' 
  AND DATE_TRUNC('month', created_at) = DATE_TRUNC('month', NOW());
```

---

## Scaling Path to $10K/Month

### Phase 1: MVP (0-3 months)

- Single instance deployment
- Shared database
- Manual billing
- Basic monitoring

### Phase 2: Scaling (3-6 months)

- Multi-instance deployment (3+ replicas)
- Database read replicas
- Automated billing (Stripe)
- Advanced monitoring (Datadog)

### Phase 3: Enterprise (6-12 months)

- Multi-region deployment
- Database sharding
- Custom integrations
- Dedicated support

---

## Support & Resources

- 📖 API Documentation: `/docs` (Swagger UI)
- 🐛 Issue Tracker: GitHub Issues
- 💬 Community: GitHub Discussions
- 📧 Support: support@example.com
- 📱 Status: status.example.com

---

**Happy deploying! 🚀**
