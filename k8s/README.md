# Kubernetes Deployment Guide

This directory contains Kubernetes manifests for deploying DREDGE-Cli in a production Kubernetes cluster.

## Architecture

The deployment consists of:

- **Redis Cache** - Distributed caching layer with persistent storage
- **DREDGE Server** - Flask API server (CPU-only, horizontally scalable)
- **Quasimoto MCP Server** - PyTorch-based MCP server with GPU support
- **Horizontal Pod Autoscaler** - Auto-scales DREDGE server based on CPU/memory usage

## Prerequisites

1. **Kubernetes Cluster** (v1.24+)
   - GKE, EKS, AKS, or self-managed cluster
   
2. **kubectl** - Kubernetes CLI tool
   ```bash
   kubectl version --client
   ```

3. **NVIDIA GPU Operator** (for GPU workloads)
   ```bash
   kubectl apply -f https://raw.githubusercontent.com/NVIDIA/gpu-operator/master/deployments/gpu-operator.yaml
   ```

4. **Ingress Controller** (optional, for external access)
   ```bash
   # NGINX Ingress
   kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.1/deploy/static/provider/cloud/deploy.yaml
   ```

5. **Cert Manager** (optional, for TLS certificates)
   ```bash
   kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
   ```

## Quick Start

### 1. Deploy with kubectl

```bash
# Deploy all resources
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n dredge
kubectl get svc -n dredge
kubectl get ingress -n dredge
```

### 2. Deploy with Kustomize

```bash
# Deploy using kustomize
kubectl apply -k k8s/

# Or build and inspect before applying
kubectl kustomize k8s/ | less
kubectl kustomize k8s/ | kubectl apply -f -
```

### 3. Verify Deployment

```bash
# Check pod status
kubectl get pods -n dredge -w

# Check logs
kubectl logs -n dredge deployment/dredge-server -f
kubectl logs -n dredge deployment/quasimoto-mcp -f

# Test health endpoints
kubectl port-forward -n dredge svc/dredge-server-service 3001:3001
curl http://localhost:3001/health

kubectl port-forward -n dredge svc/quasimoto-mcp-service 3002:3002
curl http://localhost:3002/
```

## Configuration

### ConfigMaps

Edit `configmap.yaml` to modify environment variables:

```yaml
data:
  FLASK_ENV: production
  CACHE_ENABLED: "true"
  METRICS_ENABLED: "true"
```

Apply changes:
```bash
kubectl apply -f k8s/configmap.yaml
kubectl rollout restart deployment/dredge-server -n dredge
```

### Scaling

#### Manual Scaling
```bash
# Scale DREDGE server
kubectl scale deployment/dredge-server --replicas=5 -n dredge

# Scale MCP server (not recommended for GPU workloads)
kubectl scale deployment/quasimoto-mcp --replicas=2 -n dredge
```

#### Auto-scaling
The HPA automatically scales DREDGE server (2-10 replicas) based on:
- CPU utilization (target: 70%)
- Memory utilization (target: 80%)

Monitor auto-scaling:
```bash
kubectl get hpa -n dredge -w
```

### Resource Limits

Edit resource requests/limits in deployment manifests:

```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "1Gi"
    cpu: "1000m"
```

## Storage

Redis uses a PersistentVolumeClaim (1Gi by default). To change:

```yaml
# In redis.yaml
spec:
  resources:
    requests:
      storage: 5Gi
```

## Ingress & TLS

### Update Hostnames

Edit `dredge-server.yaml` and `quasimoto-mcp.yaml`:

```yaml
spec:
  tls:
  - hosts:
    - dredge.yourdomain.com
  rules:
  - host: dredge.yourdomain.com
```

### TLS Certificates

Using cert-manager:

1. Create ClusterIssuer:
```bash
cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: your-email@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

2. Certificates will be auto-provisioned via ingress annotations

## GPU Support

### Requirements
- Nodes with NVIDIA GPUs
- NVIDIA GPU Operator installed
- Nodes labeled with `nvidia.com/gpu.present=true`

### Verify GPU
```bash
# Check GPU nodes
kubectl get nodes -l nvidia.com/gpu.present=true

# Check GPU pods
kubectl get pods -n dredge -l app=quasimoto-mcp

# Verify CUDA availability
kubectl exec -n dredge deployment/quasimoto-mcp -- python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Monitoring

### Pod Metrics
```bash
# CPU/Memory usage
kubectl top pods -n dredge

# Node metrics
kubectl top nodes
```

### Application Logs
```bash
# Stream logs
kubectl logs -n dredge -l app=dredge-server -f --tail=100

# Export logs
kubectl logs -n dredge deployment/dredge-server > dredge-server.log
```

### Events
```bash
# Check events
kubectl get events -n dredge --sort-by='.lastTimestamp'

# Watch events
kubectl get events -n dredge -w
```

## Troubleshooting

### Pods Not Starting

```bash
# Describe pod
kubectl describe pod -n dredge <pod-name>

# Check image pull
kubectl get events -n dredge | grep Failed

# Check resource constraints
kubectl describe nodes | grep -A 5 "Allocated resources"
```

### GPU Not Available

```bash
# Check GPU operator
kubectl get pods -n gpu-operator-resources

# Check device plugin
kubectl logs -n gpu-operator-resources <nvidia-device-plugin-pod>

# Verify node labels
kubectl describe node <node-name> | grep nvidia.com/gpu
```

### Connection Issues

```bash
# Test internal connectivity
kubectl run -n dredge test-pod --rm -it --image=curlimages/curl -- sh
# Inside pod:
curl http://dredge-server-service:3001/health
curl http://quasimoto-mcp-service:3002/

# Test DNS resolution
kubectl run -n dredge test-dns --rm -it --image=busybox -- nslookup redis-service
```

## Cleanup

```bash
# Delete all resources
kubectl delete -f k8s/

# Or with kustomize
kubectl delete -k k8s/

# Delete namespace (removes everything)
kubectl delete namespace dredge
```

## Production Considerations

### Security
- [ ] Enable Network Policies to restrict pod communication
- [ ] Use Secrets for sensitive data instead of ConfigMaps
- [ ] Implement Pod Security Standards (PSS)
- [ ] Scan container images for vulnerabilities
- [ ] Enable RBAC with least-privilege access

### High Availability
- [ ] Run multiple replicas of critical services
- [ ] Use Pod Disruption Budgets (PDB)
- [ ] Spread pods across multiple availability zones
- [ ] Implement proper health checks

### Monitoring & Observability
- [ ] Install Prometheus + Grafana for metrics
- [ ] Set up ELK/EFK stack for log aggregation
- [ ] Configure alerting for critical events
- [ ] Implement distributed tracing (Jaeger/Zipkin)

### Backup & Disaster Recovery
- [ ] Back up Redis data regularly
- [ ] Document recovery procedures
- [ ] Test disaster recovery scenarios

## Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/)
- [Kustomize](https://kustomize.io/)
- [Cert Manager](https://cert-manager.io/)
