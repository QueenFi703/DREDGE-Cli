# DREDGE-Cli - Unified Authentication & API Gateway System

Production-grade API key management and unified authentication middleware for Orion Gateway, DREDGE Pipeline, and multi-service orchestration.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- FastAPI
- httpx (for proxy requests)

### Installation

```bash
# Clone repository
git clone https://github.com/QueenFi703/DREDGE-Cli.git
cd DREDGE-Cli

# Install dependencies
pip install fastapi uvicorn httpx

# Optional: For development
pip install pytest pytest-asyncio black flake8
```

### Run Unified Gateway (5 minutes)

```bash
# Start the unified auth gateway on port 9000
python unified_auth_gateway.py

# Output includes test API keys:
# Test key: orion_key_XEkFnoDSCfoQ...
# Admin key: orion_key_7kQzPmN9vL2xJwRsT5fGhYuI8jKpOqAb...

# Access Swagger UI
# http://127.0.0.1:9000/docs
```

### Test Your First Request

```bash
# Health check (no auth)
curl http://127.0.0.1:9000/health

# Protected endpoint (with auth)
curl -H "x-api-key: orion_key_XEkFnoDSCfoQ..." \
  http://127.0.0.1:9000/orion/health

# Create new API key (admin)
curl -X POST http://127.0.0.1:9000/admin/keys/create \
  -H "x-api-key: orion_key_7kQzPmN9vL2xJwRsT5fGhYuI8jKpOqAb..." \
  -H "Content-Type: application/json" \
  -d '{"name":"Production","tier":"pro","mode":"invoke_only"}'
```

## 📋 What's Included

### Core System
- **api_key_manager.py** (19.9 KB)
  - Secure key generation (256-bit entropy)
  - SHA-256 hashing with random salt
  - Key validation and verification
  - Usage tracking and billing
  - Tier and mode management

- **unified_auth_middleware.py** (16.4 KB) - **NEW**
  - Centralized authentication middleware
  - Automatic route protection
  - No per-route dependencies needed
  - Unified usage tracking
  - Admin access control

- **unified_auth_gateway.py** (16.1 KB)
  - Central proxy on port 9000
  - Routes requests to backend services
  - Automatic authentication and rate limiting
  - Admin key management panel

- **fastapi_auth_middleware.py** (17.3 KB)
  - FastAPI dependency injection (legacy approach)
  - Per-route authentication helpers
  - Custom decorators

### Examples
- **orion_gateway_authenticated.py** (14.6 KB)
  - Complete standalone example
  - All endpoints with authentication
  - Admin panel
  - Production-ready

### Documentation (80+ KB)
- **API_KEY_SYSTEM_GUIDE.txt** - Complete reference guide
- **API_KEY_QUICK_REFERENCE.txt** - Developer cheat sheet
- **PORT_MANAGEMENT_GUIDE.txt** - Port configuration
- **PORT_ARCHITECTURE_VISUAL.txt** - Architecture diagrams
- **PRODUCTION_DEPLOYMENT_SUMMARY.txt** - Production guide

## 🏗️ Architecture

### Port Architecture
```
Client (x-api-key: orion_key_xxx)
    ↓
Port 9000: Unified Auth Gateway
    ├─ /orion/*     → Port 8080 (Orion Gateway)
    ├─ /dredge/*    → Port 3001 (DREDGE Pipeline)
    ├─ /advanced/*  → Port 8000 (Dashboard)
    └─ /mcp/*       → Port 8001 (MCP Gateway)
```

### Middleware Flow
```
Request
  ↓
UnifiedAuthMiddleware
  ├─ Extract x-api-key header
  ├─ Validate against stored hashes
  ├─ Check rate limits
  ├─ Verify admin access if needed
  └─ Store metadata in scope
  ↓
Route Handler
  └─ Access metadata via get_auth_metadata(request)
  ↓
Usage Tracking
  └─ Record endpoint, method, status, duration
  ↓
Response + Rate Limit Headers
  └─ X-RateLimit-Limit: 10000
  └─ X-RateLimit-Remaining: 9850
  └─ X-RateLimit-Tier: pro
```

## 🔐 Security Features

- **256-bit Key Generation**: Cryptographically secure random generation
- **SHA-256 Hashing**: With random 16-byte salt (bcrypt-like format)
- **Constant-Time Comparison**: Prevents timing attacks
- **Rate Limiting**: Per-tier monthly limits
- **Admin Access Control**: Full_access mode for admin operations
- **Usage Tracking**: Per-request tracking with metadata
- **Key Expiration**: Optional expiration support
- **Key Revocation**: Immediate key disabling

## 📊 API Key Tiers

| Tier | Monthly Limit | Use Case |
|------|--------------|----------|
| FREE | 100 | Testing |
| STARTER | 1,000 | Small production apps |
| PRO | 10,000 | Growing applications |
| ENTERPRISE | Unlimited | Enterprise deployments |

## 🔑 Access Modes

| Mode | Permissions |
|------|-----------|
| READ_ONLY | Access usage stats |
| INVOKE_ONLY | Call inference endpoints |
| FULL_ACCESS | All endpoints including admin |

## 🛣️ API Endpoints

### Public (No Auth)
```
GET  /
GET  /health
GET  /services
GET  /docs
GET  /openapi.json
```

### Protected (Auth Required)
```
POST /orion/invoke          - Inference endpoint
GET  /orion/health          - Service health
GET  /orion/usage           - Usage statistics

POST /dredge/api/architecture/pipeline/execute
GET  /dredge/api/architecture/health

GET  /advanced/dashboard
GET  /advanced/features

GET  /usage                 - Your usage
GET  /key/info              - Your key info
```

### Admin (Full_access Required)
```
POST /admin/keys/create             - Create API key
GET  /admin/keys/list               - List all keys
POST /admin/keys/{key_id}/revoke    - Revoke key
GET  /admin/stats                   - System statistics
GET  /admin/usage/{key_id}          - Usage for key
```

## 💻 Usage Examples

### Python with HTTPX
```python
import httpx

api_key = "orion_key_xxx"

async with httpx.AsyncClient() as client:
    # Call protected endpoint
    resp = await client.post(
        "http://127.0.0.1:9000/orion/invoke",
        headers={"x-api-key": api_key},
        json={"input": "Hello", "mode": "standard"}
    )
    
    print(f"Remaining: {resp.headers['X-RateLimit-Remaining']}")
    print(f"Result: {resp.json()}")
```

### JavaScript with Fetch
```javascript
const apiKey = "orion_key_xxx";

const response = await fetch(
  "http://127.0.0.1:9000/orion/invoke",
  {
    method: "POST",
    headers: {
      "x-api-key": apiKey,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      input: "Hello",
      mode: "standard"
    })
  }
);

const remaining = response.headers.get("X-RateLimit-Remaining");
const data = await response.json();
```

### cURL
```bash
# Create API key (admin)
curl -X POST http://127.0.0.1:9000/admin/keys/create \
  -H "x-api-key: ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name":"Prod","tier":"pro","mode":"invoke_only"}'

# Use API key
curl -X POST http://127.0.0.1:9000/orion/invoke \
  -H "x-api-key: NEW_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello","mode":"standard"}'

# Check usage
curl -H "x-api-key: NEW_KEY" \
  http://127.0.0.1:9000/usage

# View stats (admin)
curl -H "x-api-key: ADMIN_KEY" \
  http://127.0.0.1:9000/admin/stats
```

## 🧩 Integration

### Add to Existing FastAPI App

#### Method 1: Unified Middleware (Recommended)
```python
from fastapi import FastAPI
from api_key_manager import init_api_key_system
from unified_auth_middleware import UnifiedAuthMiddleware, get_auth_metadata

app = FastAPI()
key_store, tracker = init_api_key_system()

# Single line adds authentication to all routes!
app.add_middleware(
    UnifiedAuthMiddleware,
    key_store=key_store,
    tracker=tracker
)

@app.post("/invoke")
async def invoke(request: Request):
    # Metadata automatically available
    metadata = get_auth_metadata(request)
    return {"key_id": metadata.key_id, "tier": metadata.tier.value}
```

#### Method 2: Dependency Injection (Legacy)
```python
from fastapi import FastAPI, Depends
from api_key_manager import init_api_key_system
from fastapi_auth_middleware import APIKeyDependencies

app = FastAPI()
key_store, tracker = init_api_key_system()
deps = APIKeyDependencies(key_store, tracker)

@app.post("/invoke")
async def invoke(key = Depends(deps.verify_rate_limit)):
    return {"key_id": key.key_id}
```

## 📦 File Structure

```
DREDGE-Cli/
├── api_key_manager.py                    # Core API key system
├── unified_auth_middleware.py            # NEW: Centralized middleware
├── unified_auth_gateway.py               # Central proxy gateway
├── fastapi_auth_middleware.py            # Legacy: Dependency injection
├── orion_gateway_authenticated.py        # Standalone example
│
├── API_KEY_SYSTEM_GUIDE.txt              # Complete guide
├── API_KEY_QUICK_REFERENCE.txt           # Cheat sheet
├── PORT_MANAGEMENT_GUIDE.txt             # Port configuration
├── PORT_ARCHITECTURE_VISUAL.txt          # Architecture diagrams
├── PRODUCTION_DEPLOYMENT_SUMMARY.txt     # Production guide
│
└── data/
    └── unified_api_keys.json             # API keys storage (auto-created)
```

## 🚀 Deployment

### Development
```bash
python unified_auth_gateway.py
# Runs on http://127.0.0.1:9000
```

### Production (Docker)
```yaml
version: '3.8'
services:
  gateway:
    build: .
    ports:
      - "9000:9000"
    environment:
      - ORION_URL=http://orion:8080
      - DREDGE_URL=http://dredge:3001
      - AUTH_STORAGE_PATH=/data/api_keys.json
    volumes:
      - ./data:/data
```

### Production (Kubernetes)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: auth-gateway
spec:
  replicas: 3
  selector:
    matchLabels:
      app: auth-gateway
  template:
    metadata:
      labels:
        app: auth-gateway
    spec:
      containers:
      - name: gateway
        image: auth-gateway:latest
        ports:
        - containerPort: 9000
        env:
        - name: ORION_URL
          value: "http://orion-service:8080"
```

## 📈 Monitoring

### Check System Health
```bash
curl http://127.0.0.1:9000/health
```

### View Usage Statistics
```bash
curl -H "x-api-key: KEY" http://127.0.0.1:9000/usage
```

### Admin Dashboard
```
http://127.0.0.1:9000/docs
```

## 🔄 Rate Limiting

- **Reset Date**: 1st of each month (UTC 00:00)
- **Reset Behavior**: All keys reset to 0 requests
- **Overage**: 429 Too Many Requests returned
- **Headers**: X-RateLimit-* headers included in all responses

## 🛡️ Security Best Practices

1. **Key Storage**
   - Use environment variables (not hardcoded)
   - Different keys per environment
   - Never commit keys to version control

2. **Key Management**
   - Rotate keys monthly
   - Revoke compromised keys immediately
   - Use strong Admin keys only where needed

3. **API Exposure**
   - Use HTTPS in production
   - Firewall ports 8000, 8001, 8080, 3001
   - Expose only port 9000

4. **Monitoring**
   - Track failed auth attempts
   - Alert on unusual usage patterns
   - Audit all admin operations

## 📝 Logging

All requests are logged with:
- API key ID
- Endpoint called
- HTTP method and status
- Response time (ms)
- Client IP address
- User agent

Check logs in terminal running the gateway.

## 🐛 Troubleshooting

### Gateway won't start on port 9000
```bash
# Check if port is already in use
netstat -an | findstr 9000
# Kill existing process if needed
```

### "Invalid API key" error
```bash
# Verify key format: orion_key_xxxxx
# Check key hasn't been revoked
# Check key hasn't expired
```

### "429 Too Many Requests"
```bash
# Check remaining requests
curl -H "x-api-key: KEY" http://127.0.0.1:9000/usage
# Upgrade to higher tier if needed
# Wait for monthly reset (1st of month)
```

### Backend service returns 502
```bash
# Verify backend is running on expected port
curl http://127.0.0.1:8080/health  # Orion
curl http://127.0.0.1:3001/health  # DREDGE
curl http://127.0.0.1:8000/health  # Advanced
```

## 📚 Documentation

- **API_KEY_SYSTEM_GUIDE.txt** - 18.8 KB
  - Complete architecture and design
  - Security implementation details
  - API reference
  - Integration examples

- **API_KEY_QUICK_REFERENCE.txt** - 10.1 KB
  - Quick developer reference
  - Common commands
  - Troubleshooting tips

- **PORT_MANAGEMENT_GUIDE.txt** - 13.2 KB
  - Port configuration
  - Service routing
  - Deployment options

- **PRODUCTION_DEPLOYMENT_SUMMARY.txt** - 12.5 KB
  - Production ready checklist
  - Deployment strategies
  - Monitoring setup

## 📄 License

Proprietary - DREDGE Project

## 👥 Contributing

1. Create a feature branch
2. Commit with clear messages
3. Push and create pull request
4. Include documentation updates

## 📞 Support

For issues and questions:
1. Check documentation files
2. Review API_KEY_QUICK_REFERENCE.txt
3. Check troubleshooting section
4. Create GitHub issue

## 🔗 Links

- **GitHub**: https://github.com/QueenFi703/DREDGE-Cli
- **Documentation**: See documentation files in repository
- **API Docs**: http://127.0.0.1:9000/docs (when running)

---

## Version History

### v2.0.0 (Current)
- **NEW**: Unified Middleware authentication system
- **NEW**: Central gateway proxy (port 9000)
- **NEW**: Multi-port support (8000, 8001, 8080, 3001)
- Removed per-route dependency injection complexity
- Added comprehensive middleware documentation
- Production ready with error handling

### v1.0.0
- Initial API key system
- FastAPI dependency injection
- Basic rate limiting
- Usage tracking

---

**Status**: ✅ Production Ready

All components tested and deployed. Ready for production use with your existing infrastructure on ports 8000, 8001, and 8080.
