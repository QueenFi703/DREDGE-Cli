````markdown
# 🚀 Orion Gateway API

**The commercial infrastructure layer for DREDGE reasoning engines.**

Orion is a production-ready API gateway that turns DREDGE's reasoning capabilities into a scalable, monetizable service. It bridges your intelligence models with customer applications through a clean, rate-limited, billing-integrated API.

---

## 🎯 What Orion Does

```
┌─────────────────────────────────────────────────────────────┐
│                    CLIENT APPLICATIONS                       │
│            (SaaS / Agents / Integrations / UI)               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
           ┌─────────────────────────┐
           │   ORION GATEWAY API     │
           │  (This service)         │
           │  • Auth + API keys      │
           │  • Rate limiting        │
           │  • Usage tracking       │
           │  • Billing              │
           └────────────┬────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
    ┌────────┐    ┌──────────┐    ┌──────────┐
    │ Intent │    │ Reasoning│    │ Context  │
    │ Shaper │    │ Engine   │    │  Weave   │
    │ (Dolly)│    │ (Quas.)  │    │ (CWE)    │
    └────────┘    └──────────┘    └──────────┘
        │               │              │
        └───────────────┼──────────────┘
                        ▼
         ┌──────────────────────────────┐
         │  REASONING RESULT             │
         │  (Structured output)          │
         └──────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install fastapi uvicorn pydantic
```

### 2. Start the API Server

```bash
python -m dredge.orion_gateway
# or
python -c "from dredge.orion_gateway import run_orion; run_orion()"
```

Server starts on `http://localhost:3001`

### 3. Make Your First Request

```bash
curl -X POST "http://localhost:3001/invoke" \
  -H "x-api-key: demo-pro-key" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Summarize the key risks in this contract",
    "mode": "deep"
  }'
```

**Response:**
```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": 1697523600.123,
  "result": {
    "reasoning_depth": 4,
    "analysis": "[DEEP REASONING]: analyzed 'Summarize the key risks...'",
    "confidence": 0.87,
    "reasoning_path": "dre_engine"
  },
  "tier": "pro",
  "usage": {
    "requests_this_month": 15,
    "requests_limit": 10000,
    "remaining": 9985
  }
}
```

---

## 📖 API Reference

### Health Check (Public)

```bash
GET /health
```

Returns service status.

### Invoke Reasoning Engine

```bash
POST /invoke
Headers:
  x-api-key: your-api-key
  Content-Type: application/json

Body:
{
  "input": "Your prompt/text",
  "mode": "standard" | "deep" | "transform",
  "context": { ... },          // optional
  "metadata": { ... }          // optional
}
```

**Modes:**
- `standard` - Fast, deterministic processing
- `deep` - Multi-step reasoning with confidence scoring
- `transform` - Intent rewriting and enhancement

**Response includes:**
- `request_id` - Unique request identifier
- `result` - Reasoning engine output
- `usage` - Current quota consumption
- `tier` - User's subscription tier

### Get Usage Stats

```bash
GET /usage
Headers:
  x-api-key: your-api-key
```

Returns current month's usage breakdown.

### Get Available Tiers

```bash
GET /tiers
```

Returns pricing and tier information (public).

### Request Account Upgrade

```bash
POST /request-upgrade
Headers:
  x-api-key: your-api-key
```

Returns Stripe checkout link.

---

## 💳 Pricing Tiers

| Tier | Price | Requests/Month | Features |
|------|-------|-----------------|----------|
| **Free** | $0 | 100 | Standard mode, basic support |
| **Pro** | $29 | 10,000 | All modes, email support, usage API |
| **Enterprise** | Custom | 1M+ | Dedicated support, SLA, webhooks |

**Overages:** $0.05 per 1,000 additional requests (Pro tier only)

---

## 🔑 Authentication

### Test API Keys

Pre-configured for testing:

```bash
# Free tier (100 requests/month)
export API_KEY="demo-free-key"

# Pro tier (10,000 requests/month)
export API_KEY="demo-pro-key"

# Enterprise tier (unlimited)
export API_KEY="demo-enterprise-key"
```

### Generate Real API Keys

Using the Orion CLI:

```bash
# Create organization
python -m dredge.orion_cli create-org \
  --name "Acme Corp" \
  --email "api@acme.com" \
  --tier pro

# Output includes generated API key
```

---

## 🛠️ Configuration

### Environment Variables

```bash
# API
ORION_HOST=0.0.0.0
ORION_PORT=3001
ORION_DEBUG=false

# Database (Postgres)
DATABASE_URL=postgresql://user:password@localhost:5432/orion_db

# Cache (Redis)
REDIS_URL=redis://localhost:6379/0

# Stripe (Billing)
STRIPE_SECRET_KEY=sk_test_YOUR_KEY
STRIPE_PUBLIC_KEY=pk_test_YOUR_KEY
STRIPE_WEBHOOK_SECRET=whsec_YOUR_SECRET

# Auth
JWT_SECRET=your-super-secret-key
JWT_ALGORITHM=HS256
```

### PostgreSQL Schema (Production Baseline)

Use the following schema as a starting point for production deployments.

```sql
-- Organizations
CREATE TABLE organizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    tier VARCHAR(50) NOT NULL CHECK (tier IN ('free', 'pro', 'enterprise')),
    api_key VARCHAR(255) UNIQUE NOT NULL,
    requests_limit INT NOT NULL,
    billing_email VARCHAR(255),
    stripe_customer_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- API Keys
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id UUID NOT NULL REFERENCES organizations(id),
    key VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255),
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    last_used TIMESTAMP
);

-- Request Logs
CREATE TABLE request_logs (
    id BIGSERIAL PRIMARY KEY,
    org_id UUID NOT NULL REFERENCES organizations(id),
    request_id UUID NOT NULL,
    mode VARCHAR(50) NOT NULL,
    input_length INT,
    status VARCHAR(50),
    latency_ms INT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Usage Tracking
CREATE TABLE usage (
    id BIGSERIAL PRIMARY KEY,
    org_id UUID NOT NULL REFERENCES organizations(id),
    year_month VARCHAR(7) NOT NULL,
    requests_count INT DEFAULT 0,
    tokens_consumed INT DEFAULT 0,
    UNIQUE (org_id, year_month)
);

-- Billing Events
CREATE TABLE billing_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id UUID NOT NULL REFERENCES organizations(id),
    event_type VARCHAR(50) NOT NULL,
    amount DECIMAL(10, 2),
    invoice_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Recommended indexes
CREATE INDEX idx_organizations_tier ON organizations(tier);
CREATE INDEX idx_api_keys_key ON api_keys(key);
CREATE INDEX idx_request_logs_org_id ON request_logs(org_id);
CREATE INDEX idx_request_logs_created_at ON request_logs(created_at);
CREATE INDEX idx_usage_org_month ON usage(org_id, year_month);
```

### Local Development with Docker

```bash
# Start services
docker-compose up -d

# Initialize database
python -m dredge.orion_cli init-db

# Server runs on port 3001
curl http://localhost:3001/health
```

---

## 📊 Monetization Path to $10K/Month

### Scenario Analysis

**Option 1: Individual Developers (Recommended)**
- 350 Pro users × $29/mo = $10,150/mo
- Time to market: 3-6 months
- Requires: Content marketing, organic growth

**Option 2: SaaS Integrations**
- 10-20 SaaS companies integrate Orion
- Custom pricing: $500-5K/mo per integration
- 15 × $700/mo average = $10,500/mo

**Option 3: Enterprise Contracts**
- 1-3 enterprise customers
- $5K-20K/mo per customer
- 2 × $5K/mo = $10,000/mo

**Option 4: Blended (Most Realistic)**
- 200 Pro users: $5,800/mo
- 3 small enterprise: $4,500/mo
- Support contracts: $1,000/mo
- **Total: $11,300/mo**

### Customer Acquisition Math

| Metric | Free | Pro | Enterprise |
|--------|------|-----|------------|
| CAC | $0-5 | $50-100 | $500-2000 |
| LTV | $0 | ~$70 | ~$48K |
| Payback | N/A | 2-4 mo | 3-5 mo |
| LTV:CAC | N/A | 1.4-1.7x | 5-10x |

---

## 🚀 Deployment

### Fly.io (Recommended for startups)

```bash
# Deploy instantly
flyctl launch
flyctl deploy

# Access at https://orion-gateway.fly.dev
```

### AWS (ECS + RDS)

```bash
# Using Terraform
terraform init
terraform apply -var="aws_region=us-east-1"
```

### Kubernetes (GKE / EKS)

```bash
# Deploy
kubectl apply -f k8s-manifest.yaml

# Check status
kubectl get pods -l app=orion-gateway
```

See [ORION_DEPLOYMENT.md](ORION_DEPLOYMENT.md) for complete infrastructure setup.

---

## 📈 Analytics & Monitoring

### Key Metrics to Track

**Acquisition:**
- Sign-ups per day
- Conversion rate (free → paid)
- CAC by channel

**Retention:**
- Monthly active users (MAU)
- Churn rate by tier
- Cohort retention curves

**Revenue:**
- MRR (monthly recurring revenue)
- ARPU (average revenue per user)
- LTV (customer lifetime value)

**Product:**
- API latency (P50, P95, P99)
- Error rate
- Request volume by mode
- Cache hit rate

### Recommended Services

- **Error tracking:** Sentry
- **Performance monitoring:** Datadog / New Relic
- **Log aggregation:** Logtail / CloudWatch
- **Analytics:** Segment / Amplitude

---

## 🔗 Integration Examples

### Python Client

```python
import requests

class OrionClient:
    def __init__(self, api_key: str, base_url="http://localhost:3001"):
        self.api_key = api_key
        self.base_url = base_url
    
    def invoke(self, input_text: str, mode: str = "standard"):
        resp = requests.post(
            f"{self.base_url}/invoke",
            headers={"x-api-key": self.api_key},
            json={"input": input_text, "mode": mode}
        )
        return resp.json()

# Usage
client = OrionClient("your-api-key")
result = client.invoke("Analyze this contract", mode="deep")
print(result["result"]["analysis"])
```

### JavaScript Client

```javascript
class OrionClient {
  constructor(apiKey, baseUrl = "http://localhost:3001") {
    this.apiKey = apiKey;
    this.baseUrl = baseUrl;
  }

  async invoke(input, mode = "standard") {
    const resp = await fetch(`${this.baseUrl}/invoke`, {
      method: "POST",
      headers: {
        "x-api-key": this.apiKey,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ input, mode }),
    });
    return resp.json();
  }
}

// Usage
const client = new OrionClient("your-api-key");
const result = await client.invoke("Analyze this", "deep");
```

---

## 🎛️ CLI Commands

```bash
# Start API server
python -m dredge.orion_cli serve --port 3001 --debug

# Initialize database
python -m dredge.orion_cli init-db

# Create organization
python -m dredge.orion_cli create-org \
  --name "Company" \
  --email "api@company.com" \
  --tier pro

# Interactive demo
python -m dredge.orion_cli demo

# Show configuration
python -m dredge.orion_cli info
```

---

## 🧪 Testing

```bash
# Run tests
pytest tests/test_orion_gateway.py -v

# With coverage
pytest tests/test_orion_gateway.py --cov=src.dredge.orion_gateway

# Specific test
pytest tests/test_orion_gateway.py::TestInvoke::test_invoke_deep_mode -v
```

---

## 🛡️ Security Best Practices

1. **API Keys**
   - Rotate regularly
   - Use strong random generation
   - Never commit to git
   - Implement key rotation policies

2. **Database**
   - Use parameterized queries (prevents SQL injection)
   - Enable encryption at rest
   - Regular backups
   - Firewall to only API server

3. **Redis**
   - Enable authentication
   - Use TLS in production
   - Regular backups
   - Firewall access

4. **CORS & Headers**
   - Restrict CORS origins in production
   - Implement rate limiting per IP
   - Add security headers
   - HTTPS only in production

---

## 🚧 Advanced Features (Roadmap)

- [ ] Webhook events (invoice, upgrade, downgrade)
- [ ] Custom reasoning engine plugins
- [ ] Analytics dashboard
- [ ] Team management
- [ ] API usage export
- [ ] Custom domain support
- [ ] SSO (Google, GitHub)
- [ ] Batch processing API
- [ ] Async job queue
- [ ] Agent marketplace

---

## 📞 Support

- **Documentation:** [docs/ORION_DEPLOYMENT.md](ORION_DEPLOYMENT.md)
- **Issues:** [GitHub Issues](https://github.com/QueenFi703/DREDGE-Cli/issues)
- **Email:** support@dredge.dev
- **Status:** [Uptime Dashboard](https://status.dredge.dev)

---

## 📄 License

MIT — See LICENSE for details

---

**Built with 🧠 + 💫 by DREDGE**

*Reasoning infrastructure for the intelligence layer.*
````
