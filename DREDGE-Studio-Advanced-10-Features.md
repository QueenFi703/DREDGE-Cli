# DREDGE Studio Advanced - 10 Features Deployment Guide

## Overview

DREDGE Studio now includes **10 powerful advanced features** for complete model management, Swift development, security, and visualization. All features are accessible through a unified web dashboard.

**Access the dashboard:** `http://localhost:3001/advanced`

---

## 🚀 Quick Start

### Prerequisites
- DREDGE server running (`dredge-cli serve --port 3001`)
- Modern web browser
- Authenticated user

### How to Access
1. Start the DREDGE server:
   ```bash
   cd dredge-cli-repo
   dredge-cli serve --host 0.0.0.0 --port 3001 --debug
   ```

2. Open browser and navigate to:
   ```
   http://localhost:3001/advanced
   ```

3. Login if prompted

4. Explore features from the sidebar menu

---

## 📋 The 10 Features Explained

### 1. 🎯 Model Management Panel
**Location:** Sidebar → Models and Inference → Model Management

**Capabilities:**
- List all available models (Quasimoto 1D/4D/6D, ensemble, String Theory)
- Load models with custom configurations
- Run inference with parameter input
- Benchmark model performance (throughput, latency, memory)
- View model metadata and capabilities

**API Endpoints:**
- `GET /api/advanced/models/list` — List all models
- `POST /api/advanced/models/load` — Load model by ID
- `POST /api/advanced/models/<id>/inference` — Run inference
- `POST /api/advanced/models/<id>/benchmark` — Benchmark performance

**Use Cases:**
- Experiment with different Quasimoto models
- Evaluate model performance metrics
- Configure model parameters before inference

---

### 2. ⚙️ MCP Operations Console
**Location:** Sidebar → Models and Inference → MCP Operations

**Capabilities:**
- List all available MCP (Model Context Protocol) operations
- Execute operations with structured parameters
- Unified inference (DREDGE + Quasimoto + String Theory combined)
- String spectrum calculations
- Model loading via MCP

**Available Operations:**
1. `list_capabilities` — List models and operations
2. `load_model` — Load Quasimoto or String Theory model
3. `inference` — Run inference on loaded model
4. `string_spectrum` — Compute string vibrational spectrum
5. `unified_inference` — Combined DREDGE + models inference
6. `get_dependabot_alerts` — Fetch GitHub security alerts

**API Endpoint:**
- `GET /api/advanced/mcp/operations` — List operations
- `POST /api/advanced/mcp/execute` — Execute operation

**Example:**
```bash
curl -X POST http://localhost:3001/api/advanced/mcp/execute \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "unified_inference",
    "params": {
      "dredge_insight": "Digital memory must be human-reachable",
      "quasimoto_coords": [0.5, 0.5, 0.5],
      "string_modes": [1, 2, 3]
    }
  }'
```

---

### 3. 💡 Insight Lifting & DREDGE Processing
**Location:** Sidebar → Insights and Processing → Insight Lifting

**Capabilities:**
- Input text insights
- Process through DREDGE pipeline
- AI enhancement and transformation
- View processing results and confidence scores
- Track insight history
- See which models were applied

**API Endpoints:**
- `POST /api/advanced/insights/lift` — Lift an insight
- `GET /api/advanced/insights/history` — Get lifted insight history

**Response Example:**
```json
{
  "status": "lifted",
  "insight": "Digital memory must be human-reachable",
  "lifted_insight": "[Enhanced] Digital memory must be human-reachable with Quasimoto analysis",
  "processing_time_ms": 123.45,
  "models_applied": ["quasimoto_4d", "string_theory_10d"],
  "confidence_score": 0.89
}
```

---

### 4. 🛠 Swift Toolchain Integration
**Location:** Sidebar → Development → Swift Toolchain

**Capabilities:**
- **Build Swift CLI** — Compile DREDGECli with optimization flags
- **Run Swift Package** — Execute CLI with custom arguments
- **Swift Tests** — Run entire test suite (XCTest)
- **Build iOS MVP** — Compile DREDGEMVPApp (iOS library)
- **Swift REPL** — Interactive Swift shell

**API Endpoints:**
- `POST /api/advanced/swift/build` — Build CLI
- `POST /api/advanced/swift/run` — Run package
- `POST /api/advanced/swift/tests` — Run tests
- `POST /api/advanced/swift/ios-build` — Build iOS MVP
- `POST /api/advanced/swift/repl` — Execute REPL command

**Optimization Levels:**
- `-O` — Full optimization (fast, larger binary)
- `-Osize` — Optimize for size (smaller, slightly slower)
- `-Onone` — Debug mode (unoptimized, better debugging)

**Example Build Configuration:**
```json
{
  "optimization": "-O",
  "target": "DREDGECli",
  "build_time_estimated_seconds": 45
}
```

---

### 5. 🔒 Dependabot Alert Management
**Location:** Sidebar → DevOps and Security → Dependabot Alerts

**Capabilities:**
- Fetch all Dependabot security alerts for your repository
- AI-powered vulnerability explanations
- View severity levels (high/medium/low)
- Dismiss or reopen alerts with reasons
- Track remediation recommendations
- See version upgrade paths

**API Endpoints:**
- `GET /api/advanced/dependabot/alerts` — Fetch alerts
- `GET /api/advanced/dependabot/alerts/<id>/explain` — AI explanation
- `POST /api/advanced/dependabot/alerts/<id>/dismiss` — Dismiss alert

**Alert Response Example:**
```json
{
  "id": 1,
  "severity": "high",
  "package": "flask",
  "current_version": "2.0.1",
  "updated_version": "2.3.0",
  "description": "Flask security vulnerability",
  "risk_level": "high",
  "remediation": "Update to flask>=2.3.0",
  "impact": "Could allow remote code execution"
}
```

**Dismiss Reasons:**
- `fix_started` — Fix already in progress
- `inaccurate` — False positive alert
- `no_bandwidth` — Will address later
- `not_used` — Dependency not used
- `tolerable_risk` — Risk is acceptable

---

### 6. 🧪 API Endpoint Tester
**Location:** Sidebar → Testing and Visualization → API Tester

**Capabilities:**
- Browse all DREDGE API endpoints
- Construct requests (GET, POST, PUT)
- Send requests with custom headers/body
- View formatted responses
- Copy curl commands
- Test authentication
- API documentation access

**Testable Endpoints:**
- `/lift` — Lift insights (POST)
- `/health` — Health check (GET)
- `/api/advanced/models/list` — List models (GET)
- `/api/advanced/mcp/execute` — Execute MCP (POST)
- `/api/advanced/swift/build` — Build Swift (POST)
- And all other `/api/advanced/*` endpoints

**API Endpoint:**
- `GET /api/advanced/api-tester/endpoints` — List all endpoints
- `POST /api/advanced/api-tester/test` — Execute test

---

### 7. 📊 Container & Deployment Status
**Location:** Sidebar → DevOps and Security → Containers Status

**Capabilities:**
- Monitor Flask server (port 3001) and MCP server (port 3002)
- View container status (running/stopped/error)
- Real-time CPU and memory usage
- GPU utilization monitoring
- Uptime tracking
- Stream container logs
- Restart containers

**Metrics Displayed:**
- Container name and port
- Status (running/stopped/error)
- CPU percentage
- Memory usage (MB)
- GPU usage percentage (if available)
- Uptime in seconds

**API Endpoints:**
- `GET /api/advanced/containers/status` — Get all containers
- `GET /api/advanced/containers/<name>/logs` — Stream logs

**Example Response:**
```json
{
  "containers": [
    {
      "name": "dredge-flask",
      "port": 3001,
      "status": "running",
      "uptime_seconds": 3600,
      "cpu_percent": 0.5,
      "memory_mb": 256.32
    },
    {
      "name": "dredge-mcp",
      "port": 3002,
      "status": "running",
      "cpu_percent": 1.2,
      "memory_mb": 512.64,
      "gpu_percent": 45.0
    }
  ]
}
```

---

### 8. 📈 String Theory Visualization
**Location:** Sidebar → Testing and Visualization → Visualization

**Capabilities:**
- Compute string theory vibrational spectrum
- Interactive mode and dimension selectors
- Real-time spectrum generation
- Energy level visualization
- Modal amplitude display
- Dimension configurability (up to 26D superstrings)

**Parameters:**
- `max_modes` — Number of vibrational modes (1-64)
- `dimensions` — Dimensionality (10D for superstrings)

**API Endpoint:**
- `POST /api/advanced/visualization/string-spectrum` — Compute spectrum

**Response Format:**
```json
{
  "spectrum": {
    "modes": [0, 1, 2, 3, ...],
    "energies": [0.0, 0.1, 0.2, 0.3, ...],
    "amplitudes": [1.0, 0.5, 0.333, ...]
  },
  "max_modes": 10,
  "dimensions": 10
}
```

---

### 9. 🌊 Quasimoto Wave Function Plotter
**Location:** Sidebar → Testing and Visualization → Visualization

**Capabilities:**
- Plot Quasimoto wave functions in 2D
- Model selection (1D, 4D, 6D)
- Parameter configurability
- Time evolution visualization
- Export plot data as JSON/CSV
- Interactive amplitude adjustment

**Supported Models:**
- `quasimoto_1d` — 1D wave function
- `quasimoto_4d` — 4D spatiotemporal (can slice to 2D)
- `quasimoto_6d` — 6D high-dimensional (can slice to 2D)

**API Endpoint:**
- `POST /api/advanced/visualization/wave-function` — Generate plot data

**Plot Data Format:**
```json
{
  "model": "quasimoto_1d",
  "x": [0.0, 0.1, 0.2, ...],
  "y": [0.0, 0.05, 0.2, ...],
  "title": "Wave Function: quasimoto_1d",
  "xlabel": "Position",
  "ylabel": "Amplitude"
}
```

---

### 10. ✨ Code Generation & Templates
**Location:** Sidebar → Development → Code Generation

**Capabilities:**
- Generate boilerplate code from templates
- Multi-language support (Swift, Python, TypeScript)
- Customizable code generation
- Copy-paste ready snippets
- Project setup automation

**Available Templates:**

1. **Swift CLI Template** (`swift_cli`)
   - Swift command-line tool scaffold
   - Package structure
   - Main entry point

2. **Python Model Integration** (`python_model`)
   - Python model wrapper
   - Integration with DREDGE
   - Inference boilerplate

3. **MCP Client** (`mcp_client`)
   - Swift MCP client
   - Model loading and inference
   - String Theory integration

4. **API Client** (`api_client`)
   - TypeScript/JavaScript REST client
   - Automatic endpoint generation
   - Type definitions

**API Endpoints:**
- `GET /api/advanced/codegen/templates` — List templates
- `POST /api/advanced/codegen/generate` — Generate code

**Example Request:**
```bash
curl -X POST http://localhost:3001/api/advanced/codegen/generate \
  -H "Content-Type: application/json" \
  -d '{
    "template": "swift_cli",
    "config": {
      "project_name": "MyDREDGE",
      "author": "Your Name"
    }
  }'
```

---

## 📁 File Structure

```
dredge-cli-repo/
├── src/dredge/
│   ├── advanced_features.py          # Flask Blueprint with 10 features
│   ├── server.py                      # Updated Flask app (registers blueprint)
│   └── static/
│       └── advanced_dashboard.html    # Advanced features UI
├── DREDGE-Studio-Quick-Reference.md  # User guide
└── README.md
```

---

## 🔌 API Reference Summary

### Base URL
```
http://localhost:3001/api/advanced
```

### All Endpoints

| Feature | Method | Endpoint | Description |
|---------|--------|----------|-------------|
| Model Mgmt | GET | `/models/list` | List available models |
| Model Mgmt | POST | `/models/load` | Load model |
| Model Mgmt | POST | `/models/<id>/inference` | Run inference |
| Model Mgmt | POST | `/models/<id>/benchmark` | Benchmark model |
| MCP | GET | `/mcp/operations` | List operations |
| MCP | POST | `/mcp/execute` | Execute operation |
| Insights | POST | `/insights/lift` | Lift insight |
| Insights | GET | `/insights/history` | Get history |
| Swift | POST | `/swift/build` | Build CLI |
| Swift | POST | `/swift/run` | Run package |
| Swift | POST | `/swift/tests` | Run tests |
| Swift | POST | `/swift/repl` | REPL command |
| Swift | POST | `/swift/ios-build` | Build iOS MVP |
| Dependabot | GET | `/dependabot/alerts` | Get alerts |
| Dependabot | GET | `/dependabot/alerts/<id>/explain` | Explain alert |
| Dependabot | POST | `/dependabot/alerts/<id>/dismiss` | Dismiss alert |
| API Tester | GET | `/api-tester/endpoints` | List endpoints |
| API Tester | POST | `/api-tester/test` | Test endpoint |
| Containers | GET | `/containers/status` | Get container status |
| Containers | GET | `/containers/<name>/logs` | Get container logs |
| Visualization | POST | `/visualization/string-spectrum` | String spectrum |
| Visualization | POST | `/visualization/wave-function` | Wave function plot |
| Codegen | GET | `/codegen/templates` | List templates |
| Codegen | POST | `/codegen/generate` | Generate code |

---

## 🎨 UI Navigation

### Sidebar Sections

**Models and Inference**
- Model Management (Feature #1)
- MCP Operations (Feature #2)

**Insights and Processing**
- Insight Lifting (Feature #3)
- DREDGE Pipeline (overview)

**Development**
- Swift Toolchain (Feature #4)
- Code Generation (Feature #10)

**DevOps and Security**
- Dependabot Alerts (Feature #5)
- Containers Status (Feature #7)

**Testing and Visualization**
- API Tester (Feature #6)
- Visualization (Features #8 & #9)

---

## 🚀 Getting Started Examples

### Example 1: Load and Run a Model
```bash
# 1. List models
curl http://localhost:3001/api/advanced/models/list

# 2. Load Quasimoto 4D model
curl -X POST http://localhost:3001/api/advanced/models/load \
  -H "Content-Type: application/json" \
  -d '{"model_id": "quasimoto_4d", "config": {}}'

# 3. Run inference
curl -X POST http://localhost:3001/api/advanced/models/quasimoto_4d/inference \
  -H "Content-Type: application/json" \
  -d '{"parameters": {}}'
```

### Example 2: Lift an Insight
```bash
curl -X POST http://localhost:3001/api/advanced/insights/lift \
  -H "Content-Type: application/json" \
  -d '{
    "insight_text": "Digital memory must be human-reachable"
  }'
```

### Example 3: Build Swift CLI
```bash
curl -X POST http://localhost:3001/api/advanced/swift/build \
  -H "Content-Type: application/json" \
  -d '{"optimization": "-O"}'
```

### Example 4: Check Dependabot Alerts
```bash
curl http://localhost:3001/api/advanced/dependabot/alerts
```

### Example 5: Generate Swift Code
```bash
curl -X POST http://localhost:3001/api/advanced/codegen/generate \
  -H "Content-Type: application/json" \
  -d '{
    "template": "swift_cli",
    "config": {}
  }'
```

---

## 📚 Documentation Files

- **DREDGE-Studio-Quick-Reference.md** — Quick start and menu guide
- **DREDGE-Studio-Advanced-10-Features.md** — This comprehensive guide
- **README.md** — Main project documentation

---

## 🔐 Authentication

All `/api/advanced/*` endpoints require authentication. The dashboard (`/advanced`) will handle login redirection automatically.

To authenticate via API:
1. Login via `/auth/login`
2. Obtain session cookie
3. Include cookie in subsequent requests

---

## 🐛 Troubleshooting

**Issue:** Dashboard returns 404
**Solution:** Ensure `advanced_dashboard.html` exists in `src/dredge/static/`

**Issue:** API endpoints return 404
**Solution:** Restart Flask server with updated `server.py`

**Issue:** Features not loading in UI
**Solution:** 
- Clear browser cache
- Check browser console for errors
- Verify API is responding: `curl http://localhost:3001/api/advanced/models/list`

**Issue:** Swift build fails
**Solution:** 
- Verify Swift is installed: `swift --version`
- Check build logs from container
- Ensure correct directory structure

---

## 📈 Next Steps

1. **Test all 10 features** through the dashboard UI
2. **Integrate with CI/CD** — Add API calls to your pipeline
3. **Customize templates** — Edit code generation templates for your use case
4. **Monitor production** — Use Container Status for uptime monitoring
5. **Automate alerts** — Set up webhooks for Dependabot integration
6. **Export results** — Use API to fetch data for reporting

---

## 📞 Support

For issues or questions:
1. Check API logs: `docker logs <container_name>`
2. Review console: Browser DevTools → Console tab
3. Test endpoints directly with curl
4. Check GitHub issues on QueenFi703/DREDGE-Cli

---

**Version:** 1.0.0  
**Deployed:** 2026-05-25  
**Dashboard URL:** `http://localhost:3001/advanced`
