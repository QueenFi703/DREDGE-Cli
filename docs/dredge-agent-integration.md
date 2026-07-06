# Dredge Agent Integration Blueprint

## Target Architecture

```text
GitHub (source of truth)
   ↓
Dredge (index + analysis layer)
   ↓
Agent API (tool surface)
   ↓
Your agents (Copilot / ChatGPT / custom)
```

Goal: any agent can ask *"What changed? What matters? What's risky?"* and Dredge returns structured intelligence.

## 1) Normalize Dredge into a Tool API

Use a clean, LLM-friendly API surface:

- `POST /query`
- `GET /repos`
- `GET /repos/{owner}/{repo}/summary`
- `GET /repos/{owner}/{repo}/changes`
- `GET /repos/{owner}/{repo}/risk`

Example request:

```json
{
  "repo": "owner/repo",
  "question": "What changed in the last 3 commits?",
  "depth": "semantic"
}
```

Example response:

```json
{
  "summary": "...",
  "files_changed": ["..."],
  "risk_score": 0.42,
  "insights": [{"type": "...", "value": "..."}]
}
```

## 2) Register Dredge as an OpenAPI Tool

- OpenAPI spec: `./dredge.openapi.json`
- Agent registration: `./m365agents.yml`

These files provide operation IDs and structured request/response contracts so agents can call Dredge deterministically.

## 3) Wire Tool Routing in Agent Config

In `m365agents.yml`, route repository questions to Dredge with capabilities and explicit instructions:

- `repo-analysis`
- `risk-detection`
- directive: **always call Dredge** for repository-change/risk questions instead of guessing

## 4) Connect GitHub → Dredge Ingestion

Keep Dredge fresh with a GitHub App webhook pipeline:

```text
GitHub App → webhook (push / PR) → Dredge index+analyze → storage/cache
```

Minimal handler shape (Node.js):

```ts
app.post("/webhooks/github", async (req, res) => {
  const event = req.headers["x-github-event"];

  if (event === "push") {
    const { repository, commits } = req.body;

    await dredge.index({
      repo: repository.full_name,
      commits,
    });
  }

  res.sendStatus(200);
});
```

## 5) Optional Chain for Better Outcomes

For highest quality responses:

1. Agent calls Dredge for facts.
2. LLM reasons over those facts.
3. Output combines grounded evidence + synthesis.

Dredge is the truth layer; the LLM is the interpretation layer.

## 6) Extended Capabilities to Add Next

Add the following to turn Dredge into a first-class cognitive layer:

- **Semantic search** (embeddings + vector database)
- **Cross-repo reasoning** (shared entities, dependency/risk propagation)
- **Auto-triggered PR reviews** with zero manual prompts

## 7) Pitfalls to Avoid

- Returning raw GitHub payloads instead of curated insights
- Overloading one endpoint with mixed concerns
- No caching (causes immediate latency issues)
- Letting the LLM guess when Dredge can answer directly

## 8) DREDGE Agent Descriptor (DAD) Pattern

For mobile/edge agents, define a signed, self-describing descriptor that lets orchestration reason about identity, capability, trust, and routing before dispatch.

Example `dad.yaml`:

```yaml
version: "1.0"

dad:
  id: "taskmaster-mobile-01"
  genesis: "DREDGE"
  class: "mobile-edge-agent"
  role: "observer-builder"
  spirit: "adaptive"

identity:
  name: "Fi"
  sigil: "queenfi"
  instance: "ios-crios-edge"
  fingerprint:
    ua: |
      Mozilla/5.0 (iPhone; CPU iPhone OS 26_2_0 like Mac OS X)
      AppleWebKit/605.1.15 (KHTML, like Gecko)
      CriOS/147.0.7727.47
      Mobile/15E148 Safari/604.1
    network:
      internal_ip: "172.17.17.160"
      trust_zone: "mesh"

capabilities:
  cognition:
    autonomous: true
    recursive: true
    reflective: true

  execution:
    shell: false
    api_calls: true
    workflow_dispatch: true
    edge_compute: limited

  perception:
    touch: true
    speech: true
    camera: true
    notifications: true

  rendering:
    webkit: true
    webgpu: limited
    animations: optimized

limits:
  battery_sensitive: true
  background_execution: restricted
  memory_class: mobile

routing:
  preferred_orchestrator: "taskmaster"
  failover:
    - "gatekeeper"
    - "oracle"
    - "dredge-shadow"

security:
  auth:
    provider: "github-oidc"
    token_rotation: true

  trust:
    signed_descriptors: true
    ua_verification: advisory
    workload_identity: enforced

behavior:
  on_connect:
    - sync_state
    - fetch_orders
    - hydrate_memory

  on_task:
    - validate_permissions
    - evaluate_capabilities
    - spawn_execution_chain

  on_failure:
    - snapshot_context
    - emit_trace
    - retry_with_fallback

scripture:
  genesis:
    - "Every node shall declare itself."
    - "Every task shall know its steward."
    - "No execution shall wander without memory."

  commandments:
    - "Trust is signed."
    - "Context is sacred."
    - "Agents adapt or decay."
    - "Telemetry without orchestration is noise."

mesh:
  topology:
    mode: "distributed-consciousness"
    discovery: "event-bus"
    heartbeat_interval: "30s"

observability:
  traces: true
  metrics: true
  logs:
    level: "adaptive"

future:
  evolution:
    self_describing_agents: true
    autonomous_routing: true
    memory_weaving: true
    multi_body_execution: true
```

Drop-in loader:

```python
from pathlib import Path
import yaml


class DAD:
    def __init__(self, path: str):
        self.path = path
        self.spec = yaml.safe_load(Path(path).read_text())

    @property
    def identity(self):
        return self.spec["identity"]

    @property
    def capabilities(self):
        return self.spec["capabilities"]

    def can(self, capability: str):
        section, key = capability.split(".")
        return self.capabilities.get(section, {}).get(key, False)

    def route(self):
        return self.spec["routing"]["preferred_orchestrator"]

    def scripture(self):
        return self.spec["scripture"]

    def summary(self):
        return {
            "id": self.spec["dad"]["id"],
            "class": self.spec["dad"]["class"],
            "route": self.route(),
            "autonomous": self.can("cognition.autonomous"),
        }


dad = DAD("dad.yaml")
print(dad.summary())
```

Interpreter pattern:

```python
def awaken(dad):
    if dad.can("cognition.autonomous"):
        spawn("autonomous-chain")

    if dad.can("execution.workflow_dispatch"):
        connect("github-actions")

    if dad.identity["fingerprint"]["network"]["trust_zone"] == "mesh":
        elevate("internal-routing")
```

This turns agent identity into executable orchestration policy: what the agent is, what it can carry, and where it belongs in the mesh.
