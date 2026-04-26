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
