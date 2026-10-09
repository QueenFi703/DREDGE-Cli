# DREDGE Studio: public experience and governed workspace

## Implemented behavior

| Requested experience | Implementation |
| --- | --- |
| Public positioning and architecture | `/auth/login` and `/architecture`; responsive typography, enabled-provider states, accessible errors, explicit capability boundaries |
| Guided public preview | `/preview`; fictional library example, branching fixture graph, linked example source, no account data and no writes |
| Execution graph | `/advanced`; graph reads persisted node events from the real local DAG, including start/end times, dependencies, measured durations, and failed states |
| Source inspector | Immutable source title, URL, excerpt, provenance and verification status per proposal; submitted links are not fetched or automatically verified |
| Model/tool status | Configured simulation interfaces, disconnected external services, and actual storage/local-run observations; no synthetic health probes |
| Human approval | Operator proposes, a different reviewer approves/rejects with a reason, owner executes the approved immutable request once |
| Audit | Persistent, hash-linked proposal/review/execution events; database triggers prevent application updates or deletion; audit API verifies the stored hash chain |
| Roles | Server-assigned viewer/operator/reviewer/admin; owner-scoped runs and reports; instance-wide review/audit access for reviewers and admins |
| Usage/reliability/cost | Measured stored local run counts, success rate and mean duration; cost and token usage are explicitly unmetered |
| Mobile expanded/compact views | Responsive workspace; compact preference hides the graph and raw result while keeping selectable node summaries and source records |
| Session expiry | Idle and absolute limits, JSON 401 responses, accessible expired-session notice, same-account return to saved runs |

## Deployment configuration

This is one Studio instance, not a multi-tenant enterprise service. The reviewer/admin role has intentionally broad access to instance proposals and audit records. Grant these roles only to trusted instance members.

1. Keep a stable `SECRET_KEY` using the deployment's existing secret configuration. Without it, signed sessions are invalid after a restart. Never place secrets in the repository.
2. Mount persistent storage and set `STUDIO_DB_PATH` to an absolute path on that volume, for example `/data/studio.sqlite3`. The fallback instance-directory database is suitable for local use but may disappear when a container is replaced. A configured path alone does not prove that a volume is persistent.
3. Set `STUDIO_ROLES_JSON` to a JSON object keyed by exact OAuth IDs from `/auth/me`, such as `{"github:EXAMPLE_OPERATOR_ID":"operator","google:EXAMPLE_REVIEWER_ID":"reviewer"}`. Names and email addresses do not assign roles. Unlisted accounts are viewers. Role changes require administrator configuration; there is no browser endpoint for self-escalation.
4. Keep the configured GitHub/Google credentials and callback URLs. OAuth callbacks still return to `/advanced`. Sessions last at most 12 hours with a 30-minute idle timeout by default.
5. The existing Railway foreground ASGI gateway now mounts the Flask Studio after its API routes. `/health`, `/mcp`, `/invoke` and `/usage` remain gateway endpoints. The duplicate/broken MCP GET handler was removed; the original proxy route is retained.
6. Verify sign-in, public preview, two-account review, a completed trace, and volume persistence in the deployed environment before marking production rollout complete.

For a single-host installation, SQLite WAL and transactional status changes prevent double review/execution across local workers. Do not share this SQLite file across independent hosts or network filesystems. Move to a managed relational store before horizontally scaling. Back up the database using SQLite's backup API; treat it as private because it contains questions, source excerpts, user IDs, and review reasons.

## API workflow

All private APIs require the signed-in session. Get `/api/studio/session` for the role and CSRF token. POST requests need the `X-CSRF-Token` header.

1. `POST /api/studio/runs` with `query`, `pipeline_type` (`standard` or `ios_swift`) and optional `evidence` records (`title`, `url`, `excerpt`). Returns `pending_approval`.
2. A separate reviewer calls `POST /api/studio/runs/{id}/review` with `decision` (`approve` or `reject`) and `note`.
3. The owner calls `POST /api/studio/runs/{id}/execute`. The server atomically claims the approved run and records node transitions.
4. Read `GET /api/studio/runs/{id}` while it runs or afterward. Its graph is derived from stored events, not a reconstructed narrative.
5. Inspect `/api/studio/report`, `/api/studio/status` and (reviewer/admin) `/api/studio/audit`.

The previous `/api/architecture/pipeline/execute` path now directs callers to this approval workflow. Other private legacy API mutations require an operator/admin role and CSRF token. Legacy advanced/Dependabot responses disclose their simulated mode.

## Meaning of execution modes

- **Local:** Real computation or storage inside this server. This is not external AI inference.
- **Simulated:** Demonstration or placeholder behavior. Actual orchestration timing may be recorded around a simulated step, but it is not provider latency.
- **Live:** An external operation. External inference and retrieval are currently `not_connected`; none is fabricated or automatically enabled.

The DAG currently includes translation/normalization/cache placeholders. Passing through these steps does not establish factual correctness. Attached source records remain `user_supplied` and `unverified`. No provider tokens or dollar costs are estimated from word count. Audit hashes detect inconsistencies in the current chain; they are not a signed external compliance archive.

## Validation

Run `PYTHONPATH=src python -m pytest tests/test_studio.py tests/test_server.py -o addopts='' -q` and `node --check src/dredge/static/studio.js`.

For DOM interaction checks, install the existing package manifest's development dependencies with `npm install`, then run `npm run test:studio`. These tests exercise preview navigation, graph selection, safe source rendering, role-aware controls, session expiry and review submissions. They do not verify browser layout.

Contract tests cover approval gating, separate reviewers, duplicate decisions/execution, ownership, CSRF, source URLs, session expiry, store reopening, append-only audits, trace failures, real standard/iOS node order, honest reports and the ASGI-to-Flask mount. Browser rendering, real OAuth callbacks, deployment-volume persistence and provider billing integrations must be checked in their actual environments.

### Current evidence

| Acceptance area | Status | Evidence / limit |
| --- | --- | --- |
| Public pages, provider availability states, error copy | Confirmed at HTTP/DOM level | Public-page and login-error contract tests; real OAuth callbacks unverified |
| Guided preview, graph selection, source links | Confirmed at DOM level | Fixture has four branching nodes, source text cannot inject markup, guided navigation passes |
| Real local traces and node failures | Confirmed | Standard seven-node and iOS three-node execution tests; stored start/end events and measured durations |
| Approval, owner isolation, CSRF, role gates | Confirmed | Negative tests, concurrent execution claim test, immutable proposal workflow |
| Session expiry and record retention | Confirmed | Expired sessions return JSON 401 and do not delete stored runs; signed profile survives cache loss |
| Audit storage and consistency | Confirmed at application level | Store reopen, append-only trigger and hash-chain verification tests; external compliance archiving not implemented |
| Local usage and reliability | Confirmed | Completed/failed-run reporting and no-samples cases |
| Provider cost and tokens | Partial | Interface clearly reports unmetered/unavailable; provider billing is not integrated |
| Mobile expanded/compact layout | Partial | Responsive CSS and compact-view DOM interaction pass; cloud browser cannot reach the local server for visual testing |
| Production rollout | Unverified | Persistent volume, role assignments and deployed OAuth/runtime checks require deployment access |
