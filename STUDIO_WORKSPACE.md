# DREDGE Studio: public experience and governed workspace

## Implemented behavior

| Requested experience | Implementation |
| --- | --- |
| Public positioning and architecture | `/auth/login` and `/architecture`; responsive typography, enabled-provider states, accessible errors, explicit capability boundaries |
| Guided public preview | `/preview`; fictional library example, branching fixture graph, linked example source, no account data and no writes |
| Execution graph | `/advanced`; graph reads persisted node events from the real DAG, including live provider response identifiers and usage, including start/end times, dependencies, measured durations, and failed states |
| Source inspector | Immutable source title, URL, excerpt, provenance and verification status per proposal; submitted links are not fetched or automatically verified |
| Model/tool status | Configured Astra model and historical live successes, separate agency-gated Casework state, explicit demonstration adapters, and local storage observations; no synthetic health probes |
| Human approval | Operator proposes, a different reviewer approves/rejects with a reason, owner executes the approved immutable request once |
| Audit | Persistent, hash-linked proposal/review/execution events; database triggers prevent application updates or deletion; audit API verifies the stored hash chain |
| Roles | Server-assigned viewer/operator/reviewer/admin; owner-scoped runs and reports; instance-wide review/audit access for reviewers and admins |
| Usage/reliability/cost | Stored run counts, success rate and mean duration; provider-reported token usage, partial-usage warnings, and actual call attempts; billed cost remains unavailable |
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

1. `POST /api/studio/runs` with `query`, `execution_mode` (`live` or `demo`), `pipeline_type` and optional `evidence` records (`title`, `url`, `excerpt`). New requests default to `live`. Live requests require `consent: true`, `public_data_confirmed: true`, and `pipeline_type: "standard"`; only public or fictional input is allowed. The server snapshots the configured model and 1,200-output-token ceiling. Returns `pending_approval`. Explicit `demo` supports `standard` or `ios_swift` without provider calls. Historical proposals without a mode retain their approved local/demo behavior.
2. A separate reviewer calls `POST /api/studio/runs/{id}/review` with `decision` (`approve` or `reject`) and `note`.
3. The owner calls `POST /api/studio/runs/{id}/execute`. The server atomically claims the approved run, enforces 20 live attempts per account per rolling 24 hours, and records node transitions. A changed configured model requires a new proposal. Live execution prepares the approved question/evidence, issues one OpenAI Responses request, and inspects source IDs. No retries, external tools, or source-URL fetches occur.
4. Read `GET /api/studio/runs/{id}` while it runs or afterward. Its graph is derived from stored events, not a reconstructed narrative. Share `/advanced?run={id}` only with an existing authorized reviewer. The link selects the exact run, retains its canonical UUID through OAuth sign-in, and does not grant access or fall back to a different run when access is denied.
5. Inspect `/api/studio/report`, `/api/studio/status` and (reviewer/admin) `/api/studio/audit`.

The previous `/api/architecture/pipeline/execute` path now directs callers to this approval workflow. Other private legacy API mutations require an operator/admin role and CSRF token. Legacy advanced/Dependabot responses disclose their simulated mode.

## Meaning of execution modes

- **Local:** Real computation or storage inside this server. This is not external AI inference.
- **Simulated:** Demonstration or placeholder behavior. Actual orchestration timing may be recorded around a simulated step, but it is not provider latency.
- **Live:** A real OpenAI Responses request using the existing server-side `OPENAI_API_KEY`. `OPENAI_MODEL` selects the operator-configured model (default `gpt-6-astra`). The status panel distinguishes no credential, configured but unverified, and a historical successful request. It never treats configuration alone as proven availability. Casework web research remains a separate agency-gated workflow.

The explicit demonstration DAG includes translation/normalization/cache placeholders. The live DAG does not use those placeholders. Passing through these steps does not establish factual correctness. Attached source records remain `user_supplied` and `unverified`. Live token counts are copied from provider usage. The response ID, request ID, actual returned model, and usage are retained; incomplete responses retain usage but never become successful answers. Unknown usage stays unknown, including after a timeout; failed requests may be billed. Actual dollar cost is not inferred from token counts. The UI calls every answer a draft requiring human review, and flags unknown source IDs without claiming factual verification. Audit hashes detect inconsistencies in the current chain; they are not a signed external compliance archive.

## Validation

Run `PYTHONPATH=src python -m pytest tests/test_studio_live.py tests/test_studio.py tests/test_server.py -o addopts='' -q` and `node --check src/dredge/static/studio.js`.

For DOM interaction checks, install the existing package manifest's development dependencies with `npm install`, then run `npm run test:studio`. These tests exercise preview navigation, graph selection, safe source rendering, role-aware controls, session expiry and review submissions. They do not verify browser layout.

Contract tests cover approval gating, separate reviewers, duplicate decisions/execution, ownership, CSRF, source URLs, session expiry, store reopening, append-only audits, trace failures, real standard/iOS node order, immutable live provider payloads, one-shot/concurrent execution, consent, scope, provider errors/timeouts/incomplete results, usage accounting, safe answer rendering, honest reports and the ASGI-to-Flask mount. Browser rendering, real OAuth callbacks, deployment-volume persistence and provider billing integrations must be checked in their actual environments.

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
| Provider response, usage and failures | Confirmed with mocked HTTP boundary | 38 live backend contract cases plus live form/answer DOM checks. No paid API call is part of tests. Real deployed model access remains unverified until a bounded live check. |
| Provider billing | Partial | Tokens come from provider responses; missing usage remains explicit. Actual billed dollars and infrastructure cost are not integrated. |
| Mobile expanded/compact layout | Partial | Responsive CSS and compact-view DOM interaction pass; cloud browser cannot reach the local server for visual testing |
| Production rollout | Unverified | Persistent volume, role assignments and deployed OAuth/runtime checks require deployment access |

## Live-operation boundaries

Studio proposals are public/fictional only. Do not place real client information here; use the encrypted, agency-authorized Casework workflow, which retains its independent data-approval and per-request consent gates. A checkbox is not permission to bypass agency policy. Quasimoto, String Theory, translation demonstrations and iOS demo nodes are not converted into real trained models by this integration. MACSS, automated eligibility/enforcement decisions, legal filings, and external messages remain unsupported.

Each live request uses `store: false`, standard service tier, low reasoning, at most 12,000 input characters plus fixed instructions/JSON overhead, and 1,200 output tokens including reasoning. No automated provider retries occur. A timeout may leave a charged request running at the provider; verify the provider project before manually creating another proposal. The app does not retrieve provider billing.

Official provider reference, checked 2026-10-10: https://developers.openai.com/api/docs/models/gpt-6-astra and https://developers.openai.com/api/reference/resources/responses/methods/create . Model availability for the deployment's project must still be verified using its existing credential in place. Never print, export or commit that credential.
