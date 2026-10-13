# Research and spending controls: runtime and plugin candidate

This follow-on adds usable opt-in spending limits to the existing reviewed Studio
execute path, supplied-source comparison, and bounded public-source observations.
See RESEARCH_REUSE.md for exact scope, pricing sources and unsupported capabilities.

## Interfaces

- GET/POST `/api/studio/budget`: authenticated owner-only cumulative settings and
  held/committed conservative upper bounds. Mutation requires operator/admin and
  existing session CSRF. No account ID, reset, or invoice estimate accepted.
- `/studio/research-controls`: spending form plus saved supplied-record comparison.
- POST `/api/studio/research/refresh-plan`: compares two own saved proposals.
- GET `/api/studio/research/public-sources`: exact five-URL public catalog.
- POST `/api/studio/research/revalidate`: own run ID and 1–3 distinct source IDs.
  Only URLs matching catalog entries are fetched; no arbitrary URL forwarding.
- GET `/api/studio/runs/<run_id>/source-observations`: own check history.
- `/studio/public-source-checks`: selection, results and history page.

## Enforcement

The exact no-tool request is prepared once before the run-start transaction and
sent once. Budget reservation, dedup reservation and run start are atomic.
Unknown, failed, interrupted or incomplete billing outcomes retain their hold.
Contradictory receipts retain the hold and halt protected generation. Complete
matching receipts reduce to a conservative usage upper bound, never actual cost.
Concurrent processes share the persisted SQLite ledger. Ledger entries cannot
be deleted/reset through application controls; disabling preserves earlier sums.
The scope excludes calls started while protection was disabled, even if the user
enables protection while such a call is still in flight.

## Source observations

Stored observations represent complete returned bytes at a point in time. A first
observation does not verify an older answer. Matching hashes are not evidence of
legal validity or claim correctness. Citation dependencies are provisional and
no answer is automatically rewritten or approved for reuse. Error outcomes remain
unknown. Static public catalog, pinned public DNS address, validated TLS hostname,
no redirects/auth/cookies/proxies, byte/text limits and a socket deadline constrain
retrieval. OS resolver calls can linger in two bounded worker slots; callers time
out and later checks fail closed if both slots remain occupied.

## Critical deployment/plugin boundary

The dedicated ChatGPT tunnel rejects these website routes. Publishing to that
staging image alone does not make these features user-accessible. A separately
verified normal website deployment is required. Do not widen the tunnel allowlist.
The plugin control extension is separately gated by
`DREDGE_CHATGPT_CONTROLS_ENABLED=true` (default false). It operates only through the
existing authenticated `/chatgpt/mcp` endpoint and does not widen tunnel routes.
Enabling the flag advertises the new schemas; it never upgrades existing tokens.
New explicit consent is required for `studio:budget:read`, `studio:budget:write`,
and `studio:research:write`. Own-run source tools also require existing
`studio:runs:read`. The budget update requires both budget scopes. The public
catalog does not expose account data and uses existing base read scope.

Control tool responses use bounded field allowlists. Source checks accept catalog
IDs and an owned public/fictional run ID, map IDs to approved saved source URLs on
the server and return observation/version facts plus provisional section IDs and
actions only, not raw source text, excerpts, answers or queries. Budget tools
require explicit USD micro-amounts and never start inference. Mutation tools have
honest MCP annotations; source checks disclose public network access. Existing
six read tools remain available without the new consent. Invalid input, denied
consent, revoked/expired tokens and cross-owner requests cannot upgrade access.

Rollout sequence: publish reviewed code; obtain approval for the default-off flag;
verify advertised tool schemas and old-token denials; have the user explicitly
approve only desired new scopes in OAuth consent; test read-only access and use
mutations only on their specific instruction. Do not accept consent or grant new
permissions for the user. A new paid execution or automatic rewrite tool is not
part of this candidate.

No paid calls, secret reads, scope grants or live state changes were performed.
