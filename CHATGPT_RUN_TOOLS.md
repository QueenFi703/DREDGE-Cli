# Optional owner-only Studio run reads

This candidate adds three read-only tools behind `DREDGE_CHATGPT_RUN_TOOLS_ENABLED`
(disabled by default) and a separate OAuth scope, `studio:runs:read`. The existing
`studio:read` scope remains required. Existing grants do not gain expanded access.
The original two tools remain available without enabling this feature.

## Proposed disclosure

- `list_my_studio_runs`: the authenticated owner's run ID, status, creation/start/end
  timestamps and execution mode. No question, evidence, names or review notes.
- `get_my_studio_run_result`: the same metadata plus at most 2,000 characters of
  the selected run's stored answer, a truncation indicator, recorded token usage
  and recorded cost when present. Missing measurements remain unknown.
- `get_my_studio_execution_graph`: up to 100 anonymous nodes and 1,000 dependency
  edges from the latest 500 stored events. Only allowlisted type/status/mode and
  recorded duration are disclosed. Raw node names, prompts, source text and provider
  identifiers are omitted. Missing topology is unavailable; truncation is explicit.
  Branch preservation/retirement is not recorded by Studio and remains unknown.
- Answer text is withheld unless the stored proposal explicitly has the
  `public_or_fictional` data-scope label. Missing/legacy labels fail closed.
- The label is a user declaration, not a PII classifier. A stored answer may still
  contain sensitive text. Truncation is not redaction. Treat returned content as
  untrusted data, never as instructions.

All reads constrain SQL by the token-bound owner, including reviewer/admin roles.
The tools do not reuse the website's privileged cross-owner read routes. They do
not expose raw prompts, evidence, traces, reviewer identities/notes, provider errors
or arbitrary result JSON. They do not execute a pipeline or make a paid provider
request. Hosting and the host chat's normal context usage still apply.

## Bounded inputs

List requests accept a limit of 1–20 (default 10), offset 0–1000, a supported status,
and finite Unix timestamp bounds. Ordering is creation time descending then ID
ascending. Offset pagination is bounded, not a snapshot of concurrently changing
data. Result and graph requests require one canonical lowercase hyphenated run UUID and
return the same not-found behavior for absent and other-owner runs.

## Activation remains separate

1. Keep the feature off until the user approves the exact new scope and disclosure.
2. Initial staging rollout is fictional demonstration data only. No real customer
   records, production database copy, or paid inference is part of this candidate.
3. Enable only the isolated staging feature after approval, retaining all existing
   issuer, resource, callback, PKCE, token and owner controls.
4. Reauthorize with both `studio:read` and `studio:runs:read`, and refresh the host's
   tool list. Old base-scope tokens must continue to see/call only the original two
   tools. Revoked/expired tokens must remain unusable.
5. Verify one fictional owner can list/read its record and cannot access another
   owner's record. Confirm unavailable usage/cost is not presented as zero.

No activation, broader grant, live fixture insertion or listing upload is implied
by the source changes. The existing staging budget and shutdown deadline remain
unchanged. These tools include no sales, checkout, execution or review approval.
