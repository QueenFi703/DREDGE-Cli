# Source and spending control foundation

Status: **offline-tested primitives, not enabled product controls**. No network
fetch, provider invocation, secret configuration, route, migration on app startup,
UI control, deployment or paid test is introduced. Existing execution behavior is
unchanged. The separate research-reuse change can integrate these primitives later.

## Source verification and changed-section planning

`Snapshot` records an exact-byte SHA-256 content hash with account/case scope,
source ID, observation time, collector version and a distinct observation receipt.
The helper does not fetch URLs or validate that a caller really observed a source.
Only a trusted, authorized server collector may construct these records. A search
citation, timestamp, HTTP error, model assertion or user-provided hash is not such
a receipt. Caller-controlled snapshots must never enter this seam directly.

`compare_source` returns `unchanged`, `changed`, or `unknown`. An unchanged result
requires a new observation during the requested verification interval, a different
receipt, matching source and collector version, and equal bytes. Account or case
mismatch rejects the comparison. Missing observations, reused receipts, collector
version changes and observations outside the verification interval stay unknown.
An old answer's age alone says nothing about whether its sources changed.

`plan_sections` takes a trusted, complete dependency map and source states. Sections
with changed dependencies are refresh candidates; sections whose dependencies are
all unchanged are reuse candidates. Any unknown or missing dependency blocks the
section. An empty dependency list also blocks. The mapping does not establish
claim correctness, completeness, current user intent, or permission to reuse.
Changes to instructions, question, scope, source-set membership, dependency-map
version, extraction semantics, or source identity require invalidating the old
plan before comparison. Human review remains required.

## Conservative preflight quotes

`quote_cost` requires a model-specific, versioned, currently valid server-verified
USD `PriceCard`, plus certified `Bounds` for all input tokens, all output tokens
(including reasoning), and all tool calls. Prices have no defaults and include a
tool-call charge; unavailable, mismatched or expired prices fail closed. Input
bounds must cover instructions, history, tool results and any provider-billed
retrieval context. Unknown retrieval/tool charges cannot be represented as zero.
Use separate quotes/adapters for heterogeneous tool costs, retries or multiple
calls; they must all be included before reserving the aggregate bound.

The result is a conservative upper quote in integer micro-USD, rounded up. It is
not a bill, measured savings, or a token estimate inferred from character count.
There is currently no trusted production price feed, tokenizer/bound certificate,
or charge reconciliation adapter wired to this module. Existing provider receipts
report `billing_unavailable`. Do not turn those receipts into actual USD spend.

## Reservations

`BudgetLedger` holds a user-selected cumulative cap per exact account/case scope.
JSON tuple encoding avoids delimiter collisions. A single SQLite database uses
`BEGIN IMMEDIATE` to serialize cap changes, reservation checks and reconciliation
across connections/processes. A limit cannot be lowered below spent-plus-held
funds. Unknown budgets fail closed. Only a new reservation returns `True`; a repeat
of the same run/quote returns `False` and must never trigger a second dispatch.
Changing a reserved run's quote rejects the request.

A timeout, cancellation after dispatch, missing receipt or uncertain charge keeps
the full reservation indefinitely. Only an authoritative final charge can settle
it and release the difference. A server-proven never-dispatched cancellation may
settle at zero, with a unique trusted receipt. Receipt IDs cannot be reused to
settle two runs. A detected charge above the reserved bound durably halts the scope
and retains its hold; increasing the limit does not remove that halt. There is no
automatic unfreeze or refund handler. These require explicit future reconciliation
work. Budgets are cumulative, not monthly subscriptions, prepaid balances or credits.

## Required integration before claiming enforceable live limits

1. Authenticate account and case authorization on every operation. Derive `Scope`
   from trusted server state; never accept another account/case or fabricated
   quote/receipt from a request. Bind source-set, dependency-map and run-payload
   versions to the approved immutable execution plan.
2. Add a policy-reviewed source collector and durable snapshots with safe network
   boundaries, authorization, extraction/version semantics and revalidation errors.
   It must actually reobserve sources; this module intentionally adds no fetcher.
3. Add trustworthy, revisioned pricing for every charge component and a certified
   complete input bound. If either is unavailable, block budget-controlled dispatch
   and explain why. Show the selected cap, quote and limitations before approval.
4. Persist the user's cap choice through an authenticated/CSRF-protected interface.
   Verify a current quote, reserve before paid dispatch, and enforce the very same
   input/output/tool limits at the provider boundary. Disable hidden retries and
   unbounded tool/context expansion. Include every execution path and worker.
5. Coordinate the separate-run dedup guard and budget reservation in a single
   durable execution state machine. This standalone ledger currently has its own
   transaction: it is not atomic with Studio's run or dedup transaction. Dedup
   before reservation prevents waste, but crashes must still retain safe holds;
   do not auto-retry a claimed reservation after a crash or dispatch uncertainty.
6. Use one correctly managed transactional database shared by every deployment
   worker. Independent per-instance SQLite files do not enforce a shared cap.
   Add migrations, authorized receipt reconciliation, durable dispatch IDs,
   operational halt handling and tests for crash/restart/recovery behavior.
7. Test authenticated route/provider integration and concurrent deployment workers
   before enabling. The local tests exercise primitives with fixture prices and
   observations; they do not validate live collection, actual charges or deployment.

No energy/carbon savings or avoided dollar/token quantities are computed. Such
claims require a real measured baseline and a defensible measurement method.

## Local validation

Run `PYTHONPATH=src python -m pytest tests/test_research_controls.py -o addopts='' -q`.
Tests cover source verification boundaries, scope/version tampering, unknown-source
blocking, pricing failure modes and rounding, parallel reservations, duplicate
claims, concurrent cap edits, persistent uncertainty holds, trusted zero charges,
overrun halts and repeated receipt rejection. No external services are called.
