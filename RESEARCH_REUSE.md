# Research reuse: bounded first release

## Offering copy

Find prior work before starting again. DREDGE's read-only plugin can search a
bounded page of your own completed Studio answer summaries and show candidates
within an age window you choose. Review the saved answer and its recorded usage
before deciding whether it is useful for your next task. Source freshness and
correctness still need checking; finding a candidate does not prove either.

## Implemented in this candidate

`find_my_saved_research` uses the existing optional run tools flag and both
`studio:read` and `studio:runs:read` OAuth permissions. There is no new scope,
provider request, paid execution, database mutation, or cross-account search.
An account's admin/reviewer workspace privileges never widen this owner-only read.

- Required `query`: 3–200 characters; case-insensitive literal match against only
  the already-exposable first 2,000 characters of saved answer text.
- `max_age_days`: integer 1–365; defaults to 7, measured from recorded completion.
  A valid completion timestamp, completion status, and `public_or_fictional`
  proposal label are required. This label is declarative, not automatic redaction.
- `limit`: 1–20 scanned completed runs, default 10. `offset`: 0–1000.
  The most recently created runs are checked first. A page can have no candidates
  and still have `has_more=true`; follow `next_offset`. Offset pages are not a
  stable snapshot during concurrent insertions. This is bounded discovery, not
  an exhaustive full-text index or semantic similarity engine.
- Answers remain untrusted data. Prompts, private evidence, raw traces, reviewer
  notes, and malformed stored JSON are not exposed. Unknown/legacy labels are
  excluded. Demo candidates stay labeled `demo`.
- Age and historical token/cost records are shown without inventing missing data.
  `freshness_status=sources_not_revalidated`, `savings_status=not_measured`, and
  `new_research_executed=false` make the limits explicit.

## Exact-input duplicate guard (existing Studio execution)

Separately proposed live Studio jobs now reserve an owner-scoped exact-input
fingerprint in the same SQLite write transaction that moves an approved run to
running. Within 24 hours, an identical request returns HTTP 409 with the same
owner's original run ID and makes no new provider call. Model, provider options,
instructions, declared data scope, source content/order, and all substantive
proposal fields are included. Only attachment bookkeeping timestamps are omitted.
Unknown or failed provider outcomes keep their reservation for the entire window.
A reviewer must still approve every distinct proposal; this does not bypass
consent, CSRF, roles, subscription checks, or existing attempt limits.

The guard covers reservations created after deployment, not historical runs,
Casework jobs, semantic near-duplicates, different provider configurations, or
other accounts. Demo execution is unchanged. After 24 hours a separately approved
identical job may execute. Stored completed results are not silently substituted.

The activity report exposes `duplicate_requests_blocked` (actual rejected
execution requests recorded since deployment). This is not proof of completed
provider calls, tokens, or money avoided. `savings_status=not_measured` remains
explicit. Historical provider-reported tokens continue to be reported separately.

## Requested capabilities still blocked / not implemented

These are requirements for the broader offering, not capabilities of this release:

1. **Freshness verification and changed-source-only refresh.** Studio does not
   fetch source URLs. Casework retains bounded web-search receipts but no source
   content snapshots, hashes, ETags, or claim-to-source refresh map. An age window
   cannot replace source revalidation. Safe retrieval, provenance storage,
   changed-section dependencies, failure handling, and tests are still needed.
2. **Pre-run token/cost estimates and user-selected hard spending limits.** The
   existing provider layer records usage when returned but reports cost as
   `billing_unavailable`. A reliable versioned price source and conservative
   enforceable maximum covering input, output, reasoning, and tool charges are
   needed. An estimate must not be represented as a hard cap. Unknown-outcome
   calls must retain budget reservations rather than silently release them.
3. **Savings measurement.** Showing historical tokens is not measuring avoided
   tokens or costs. A defined counterfactual, reuse decision/event accounting,
   a defined mapping from blocked attempts to avoided work, and provider cost reconciliation are still required.
   No token, dollar, energy, or carbon reduction is claimed.

## Verification and rollout

Offline tests exercise account isolation, existing text gates, malformed inputs,
scopes/feature flag, bounded paging, age boundaries, unknown/future timestamps,
demo labeling, and a real local OAuth/MCP request. No paid provider calls are
needed. Local tests are not evidence that a connected plugin has been deployed
or refreshed. Verify the deployed tool catalog and a safe owner-scoped call
before describing this feature as live.

## Follow-on candidate: runtime controls and public observations (2026-10-13)

This follow-on implements a limited portion of the above requirements. The
website `/studio/research-controls` exposes an opt-in, own-account cumulative USD
limit for subsequent protected Studio generation. It reserves full context at
verified worst-case Standard long-context cache-write/input and output rates,
without contacting a token-count endpoint or estimating prompt tokenization.
At 1,200 maximum output tokens, the reservation is $26.34; a $30 limit allows
one pending request. Complete matching receipts reduce held funds to a
usage-derived conservative upper bound. Neither figure is an invoice or a
claim about actual cost. Unknown outcomes retain the full reservation across
restarts. Pricing expires October 20, 2026 and then fails closed. Disabling the
limit permits uncapped new calls without resetting existing ledger entries.
Only the exact gpt-6-astra Standard no-tool request is supported. Prior/uncapped
calls, Casework, other accounts, infrastructure and taxes are outside the limit.

`/studio/public-source-checks` checks selected sources against five exact
allowlisted Missouri/federal public URLs, without prompts, account cookies or
credentials. Redirects and unsupported responses fail closed. DNS public-IP
validation/pinning, hostname-verified TLS, byte limits and an absolute socket
deadline protect retrieval. First observations establish a baseline; comparisons
record whole returned-byte differences and provisional paragraph review actions.
Cosmetic changes may trigger review. The saved answer is not rewritten, and
unchanged bytes do not establish legal validity or correctness. Ten checks/hour
per owner and at most three selected sources/check are enforced. Existing
libc DNS operations are bounded to two worker slots; stalled resolvers may keep
those slots busy until they return, while callers fail closed after three seconds.

Official price references, checked 2026-10-13:
- https://developers.openai.com/api/docs/models/gpt-6-astra
- https://developers.openai.com/api/docs/pricing

Still absent: exact pre-call token estimates, invoice reconciliation, automatic
changed-section generation, comprehensive source coverage and measured savings.
The optional plugin controls expose limited budget settings and public-source observations through separately consented scopes; existing read-only grants do not gain access. No paid inference or rewrite tool is added. See docs/STUDIO_RESEARCH_CONTROLS.md for the rollout boundary.
