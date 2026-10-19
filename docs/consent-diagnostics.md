# Consent and prepared-redirect diagnostics

This change adds best-effort INFO events under `dredge.oauth.consent` for the
authorization consent handler. An isolated JSON-only stderr handler enables
these events under Uvicorn without changing global logging or access logs.
It does not repair a redirect, renew a grant,
change configuration, or establish that a browser completed the callback.

## Privacy boundary

The JSON message has a fixed schema, `dredge.oauth.consent.v1`. Its only fields
are `schema`, `event`, `request_id`, `method`, `outcome`, `commit_phase`,
`transaction_stage`, optional `age_seconds`, and optional numeric `status`.
For POST `http_response_prepared` only, three optional fields classify the
redirect: fixed-enum `response_kind` and boolean `callback_origin_matches` /
`callback_path_matches`. No raw header, query, URL or parameter value is emitted.
Every text value other than the ID is a fixed enum. Each HTTP request gets a
fresh random UUID unrelated to OAuth or user data. It is not returned to the
browser, persisted in the consent database, or accepted from request headers.

There are no owner/client IDs, credential hashes, OAuth state, authorization
codes, tokens, cookies, request bodies, parameters, URLs, redirect locations,
exception messages, or raw access logs. Numeric consent age is derived from the
existing expiration time and unchanged 600-second TTL, rounded down and bounded
to 0 through 86,400 seconds. It is not a user timestamp. Diagnostics do not create
a cross-request or cross-user identifier.

## How to interpret events

- `consent_marked_used`: the update ran inside the original open transaction.
  `commit_phase=pending` means it is not yet durable.
- `response_created`: Authlib produced a response, before database commit.
  The HTTP status is present. `allow_response` is not a claim that the callback
  or connection succeeded. In particular, Authlib can return a 302 carrying an
  OAuth error after an Allow decision. Inspect the prepared-response category
  and subsequent commit event, not the decision label alone.
- `transaction_committed`: the existing database context exited successfully.
  `transaction_stage` distinguishes initial consent creation from resolution.
  A rejected request can also commit a transaction containing no writes. Read
  the `outcome`; this event alone does not imply consent was granted.
- `transaction_failed`: response creation, database work, or context exit
  raised an exception. It does not claim that a commit was successful or that
  rollback definitely succeeded after an ambiguous storage failure.
- `http_response_prepared`: Flask prepared a response and applied its existing
  security headers. This is not a network-delivery or browser-navigation event.
  POSTs also classify the Location shape without emitting its value:
  - `code_present`, `error_present`, `code_and_error`, or `neither` describes
    query-key presence only. Empty values still count; validity is not checked.
    Both keys is ambiguous, not success. Code presence is not delivery, exchange
    or proof of usable portal credentials.
  - `no_location` means no Location header was present.
  - `unclassified` means duplicate Location/query keys, malformed or unsupported
    URL shape, a parsing bound, invalid configuration or a diagnostic failure.
    It is not an authentication rejection; the original response is untouched.
  - Match booleans compare parsed scheme/netloc and literal path independently
    against the registered callback. No host, port or path normalization is
    added. Query values are never decoded, retained in diagnostic state or logged.
    Booleans are omitted when classification is incomplete.

Classification is bounded to 8,192 characters per URL, 32 query fields, and
64 characters per encoded key. Only key names are percent-decoded. The query is
removed before calling the standard URL parser, so diagnostics do not add OAuth
query values to its cache. This does not change framework caching behavior or
claim to eliminate all in-memory URL retention. Relative,
non-HTTPS, credential-bearing, fragment-bearing and control/space-containing
URLs are conservatively unclassified. These rules affect diagnostics only, not
validation, redirect headers, response status or response body. Token endpoint
requests remain uninstrumented; missing events cannot prove no exchange occurred.

The same generic browser error can now be distinguished internally as
`missing_consent`, `expired`, `already_consumed`, or `binding_mismatch`. Other
fixed outcomes cover parameter validation, CSRF, login/account checks,
configuration changes, displayed consent, and allow/deny response creation.
Rejections before the consent handler (for example, ingress or blueprint
boundary checks) are deliberately outside this instrumentation.

Logging/ID-generation/serialization exceptions are swallowed without logging
their contents. Existing transaction exceptions still propagate normally.
The original lock, atomic commit, one-use behavior, TTL, PKCE, CSRF, binding,
response body, security headers and callback allowlist remain unchanged.

## Rollout is separate and requires approval

1. Review this patch against source commit
   `3028712403023aaf4cb1dd41372881b7ca35bff7` and rerun the local regression tests.
2. Confirm the target still uses that base, or rebase and review the delta.
   Obtain separate approval before pushing, opening a PR, merging or deploying.
3. Deploy only to the approved Studio service, keeping its existing environment,
   database, grant records, publication, CSP and disabled access logging intact.
   No schema migration, reseeding or new credentials are needed.
4. Confirm the logger's INFO level reaches the approved restricted log sink.
   Never enable raw access logs as a substitute. Apply existing access and
   retention policy to these anonymous operational events.
5. Any new browser authorization is a separate, explicitly authorized action.
   For that single attempt, use event order and numeric status to distinguish
   backend rejection/rollback from committed response preparation. Successful
   events cannot prove delivery beyond the application; use separately approved,
   sanitized browser/edge evidence if needed.
6. Roll back this source-only change if required. No database rollback is needed;
   do not delete consents, tokens or audit records as part of rollback.

Remaining limits: logging is best effort, synchronous logger I/O has some cost,
lost logs cannot prove a request never arrived, IDs cannot correlate distinct
HTTP requests, and this patch cannot identify an external redirect/browser
failure by itself. It must not justify weakening one-use or callback defenses.
