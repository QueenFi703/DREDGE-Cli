# ChatGPT subscription integration review

Reviewed 2026-10-15 UTC (2026-10-14 America/Chicago). Implementation base:
`staging/chatgpt-tunnel-20261011`, commit `657c3367206da87620f0e12ee7323518d8297508`.
The website `main` branch does not contain this newer OAuth/MCP integration.
This review does not establish which commit or billing configuration is deployed.

## Current policy boundary

The [plugin guidelines](https://developers.openai.com/plugins/plugin-guidelines)
allow sign-in to existing paid accounts and use of included features. They prohibit
digital subscription sales, subscription-plan displays, upgrades, and direct links
to checkout or other transactional pages. Informational entitlement explanations
and informational links are allowed. The
[monetization reference](https://developers.openai.com/plugins/build/monetization)
describes generally available external checkout for eligible physical goods; moving
a digital subscription purchase outside ChatGPT does not by itself make it permitted.
Commerce protocol schema support is not approval for this plugin's subscription sales.

Accordingly, do not add a Checkout tool, Stripe portal-session tool, payment widget,
price/upgrade prompt, or link to `/billing` to the ChatGPT integration. A general
customer portal can expose transactional subscription changes; its use from ChatGPT
has not been established as permitted. Keep that existing website operation separate.
The user's account of support case 17010327 is consistent with this boundary; the
private support message was not independently retrieved for this review.

## Reuse already implemented

- `chatgpt_oauth.py`: existing account consent, CSRF/session binding, exact registered
  client and redirect, S256 PKCE, issuer/resource-bound hashed bearer tokens, expiry
  and revocation checks. Browser sign-in alone is not a plugin grant.
- `chatgpt_transport.py`: verifies tokens again at tool execution and binds the
  verified owner inside an isolated request context. Tool arguments cannot select
  the billing owner. Tool discovery is an explicit allowlist.
- `get_dredge_access`: uses `billing.subscription_access` for persisted, verified,
  active, unexpired live entitlements. Sandbox records never grant paid access;
  roles remain separate. Stored reads retain their existing scope requirements.
- Website `/api/billing/status`: includes prices and browser CSRF state, so it is
  not a suitable unfiltered MCP result. Reuse its underlying helpers, not its JSON.
- Website Checkout and Stripe portal: server-selected customer mapping, session
  authentication and CSRF checks. Signed webhook reconciliation controls access;
  a return URL cannot grant it. Preserve these operations without exposing them as
  plugin tools. The staging branch also has PayPal safeguards; do not duplicate them.
- `/subscription-info`: fixed informational page on the configured plugin issuer;
  tunnel routing excludes website billing endpoints and transactional UI.

## Improvement in this patch

The existing access tool adds explicit stored-record provenance, a false live-check
indicator, a false in-chat-purchase indicator, and separate Stripe/PayPal environment
values. The legacy `billing_environment` remains the Stripe environment for compatibility.
No endpoint, OAuth scope, provider request, entitlement rule, payment configuration,
or database schema is added or changed. `billing_configured` means configuration is
present; it does not mean billing is approved, deployed, or launched.

Stored status can lag a delayed or missing provider event. These new fields make
that limitation explicit; they do not reconcile subscriptions or prove webhook health.

## Existing customer access and next decision

Existing customers can connect the same DREDGE account and read their entitlement in
ChatGPT. They retain independent access to existing website billing management after
website sign-in. Do not pass bearer tokens, customer IDs, portal-session URLs, or browser
CSRF tokens into ChatGPT to bridge these surfaces. Only the informational page is
linked from the plugin. A dedicated non-transactional customer-support view could be
considered later; do not invent a policy exception for a transactional portal.

Before any separately approved live launch, verify the actual deployed commit,
persistent database, provider environment and account binding, signed event delivery,
cancellation/payment failure, duplicate checkout protection, and two-account isolation
in sandbox. Do not run live checkout to perform this review. Live billing, access-gate
changes, payment deployments and purchases require the user's separate approval.

## Local verification

`PYTHONPATH=src python -m pytest -q -o addopts='' tests/test_chatgpt_foundation.py tests/test_billing.py tests/test_paypal_billing.py tests/test_chatgpt_oauth.py tests/test_chatgpt_transport.py tests/test_chatgpt_tunnel.py`

Result: 281 passed, 2 skipped (socket-dependent tunnel checks), one existing Starlette
WSGI deprecation warning. Authlib 1.8.0, Stripe 15.6.0 and MCP 1.29.0 matched the
repository pins. Tests used fictional accounts, temporary databases and mocked payment
providers; no real payments or deployed entitlement checks were performed. The added
regression verifies mixed provider environments, provenance, disabled purchases, no
provider-client access and no database changes during the read.
