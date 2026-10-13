# DREDGE monthly billing — sandbox preparation

The founding plan is USD 19/month for one signed-in account. Hosted Checkout and the customer portal are server-created and account-bound. Card data stays with Stripe. Billing does not assign operator or reviewer roles. External inference/retrieval is not included.

## Railway configuration

Use the Cultivating Faith sandbox (acct_1TqQ57PdkvMfXnn1):

- STRIPE_BILLING_MODE=sandbox
- STRIPE_PRICE_ID=price_1UOjCGPdkvMfXnn1j72TmA0j
- STRIPE_PORTAL_CONFIG=bpc_1UOjCSPdkvMfXnn1bxXPLo07
- BILLING_ORIGIN=https://dredgeoriongateway.com
- STRIPE_RESTRICTED_KEY: set securely in Railway; never commit or send in chat.
- STRIPE_WEBHOOK_SECRET: signing secret for the endpoint below, set securely in Railway.

Restricted-key permissions: read Prices, Subscriptions and Invoices; write Customers, Checkout Sessions and Customer Portal Sessions. Verify the Dashboard permission names and SDK operations against Stripe before enabling.

Create a sandbox account webhook endpoint at https://dredgeoriongateway.com/api/billing/webhook using API version 2026-08-26.dahlia. Events: checkout.session.completed, checkout.session.async_payment_succeeded, customer.subscription.created, customer.subscription.updated, customer.subscription.deleted, invoice.paid, invoice.payment_failed. Do not subscribe to connected-account events.

## Verification before selling

Complete a sandbox checkout; verify a signed event stores the subscription for the signed-in account. Test payment failure, cancellation at period end, customer portal and a second account. Existing automated tests check signature rejection, account/environment binding, duplicate/reordered events, unpaid invoices, price mismatch, checkout reuse and CSRF.

The return URL does not grant access. The server reconciles the current subscription and paid invoice through verified events. Data uses the existing persistent SQLite volume. Sandbox cannot enforce paid access.

Live billing requires a separate live product, price, portal, restricted key and webhook signing secret, account readiness and an explicit live launch decision. BILLING_REQUIRE_SUBSCRIPTION=true opts into the live create/execute gate; browsing and role checks remain independent. No automatic tax is enabled; confirm tax requirements and registrations before launch.

## Provider-neutral website subscriptions

The subscription-access candidate adds PayPal independently of Stripe. Existing
Stripe tables and records are retained; PayPal uses separate provider-namespaced
tables in the same persistent SQLite database. Registration creates tables only.
It does not make provider calls, create products or webhooks, configure credentials,
enable charging, or change roles.

`billing.subscription_access(owner)` is the common account-bound entitlement
reader for the website gate and ChatGPT. Only active, unexpired records in each
provider's currently configured **live** environment grant paid access. Sandbox,
unpaid, expired, failed and trial records do not. The existing $19 plan excludes
paid AI inference and public web research usage. Existing stored reads remain
available according to their existing authorization scopes.

### PayPal preparation (disabled by default)

These are configuration names, not instructions to enable them automatically:

- `PAYPAL_BILLING_ENABLED=true` is required to enable the adapter.
- `PAYPAL_BILLING_MODE=sandbox` is the default; only `sandbox` and `live` are valid.
- `PAYPAL_CLIENT_ID` and `PAYPAL_CLIENT_SECRET` must be entered through an approved
  secure configuration flow. Never place either credential in this repository.
- `PAYPAL_PLAN_ID` must identify the matching environment's active plan: exactly
  $19 USD, every month, indefinitely, one regular cycle, no trial, no setup fee,
  no shipping or tax surcharge. A mismatch disables checkout.
- `PAYPAL_WEBHOOK_ID` is the provider-created webhook registration for
  `/api/billing/paypal/webhook` in the same environment and merchant app.
- `BILLING_ORIGIN` remains the trusted HTTPS website origin.

Subscribe the merchant webhook to the lifecycle/payment events declared in
`paypal_billing.EVENTS`. The adapter verifies the original event with PayPal's
signature-verification API and fixed environment origin; no supplied URL is
fetched. It retrieves current subscription, plan and transaction data before
updating entitlements. Initial activation alone is insufficient: a matching
completed payment and bounded paid period are required. Refund/reversal evidence
is retained so a delayed completed-payment response cannot restore access.

Provider callbacks are public signature-verified routes. Website checkout,
cancellation and reconciliation require an authenticated session and CSRF token;
they select only stored account-bound IDs and never accept a browser-provided
subscription ID. Return URLs cannot grant access. Cancellation stops future
PayPal renewals and retains only an already verified paid period. Suspension and
failed/unpaid state do not grant access; there is no assumed grace period.

### Recovery and duplicate prevention

Website **Refresh payment status** invokes the authenticated provider reconciliation
routes, recovering missed webhook state without starting a subscription. Stripe
reconciliation can clear an expired checkout. A pending or nonterminal agreement
on either provider blocks checkout on the other. Finish or cancel/reconcile that
agreement before changing providers.

PayPal reserves a durable server-selected request identity and opaque account
binding before the create call. Retries reuse its exact payload and request ID.
An uncertain create older than 71 hours requires merchant reconciliation; it is
never silently replaced after PayPal's documented idempotency-retention window.
Do not delete a pending intent to work around that guard. Resolve provider state
first through a separately approved operator procedure.

Stripe now also commits a durable reservation before creating its customer or
checkout, then persists the immutable checkout payload before sending it. An
uncertain request remains a cross-provider blocker and retries reuse the same
payload/idempotency key, including after a restart. Unresolved Stripe requests
older than 23 hours require merchant reconciliation before Stripe's minimum
24-hour idempotency retention can expire. Provider-returned session expiration,
or reconciliation of that exact session's terminated subscription, clears the
reservation; an old cancellation event cannot clear a newer checkout.

All webhook and reconciliation writes are serialized with SQLite `BEGIN IMMEDIATE`.
Provider IDs and events are scoped by provider and environment. Provider network
timeouts fail closed and return a retryable error; read tools never call providers.

### Deployment and verification limits

Keep the SQLite volume mounted and backed up. The schema extension is additive;
rollback code may leave the new tables in place. Do not delete payment records or
roll back entitlement data without a separate recovery plan.

Use the existing authorized staging service and tunnel only. Do not create new
Railway PR previews. A code rollout does not enable PayPal, create credentials,
change OAuth scopes, enable live charging, or prove a payment was processed.
Provider sandbox and live end-to-end tests require separate approval and setup.
See `CHATGPT_COMMERCE_GATE.md` for the read-only plugin boundary.
