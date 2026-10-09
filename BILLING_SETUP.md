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
