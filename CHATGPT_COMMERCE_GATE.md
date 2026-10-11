# Approval-gated commerce design

**Preparation only. No commerce tool, purchase route, upgrade link, or runtime
commerce enable switch is included in the ChatGPT integration.** Read-only access
can be tested independently; permission to sell digital subscriptions must not be
inferred from a working sign-in or tool review.

## Decisions already preserved

- Existing DREDGE website billing is the baseline: account-bound Stripe Checkout,
  currently $19 USD/month, and server-derived subscription status.
- Do not impose a new paid gate on the existing owner-only Studio activity report.
- Do not grant workspace roles through billing or OAuth consent.
- Do not claim PayPal availability: the current US Stripe account cannot enable
  Stripe's native PayPal method. Stripe's merchant-hosted adapter preview requires
  separate eligibility and recurring-subscription confirmation. A PayPal inquiry
  is not approval or proof that renewals/cancellations/webhooks are compatible.

## Required gates before a separately reviewed sales implementation

1. Written OpenAI category approval identifying the permitted DREDGE subscription
   offering, approved sales surface, required review flow, and any restrictions.
2. Processor confirmation for the exact payment flow. For PayPal, confirm US
   merchant eligibility, recurring Checkout/Billing, renewals, cancellations,
   failed payments, webhook accounting and fees. Do not force a payment-method
   string or add an unverified adapter in anticipation.
3. User authorization for actual live configuration, credentials, contractual
   acceptance and deployment. Follow secure credential and approval workflows.
4. Reviewed commerce specification covering product/price truth, recurring terms,
   taxes and currency, cancellation/refund terms, explicit purchase confirmation,
   account ownership, idempotency, webhook reconciliation, duplicate subscriptions,
   race/replay safety and audit records.
5. Separate security review and end-to-end sandbox validation, followed by the
   approved OpenAI submission/review process and explicitly authorized release.

## Proposed future boundary, not implemented

An approved commerce adapter would derive the buyer from the same validated
DREDGE identity, use server-selected pricing, and reuse the existing billing
ownership/reconciliation logic. Purchase authorization would be separate from
`studio:read`; a read token must never authorize a payment. A processor-authenticated
webhook would establish entitlement, never a browser redirect or tool argument.
Any sales tool would be registered only in a separately reviewed approved release.
The read-only integration contains no dormant purchase handler and cannot bypass
these gates.

The offering/review packet can be drafted while approval is requested, clearly
labeling approval-dependent behavior as planned rather than currently available.
