# Website billing and informational ChatGPT access

This candidate keeps all subscription transactions on the DREDGE website. The
ChatGPT integration remains read-only: no payment tool, purchase flow, upgrade
prompt, pricing widget, transactional link, or dormant commerce enable switch.

## Preserved boundaries

- The website founding plan remains **$19 USD per month**, for one signed-in
  account. Paid AI inference and public web research charges are excluded.
- Stripe Checkout and its customer portal remain available according to existing
  configuration. A separate PayPal Subscriptions adapter is disabled by default.
  This is not Stripe's PayPal payment method or an eligibility claim.
- The verified OAuth account identity selects entitlements. Tool arguments cannot
  select another account or payment provider. Only current configured-live,
  verified, active, unexpired subscription records indicate paid access.
- Subscription payments do not grant operator, reviewer or administrator roles.
  Existing owner-only report and stored-run reads retain their scope checks and
  are not newly paywalled.
- `get_dredge_access` provides the fixed `/subscription-info` URL. This generic
  page contains no checkout controls, upgrade prompts, transactional links,
  account state, forms, or scripts. It is separate from `/billing`.

## Live release gates

Code deployment and payment activation are separate decisions. Do not claim
provider verification merely because mocked tests or a browser return succeeded.

1. Review the exact combined patch and run its regression/security checks.
2. Release only to the specifically authorized existing environment. Do not create
   Railway PR preview services or duplicate the ChatGPT tunnel credential.
3. Verify the deployed commit and read-only plugin behavior without making a
   purchase, granting new OAuth scopes, or exposing payment links in tool output.
4. PayPal activation needs its own merchant eligibility review, secure credentials,
   matching environment plan/webhook configuration, and approved provider sandbox
   end-to-end tests (renewal, cancellation, failure, replay and second account).
5. Live charging requires explicit authorization and review of currency, recurring
   terms, taxes, refund restrictions and provider agreements. Do not auto-create
   products, plans, credentials, webhooks, or payment permissions during deployment.

## References checked October 13, 2026

- [OpenAI plugin guidelines](https://developers.openai.com/plugins/plugin-guidelines)
- [Stripe webhooks](https://docs.stripe.com/webhooks)
- [PayPal Subscriptions API](https://developer.paypal.com/docs/api/subscriptions/v1/)
- [PayPal webhook verification](https://developer.paypal.com/docs/api/webhooks/v1/#verify-webhook-signature_post)

Check current guidelines before public submission. A local implementation or
staging release is not OpenAI review acceptance or permission to sell in ChatGPT.
