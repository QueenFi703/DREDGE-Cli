# Thresh paid-access bridge — candidate for review

Credited by Sophia Cole. © 2026 Sophia Cole. ColeWorld™.

This optional integration links a Sites-authenticated Thresh subject to a DREDGE owner after both sign-ins and an explicit same-origin, CSRF-protected confirmation. It never matches accounts by email, replaces an existing owner pin, assigns Studio roles, reads investigations, or creates payments.

The bridge is disabled unless a new `THRESH_BRIDGE_KEY` is configured. While disabled, it registers no routes and creates no tables. Enabling it adds only `thresh_links` and `thresh_link_challenges` to the existing Studio store. Back up that store before enabling. No original tables are migrated.

After review and approval:

1. Deploy this candidate to the existing DREDGE application. Do not replace its database, OAuth clients, existing secrets, or Stripe configuration.
2. Create one strong server-only bridge secret and set the same value as `THRESH_BRIDGE_KEY` on DREDGE and `THRESH_ENTITLEMENT_KEY` on the existing Thresh Site. Do not disclose it in source, chat, or browser fields. This is not an OpenAI API key.
3. Set the Thresh Site's `THRESH_ENTITLEMENT_URL` to the deployed DREDGE origin plus `/api/thresh/entitlement`, after verifying that deployment. Do not use the session-only `/api/billing/status` route.
4. Verify the current billing environment. The default checker requires live-mode, configured billing and the existing account-bound subscription check; sandbox state cannot unlock tools. No live billing activation or charge is authorized by this document.
5. On the Thresh Site, use Link account. Sign in to DREDGE and explicitly confirm the displayed account. A five-minute one-use challenge binds the exact Site subject. Check authorized access, another subject, expired challenge, revoked link, unpaid status, and a subscription-state change.
6. Prepare an isolated reviewer journey and run all submission cases through the actual saved plugin. Local tests with synthetic subscription callbacks are not live OAuth/Stripe verification.

## Rollback

Clear `THRESH_BRIDGE_KEY` and redeploy DREDGE to disable the bridge, and remove the Thresh Site's two entitlement settings. The tools then deny access. Retain the new link tables for audit/recovery unless a separately authorized deletion is required. Original billing, login, private records, and GitHub-agent routes remain in place.

## Evidence and limits

Sixteen local tests cover authority authentication, exact Site/scope binding, login/consent/CSRF, paid versus unpaid callback results, fresh subscription checks, expiry, replay, ownership pins, revocation, and the original Studio login-return behavior. These use synthetic accounts and a temporary SQLite store. The full DREDGE integration suite and real website login/Stripe state have not been verified in this minimal test environment. Review CI before merge; request explicit approval before enabling the bridge on production.
