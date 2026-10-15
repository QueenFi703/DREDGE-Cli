# Isolated staging reviewer login

Default off. This candidate contains no live credentials and does not authorize
account provisioning, credential disclosure or deployment. Existing GitHub and
Google logins remain separate. Reviewer identity is `review:<digest>`, never an
invented provider account ID.

## Boundaries

- Enabled only on the existing dedicated staging tunnel at
  `https://chatgpt-staging.dredgeoriongateway.com`. Ordinary production cannot enable
  it merely by setting the review flag.
- The review identity has operator capabilities only within its own separate
  fictional-data store. It is never Studio admin or the cross-account website
  reviewer role. All other website routes are denied for review browser sessions.
- Existing OAuth consent, S256 PKCE, exact callback matching, issuer/resource and
  scope checks remain mandatory. A password session alone does not authorize MCP.
- The same 11 existing tools operate on owner-bound fixture data. Budget changes
  and approved public source checks are real changes inside that isolated store.
  Source checks can fetch the fixed approved public catalog pages. No run execution,
  case data access, paid entitlement or payment is available to this identity.
- Fixtures are labeled manually authored demonstrations. Completion time records
  fixture preparation, not provider execution. Usage/cost remain unknown; no
  synthetic execution graph is presented as measured. Seeded budget is enabled at
  zero USD and never reset on restart.

## Configuration after separate approval

The deployment's normal stable session key, dedicated tunnel and OAuth configuration
must already exist. Do not paste secrets into source, shell arguments, logs or chat.
Supply through the approved secret-management mechanism:

- `DREDGE_REVIEWER_LOGIN_ENABLED=true` (absent or `false` registers no login).
- `DREDGE_REVIEWER_USERNAME`: dedicated 8–64 character ASCII letters/digits/_/-.
- `DREDGE_REVIEWER_PASSWORD_HASH`: Werkzeug `scrypt:32768:8:1` hash with at least
  16 salt characters. Generate a strong unique password only after approval and
  deliver it securely. This module never generates or reveals plaintext passwords.
- `DREDGE_REVIEWER_GENERATION`: positive, monotonically increasing integer.
- `DREDGE_REVIEWER_EXPIRES_AT`: Unix timestamp, no more than 30 days ahead.
- `DREDGE_REVIEWER_DB_PATH`: absolute path on a durable volume, distinct from the
  ordinary Studio DB and not a symlink/hardlink alias. Use a new empty path or the
  same previously initialized review-only database; unmarked databases are rejected.

All workers and generations MUST use the same durable reviewer DB path. Never
change that path to rotate credentials or evade throttling/revocation. The initial
empty store is marker-checked, schema-initialized and seeded with two fictional
records. Existing review stores are not wiped or reset. Do not copy production
records into this database. Configure volume permissions and backup handling through
separately approved operations, not public application routes.

The global durable throttle allows five password attempts per five minutes across
all sessions/workers, including unknown usernames. This intentionally trades some
availability for bounded offline review access. A malicious party can temporarily
lock out this single account; an operator should investigate instead of disabling
throttling. CSRF, request size, exact origin and form-field checks precede password
verification. Sessions are renewed on login and cookies are Secure/HttpOnly/Lax.

## Revocation, rotation and expiry

An approved administrator can invoke the internal `revoke_review_access(app)` helper
in the configured service context. It marks the current identity revoked in shared
storage, invalidating browser sessions, unredeemed authorization codes and issued
MCP tokens on their next authorization check. There is deliberately no public
provision/revoke endpoint. This document is not permission to invoke the helper.

For approved password rotation, use a new strong password/hash and increment the
generation. Provisioning a higher generation atomically revokes all previous ones;
running older workers immediately reject old sessions and tokens. Old-generation
startup and same-generation credential changes are rejected. Reusing the same
review DB prevents rollback from restoring revoked access. Never restore an old DB
backup as a recovery shortcut; it may restore revoked state and requires a separate
reviewed invalidation plan.

Natural expiry invalidates the reviewer without affecting ordinary users. Restart
with an already expired positive timestamp leaves the review feature disabled and
does not open or seed its store. Disabling the feature in configuration rejects its
grants once all workers have restarted; use durable revocation for immediate
cross-worker cutoff. Already-in-flight actions cannot be undone by revocation.

## Verification and remaining release gates

Offline tests use explicit fictional credentials only. They cover real local
password-to-OAuth-to-MCP transport, all 11 tools (public-source retrieval stubbed),
owner/store isolation, legacy-route denial, role clamp, CSRF, malformed inputs,
rate limits, expiry, cross-worker revocation/rotation and rollback rejection.
Existing OAuth/PKCE/redirect and tunnel regressions must also pass.

No live password, reviewer account or review video is created by these tests.
Before enabling, validate the deployment with default-off code, then obtain the
separate approval for credential creation, feature activation and reviewer sharing.
Verify actual HTTPS sign-in, consent, reconnect, expiry/revocation and two-account
isolation on staging. Record the real functioning review account with secrets
redacted and submit only after reviewer details and video are complete.
