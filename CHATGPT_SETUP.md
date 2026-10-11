# DREDGE ChatGPT integration: local candidate setup

**Default off. Not deployed, connected, submitted, or approved.** This candidate
adds an OAuth authorization server using Authlib 1.8.0 and a Streamable HTTP
resource server using the official MCP Python SDK 1.29.0 (supported v1 maintenance
line). It does not use custom JWT decoding, pass Google/GitHub tokens through, or
reinterpret browser cookies as MCP credentials.

## What the candidate serves when explicitly configured

- `/.well-known/oauth-authorization-server`: authorization-server discovery.
- `/chatgpt/oauth/authorize`: existing DREDGE Google/GitHub login, then explicit
  consent for `studio:read`. The selected existing account owns the grant.
- `/chatgpt/oauth/token`: public-client authorization-code grant, enforced S256
  PKCE, exact registered redirect, client and resource binding. No refresh tokens.
- `/chatgpt/oauth/revoke`: token revocation.
- `/.well-known/oauth-protected-resource/chatgpt/mcp`: SDK-generated resource metadata.
- `/chatgpt/mcp`: official SDK Streamable HTTP, protected on every request, with exactly
  `get_dredge_access` and `get_studio_activity_report`.

Access tokens are opaque random credentials. Durable token hashes are looked up
locally with issuer/resource, expiry, revocation and scope checks. This is a
reference-token design; there is no JWT signature to validate. The identity is the
provider-prefixed owner saved during consent, never an account ID from a tool.
Token checks run at both the HTTP gate and tool execution. Roles and entitlement
remain server-derived. Separate request contexts prevent identities crossing users
or overriding browser sessions.

The existing Studio report is not newly paywalled. It remains owner-only even for
administrators. Missing usage/cost stays unknown. Tool results omit prompts,
evidence, run output, payment IDs, prices and checkout links. Tools cannot execute,
buy, change a subscription, or create an account.

## Explicit configuration needed for an eventual private connection

These are deployment inputs, **not permission to set live credentials or grants**.

1. Existing DREDGE persistent storage and stable session signing key. Keep
   `STUDIO_DB_PATH` an absolute durable-volume path shared by all application
   workers, not ephemeral container storage. Existing `SECRET_KEY` must contain at
   least 32 bytes and remain stable across workers/restarts. Enter/manage secrets
   through an approved secure handoff; do not paste them into chat or source.
2. Existing Google/GitHub sign-in configured for the chosen HTTPS origin. The
   existing provider redirect base (`OAUTH_REDIRECT_BASE`) and provider callback
   allowlists must match the deployed DREDGE origin. No provider app was created
   or reconfigured here.
3. A canonical ASCII/punycode HTTPS origin (no explicit port) for `CHATGPT_ISSUER`, e.g. `https://dredge.example.com`.
   Set `CHATGPT_RESOURCE` to that exact value plus `/chatgpt/mcp`. The pre-existing Gateway `/mcp` route is unchanged.
4. `CHATGPT_CLIENT_ID`: one predefined public OAuth client identifier configured
   consistently in ChatGPT management and this server. There is no client secret.
   `CHATGPT_REDIRECT_URI`: copy the exact callback shown in ChatGPT's management
   page. Do not infer a callback ID or accept a wildcard. This slice supports one
   explicitly registered client; it does not advertise dynamic registration or
   Client ID Metadata Document fetching.
5. Install the repository package using its normal workflow: `pip install -e .`.
   Base requirements now include MCP 1.29.0 and Authlib 1.8.0, with compatible Flask
   and Pydantic floors, so normal CI and Docker builds exercise the integration.
   Installation does not enable the feature. Review security updates before release.
6. After separate deployment/access authorization, set
   `DREDGE_CHATGPT_ENABLED=true`. An absent/false flag registers no new OAuth or
   ChatGPT MCP endpoints; incomplete/invalid configuration fails startup. The
   existing Railway Docker build installs the updated base requirements; the existing start
   command and Orion Gateway routes stay unchanged. The opt-in mounted app's
   lifecycle is explicitly started/stopped by the Gateway. No Docker build or
   Railway deployment has been executed in this workspace.

   For an isolated local ASGI test without the Gateway, use:

       uvicorn dredge.chatgpt_transport:create_chatgpt_app --factory

   Both paths use `/chatgpt/mcp` to avoid colliding with the pre-existing Gateway
   `/mcp` proxy. Ordinary Flask WSGI startup does not serve the ChatGPT transport.
7. Terminate HTTPS at a trusted reverse proxy. Preserve the verified external
   scheme/host to the ASGI/WSGI application; never trust forwarded headers from
   arbitrary internet clients. Configure Uvicorn proxy trust only for the actual
   proxy address. OAuth rejects requests whose scheme/host do not match the issuer;
   MCP rejects unexpected hosts/origins. Keep secure, HTTP-only, same-site cookies.
   Enforce ingress rate limits and request limits. Do not log Authorization headers,
   token request bodies, authorization codes or consent parameters at the proxy.
8. Use a dedicated fictional-data reviewer account for private tests. Confirm the
   actual ChatGPT authorize/consent/cancel flow, both tools, disconnect/reconnect,
   cross-account isolation, expiry, rate limiting, and recovery through the deployed
   HTTPS route. Unit/ASGI tests do not replace this final integration test.

## Operational bounds

Access tokens expire after one hour; the user reconnects when needed. Refresh-token
rotation/offline access is deliberately not implemented. Codes expire after five
minutes and consent transactions after ten. Expired records should be cleaned up
by a reviewed retention job before sustained production use; no background job or
new external monitoring was installed. Back up the database and restrict filesystem
access. Losing it invalidates grants; restoring old backups could restore revoked
state and requires a grant invalidation plan before production recovery.

No live end-to-end ChatGPT test has occurred. No publisher verification, privacy/
support registration, category approval, payment-provider approval, publication or
deployment is implied by this implementation.

## Commerce preparation

See `CHATGPT_COMMERCE_GATE.md`. Sales are not registered as tools and cannot be
activated by the read-only integration flag.

## Official sources

Checked October 11, 2026:
- https://developers.openai.com/plugins/build/auth
- https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization
- https://github.com/modelcontextprotocol/python-sdk/tree/v1.29.0
- https://docs.authlib.org/en/stable/flask/2/authorization-server.html
- https://docs.authlib.org/en/stable/specs/rfc7636.html
