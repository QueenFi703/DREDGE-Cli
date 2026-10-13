# Dedicated ChatGPT tunnel image — local candidate

This is a separate opt-in deployment candidate. It does not alter the ordinary
Railway Dockerfile, start script, production Gateway, or existing OAuth/MCP
implementation. A narrow shared auth logging cleanup removes raw provider exceptions, tracebacks,
and profile debug output. Google sign-in-start failures now return the existing
login error page rather than an uncaught traceback; successful login is unchanged. No tunnel, DNS record,
provider application, or credential was
created by this patch. Do not deploy until the exact hostname, access, cost,
secret-entry and ingress-removal steps have been separately approved.

## Why a separate image

Railway terminates public TLS before the application. Observed proxy IPs are not
a durable origin identity. The temporary staging IP allowlist demonstrated the
problem; it is not this deployment's trust boundary.

A named authenticated Cloudflare Tunnel connects outward from the same container.
Its local origin is `unix:/run/dredge/gateway.sock`, a permission-restricted Unix
socket. The ASGI app has **no TCP listener**. The launcher passes the prebound
socket explicitly, and the app verifies that descriptor before starting. Requests
with TCP client information are rejected, even from loopback. The tunnel entry
validates one exact canonical Host and one literal `X-Forwarded-Proto: https`,
changes only the ASGI scheme, and does not interpret forwarded client identities.
Cloudflare documents that it replaces X-Forwarded-Proto with the actual visitor
protocol. The socket's filesystem boundary, not a header alone, authenticates the
local ingress path.

The dedicated surface includes only health, OAuth discovery/authorize/token/revoke,
existing Google/GitHub login/callback handlers, `/chatgpt/mcp`, and GET/HEAD of
the generic static `/subscription-info` page. The informational page contains no
account state, forms, checkout links or scripts and uses the configured ChatGPT
issuer origin, which may differ from website billing. Its small login
page contains no billing or execution links. Legacy Gateway `/mcp`, website APIs,
billing, dashboards and execution routes are intentionally not served by this
separate integration image. The original production image retains them.

## Runtime layout

- Root-owned immutable source, executable and supervisor files.
- Minimal root bootstrap prepares restricted runtime directories, socket and token
  file, then runs children under separate non-root identities.
- Application and readiness process use the app identity. The readiness process
  receives no secrets, but shares the app UID and therefore is a trusted process,
  not an OS-level isolation boundary from the Studio database.
- Existing database files must already be writable by app UID 1000; bootstrap
  deliberately does not recursively change ownership of existing data.
- `cloudflared` has a distinct identity and can connect to the Unix socket, but
  cannot replace it, read the database, or receive app secrets in its environment.
- Tunnel credentials are provided by an approved secure handoff and held in a
  restrictive token file. They are not command arguments, source, or app variables.
- `/run/dredge/gateway.sock` is the sole application listener.
- `127.0.0.1:2000` is private tunnel readiness/metrics. It is never published.
- `0.0.0.0:$PORT` serves **only** an empty-body `/health` response. It does not proxy
  arbitrary paths, URLs, headers or request bodies. Its fixed probes check the app
  through the Unix socket and the tunnel's `/ready` endpoint without redirects.
- The supervisor terminates the remaining processes if a child exits, handles
  shutdown signals, and waits for child cleanup. Transient tunnel reconnection is
  reflected in readiness; it does not create a second ingress route.

## Approved deployment inputs required later

1. A verified Cloudflare account and owned zone, plus one specifically approved
   hostname such as `chatgpt-staging.<owned-domain>`. Do not assume the user owns an
   example or create a DNS record without approval. Dashboard public-hostname route
   creation can itself create a DNS CNAME.
2. A **named** tunnel routed to the exact Unix socket above. No quick-tunnel or
   arbitrary-URL fallback. No Cloudflare Access login, interactive challenge, or
   cached OAuth response in front of the MCP/OAuth endpoints. OAuth provides the
   application account authorization. Review edge rules for MCP compatibility.
3. `CHATGPT_ISSUER=https://<approved-hostname>` and `CHATGPT_RESOURCE` equal to that
   issuer plus `/chatgpt/mcp`; exact predefined `CHATGPT_CLIENT_ID` and the callback
   displayed by ChatGPT management. The stable callback was confirmed in the
   staging form: `https://chatgpt.com/connector_platform_oauth_redirect`.
4. Stable, fresh staging `SECRET_KEY`; `STUDIO_DB_PATH=/data/studio/studio.sqlite3`
   on the isolated existing volume; both `DREDGE_CHATGPT_ENABLED=true` and
   `DREDGE_CHATGPT_TUNNEL_ENABLED=true` for the dedicated image. Neither flag creates
   a provider credential or an account grant. Leave payment/AI execution secrets
   absent and subscription commerce unavailable.
5. Secure user entry of the tunnel token. Whoever possesses that token can run a
   connector for the tunnel, so treat it as a highly sensitive credential. Do not
   paste it into chat or use shell-history-bearing command arguments.
6. Remove the Railway public application domain and verify no TCP proxy after
   explicit approval. The app is socket-only regardless, but do not leave obsolete
   ingress routes around. Keep Railway's internal healthcheck on the separate
   readiness listener. Explicitly unset the temporary proxy-IP setting.
7. Reconfigure `OAUTH_REDIRECT_BASE` and a separately approved staging-specific
   Google/GitHub provider callback for the new hostname before actual sign-in.
   No provider credential is required merely to inspect OAuth discovery. Do not
   copy production provider secrets into staging.
8. Reconfirm cost and lifetime: persistent tunnel connections may prevent Railway
   serverless sleeping. The existing 24-hour/$5 test budget is not permission for
   an indefinite always-on service or new paid products.

## Verification and release gates

Local ASGI tests cover duplicate/malformed protocol headers, exact host checks,
TCP-client rejection, minimal route surface, two authenticated owner-bound tools,
no forwarded-identity rewriting, and the fixed readiness target contract.

The current executor denies AF_UNIX socket creation and has no Docker binary.
Real Unix-socket/Uvicorn tests remain in the suite and explicitly skip here. A
successful pure-ASGI test does **not** prove Unix permissions, separate-UID process
isolation, the image build, or the actual Cloudflare connection. These require an
approved CI/container run before deployment, followed by deployed tests:

- Exact socket owner/group/mode before exposure; no alternate app TCP ports.
- Tunnel UID cannot read DB/app secrets or replace the socket; app UID cannot read
  tunnel token or modify privileged bootstrap files.
- Full named-tunnel route, public TLS/Host/protocol behavior and safe rejection of
  forged/duplicate headers and private TCP requests.
- Readiness transitions, child exit/SIGTERM cleanup and no orphaned processes.
- Restart preserves intended database state and requires valid credentials.
- Actual ChatGPT discovery, user sign-in, consent/cancel, both tools, account
  isolation, revocation and reconnect. No real grant has been exercised here.

Provider registration/callback errors now use fixed messages instead of raw
exceptions, tracebacks, provider-object representations or profile-name prints.
Sentinel tests demonstrate these application logs omit fictional sensitive values.
Keep dependency debug logging disabled and verify real proxy/provider logs during
the approved integration test; this is not a blanket guarantee about every
external service or dependency log.

## Official references

- https://developers.cloudflare.com/tunnel/concepts/routing/
- https://developers.cloudflare.com/tunnel/reference/tunnel-tokens/
- https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/configure-tunnels/run-parameters/
- https://developers.cloudflare.com/fundamentals/reference/http-headers/#x-forwarded-proto
- https://developers.cloudflare.com/tunnel/guides/kubernetes/
- https://developers.openai.com/plugins/build/auth
- https://docs.railway.com/networking/private-networking/how-it-works
