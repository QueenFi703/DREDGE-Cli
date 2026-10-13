"""Opt-in OAuth-protected MCP Streamable HTTP using the official MCP SDK.

ASGI factory: uvicorn dredge.chatgpt_transport:create_chatgpt_app --factory
The existing Flask application is mounted alongside the read-only MCP route.
"""
import os
import json
import re
from contextlib import asynccontextmanager
from urllib.parse import urlsplit
from typing import Annotated



class _JsonEnvelope:
    """Bound JSON bytes before SDK parsing; reject nonportable Unicode safely."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope['method'] != 'POST':
            return await self.app(scope, receive, send)
        from starlette.responses import JSONResponse
        body = bytearray()
        while True:
            message = await receive()
            if message['type'] == 'http.disconnect':
                return
            body.extend(message.get('body', b''))
            if len(body) > 16384:
                return await JSONResponse({'error':'Request too large.'}, status_code=413)(scope, receive, send)
            if not message.get('more_body', False):
                break
        try:
            # Round-trip encoding detects unpaired surrogates, which otherwise
            # raise during SDK response serialization. No protocol fields change.
            data = json.loads(bytes(body).decode('utf-8'))
            json.dumps(data, ensure_ascii=False, allow_nan=False).encode('utf-8')
        except (ValueError, UnicodeError, RecursionError):
            return await JSONResponse({'error':'Invalid UTF-8 JSON request.'}, status_code=400)(scope, receive, send)
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {'type':'http.request', 'body':bytes(body), 'more_body':False}
            return await receive()

        await self.app(scope, replay, send)


def configure_chatgpt(app):
    """Fail closed on partial or ambiguous configuration, with no network access."""
    flag = os.environ.get('DREDGE_CHATGPT_ENABLED', '').strip().lower()
    app.config['DREDGE_CHATGPT_ENABLED'] = False
    app.config['DREDGE_CHATGPT_RUN_TOOLS_ENABLED'] = False
    app.config['DREDGE_CHATGPT_CONTROLS_ENABLED'] = False
    if flag in {'', '0', 'false', 'off', 'no'}:
        return
    if flag != 'true':
        raise RuntimeError('DREDGE_CHATGPT_ENABLED must be true or false.')
    runs_flag = os.environ.get('DREDGE_CHATGPT_RUN_TOOLS_ENABLED', '').strip().lower()
    if runs_flag not in {'', '0', 'false', 'off', 'no', 'true'}:
        raise RuntimeError('DREDGE_CHATGPT_RUN_TOOLS_ENABLED must be true or false.')
    controls_flag = os.environ.get('DREDGE_CHATGPT_CONTROLS_ENABLED', '').strip().lower()
    if controls_flag not in {'', '0', 'false', 'off', 'no', 'true'}:
        raise RuntimeError('DREDGE_CHATGPT_CONTROLS_ENABLED must be true or false.')
    keys = ('CHATGPT_ISSUER', 'CHATGPT_RESOURCE', 'CHATGPT_CLIENT_ID', 'CHATGPT_REDIRECT_URI')
    values = {key: os.environ.get(key, '').strip() for key in keys}
    if not all(values.values()):
        raise RuntimeError('ChatGPT OAuth configuration is incomplete.')
    issuer = urlsplit(values['CHATGPT_ISSUER'])
    if (issuer.scheme != 'https' or not issuer.hostname or issuer.username or issuer.password
            or issuer.path or issuer.query or issuer.fragment or issuer.port is not None
            or values['CHATGPT_ISSUER'] != 'https://' + issuer.hostname
            or len(issuer.hostname) > 253
            or not re.fullmatch(r'[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)*', issuer.hostname)):
        raise RuntimeError('CHATGPT_ISSUER must be a canonical HTTPS origin without a trailing slash.')
    # Compare with the exact parser used by SDK metadata. DNS-looking numeric
    # forms (e.g. 127.1) can otherwise normalize into a different issuer.
    from pydantic import AnyHttpUrl, TypeAdapter, UrlConstraints
    canonical = TypeAdapter(Annotated[AnyHttpUrl, UrlConstraints(preserve_empty_path=True)])
    if str(canonical.validate_python(values['CHATGPT_ISSUER'])) != values['CHATGPT_ISSUER']:
        raise RuntimeError('CHATGPT_ISSUER must match its canonical HTTPS URL exactly.')
    if values['CHATGPT_RESOURCE'] != values['CHATGPT_ISSUER'] + '/chatgpt/mcp':
        raise RuntimeError('CHATGPT_RESOURCE must equal CHATGPT_ISSUER plus /chatgpt/mcp.')
    callback = urlsplit(values['CHATGPT_REDIRECT_URI'])
    if (callback.scheme != 'https' or callback.netloc != 'chatgpt.com'
            or values['CHATGPT_REDIRECT_URI'] != 'https://chatgpt.com' + callback.path or callback.query
            or callback.fragment or not re.fullmatch(
                r'/connector/oauth/[A-Za-z0-9_-]+|/connector_platform_oauth_redirect', callback.path)):
        raise RuntimeError('CHATGPT_REDIRECT_URI must be the exact HTTPS ChatGPT callback from management.')
    if len(values['CHATGPT_CLIENT_ID']) > 256 or any(c.isspace() for c in values['CHATGPT_CLIENT_ID']):
        raise RuntimeError('CHATGPT_CLIENT_ID must be one pre-registered public client identifier.')
    if len(os.environ.get('SECRET_KEY', '').encode()) < 32:
        raise RuntimeError('ChatGPT requires an existing stable SECRET_KEY of at least 32 bytes.')
    if not os.path.isabs(os.environ.get('STUDIO_DB_PATH', '')):
        raise RuntimeError('ChatGPT requires an explicit absolute persistent STUDIO_DB_PATH.')
    app.config.update(values)
    app.config.update(DREDGE_CHATGPT_ENABLED=True,
                      DREDGE_CHATGPT_RUN_TOOLS_ENABLED=(runs_flag == 'true'),
                      DREDGE_CHATGPT_CONTROLS_ENABLED=(controls_flag == 'true'),
                      SESSION_COOKIE_SECURE=True,
                      SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE='Lax')
    from .chatgpt_oauth import register_chatgpt_oauth
    register_chatgpt_oauth(app)


def create_chatgpt_app(flask_app=None):
    """Build the combined local ASGI app; default configuration exposes no MCP."""
    from starlette.applications import Starlette
    from starlette.middleware.wsgi import WSGIMiddleware
    from starlette.routing import Mount, Route

    if flask_app is None:
        from .server import create_app
        flask_app = create_app()
    if not flask_app.config.get('DREDGE_CHATGPT_ENABLED'):
        return Starlette(routes=[Mount('/', app=WSGIMiddleware(flask_app))])

    # MCP machinery is never imported for the disabled ordinary Flask app.
    from anyio import to_thread
    from flask_login import login_user
    from mcp.server.auth.middleware.auth_context import AuthContextMiddleware, get_access_token
    from mcp.server.auth.middleware.bearer_auth import BearerAuthBackend, RequireAuthMiddleware
    from mcp.server.auth.provider import AccessToken
    from mcp.server.auth.routes import build_resource_metadata_url, create_protected_resource_routes
    from mcp.server.fastmcp.server import StreamableHTTPASGIApp
    from mcp.server.lowlevel import Server
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from mcp.server.transport_security import TransportSecuritySettings
    from mcp.types import CallToolResult, TextContent, Tool
    from pydantic import AnyHttpUrl, TypeAdapter, UrlConstraints
    from starlette.middleware import Middleware
    from starlette.middleware.authentication import AuthenticationMiddleware

    from .auth import User
    from .chatgpt_foundation import SCOPE, call_read_tool, supported_scopes, tool_definitions
    from .chatgpt_oauth import verify_access_token
    from .chatgpt_runs import validate_run_arguments
    from .chatgpt_research import validate_research_arguments
    from .chatgpt_controls import CONTROL_NAMES, validate_control_arguments

    issuer = flask_app.config['CHATGPT_ISSUER']
    resource = flask_app.config['CHATGPT_RESOURCE']
    resource_metadata_url = build_resource_metadata_url(AnyHttpUrl(resource))

    class Verifier:
        async def verify_token(self, token):
            record = await to_thread.run_sync(verify_access_token, flask_app, token)
            if not record:
                return None
            return AccessToken(token=token, client_id=record['client_id'],
                               scopes=record['scope'].split(), expires_at=record['expires_at'],
                               resource=record['resource'], subject=record['owner'])

    server = Server('DREDGE Studio', version='0.1.0',
                    instructions='Read existing DREDGE account access and owner-only Studio activity. No sales or execution.')

    @server.list_tools()
    async def list_tools():
        token = get_access_token()
        record = (await to_thread.run_sync(verify_access_token, flask_app, token.token)
                  if token else None)
        if not record:
            raise PermissionError('DREDGE authorization is no longer valid.')
        # Discovery exposes enabled schemas and their required scopes, never data.
        # Existing base grants must be able to discover tools that need consent.
        return [Tool(**definition) for definition in tool_definitions(flask_app)]

    def invoke(name, arguments, token):
        # Recheck revocation/expiry at execution, not only at the HTTP gate.
        record = verify_access_token(flask_app, token)
        if not record:
            raise PermissionError('DREDGE authorization is no longer valid.')
        definition = next((tool for tool in tool_definitions(flask_app)
                           if tool['name'] == name), None)
        if definition is None:
            raise ValueError('Unknown or disabled tool.')
        required_scopes = definition['securitySchemes'][0]['scopes']
        if not frozenset(required_scopes) <= frozenset(record['scope'].split()):
            # Check semantic constraints too (e.g. timestamp ordering). Invalid
            # inputs must not prompt consent, and no run data may be read here.
            if name in CONTROL_NAMES:
                validate_control_arguments(name, arguments)
            elif name == 'find_my_saved_research':
                validate_research_arguments(arguments)
            else:
                validate_run_arguments(name, arguments)
            permission = 'permission' if name in CONTROL_NAMES else 'read-only permission'
            challenge = (f'Bearer resource_metadata="{resource_metadata_url}", '
                         'error="insufficient_scope", '
                         f'error_description="This tool requires additional DREDGE {permission}.", '
                         f'scope="{" ".join(required_scopes)}"')
            return CallToolResult(isError=True, content=[TextContent(
                type='text', text=f'Additional DREDGE {permission} is required for this tool.')],
                **{'_meta': {'mcp/www_authenticate': [challenge]}})
        # An isolated request context receives no browser cookies or request input.
        # login_user here only binds the verified principal for existing helpers;
        # no session is sent to the MCP client or added to the browser session.
        with flask_app.test_request_context(base_url=issuer):
            login_user(User(record['owner'], '', '', 'chatgpt'))
            return call_read_tool(name, arguments, scopes=frozenset(record['scope'].split()))

    @server.call_tool(validate_input=True)
    async def call_tool(name, arguments):
        token = get_access_token()
        if token is None:
            raise PermissionError('DREDGE authorization is required.')
        try:
            return await to_thread.run_sync(invoke, name, arguments, token.token)
        except (PermissionError, ValueError):
            return CallToolResult(isError=True, content=[TextContent(type='text', text='The request is not authorized or is invalid.')])
        except Exception:
            flask_app.logger.warning('DREDGE ChatGPT request failed; no account data or tokens logged.')
            return CallToolResult(isError=True, content=[TextContent(type='text', text='DREDGE data is temporarily unavailable. Try again later.')])

    manager = StreamableHTTPSessionManager(
        app=server, json_response=True, stateless=True, max_request_body_size=16384,
        security_settings=TransportSecuritySettings(enable_dns_rebinding_protection=True,
            allowed_hosts=[urlsplit(issuer).netloc], allowed_origins=[issuer, 'https://chatgpt.com']))
    protected = RequireAuthMiddleware(_JsonEnvelope(StreamableHTTPASGIApp(manager)), [SCOPE],
                                      resource_metadata_url)
    routes = [Route('/chatgpt/mcp', endpoint=protected)]
    routes.extend(create_protected_resource_routes(
        resource_url=AnyHttpUrl(resource), authorization_servers=[TypeAdapter(Annotated[AnyHttpUrl, UrlConstraints(preserve_empty_path=True)]).validate_python(issuer)],
        scopes_supported=sorted(supported_scopes(flask_app)), resource_name='DREDGE Studio'))
    routes.append(Mount('/', app=WSGIMiddleware(flask_app)))

    @asynccontextmanager
    async def lifespan(app):
        async with manager.run():
            yield

    app = Starlette(routes=routes, lifespan=lifespan, middleware=[
        Middleware(AuthenticationMiddleware, backend=BearerAuthBackend(Verifier())),
        Middleware(AuthContextMiddleware),
    ])
    app.state.dredge_flask = flask_app
    return app
