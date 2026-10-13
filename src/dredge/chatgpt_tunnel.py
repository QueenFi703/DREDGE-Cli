"""Dedicated, socket-only entry point for a separately approved named tunnel.

Never mount this adapter on a TCP listener. The launcher verifies and passes a
permission-restricted Unix socket; Cloudflare's authenticated connector is the
only unprivileged principal outside the app permitted to connect to that socket.
No forwarding header is an identity or account credential.
"""
import os
import socket
from urllib.parse import urlsplit

from flask import render_template_string
from starlette.responses import JSONResponse

SOCKET_PATH = '/run/dredge/gateway.sock'
_READ_PATHS = frozenset({
    '/subscription-info',
    '/health', '/auth/login', '/auth/github', '/auth/github/callback',
    '/auth/google', '/auth/google/callback',
    '/.well-known/oauth-authorization-server',
    '/.well-known/oauth-protected-resource/chatgpt/mcp',
})
_OAUTH_PATHS = {
    '/chatgpt/oauth/authorize': {'GET', 'POST'},
    '/chatgpt/oauth/token': {'POST'},
    '/chatgpt/oauth/revoke': {'POST'},
    '/chatgpt/mcp': {'GET', 'POST', 'DELETE'},
}


class SocketIngress:
    """Interpret only HTTPS provenance over the launcher's restricted IPC path."""
    def __init__(self, app, issuer):
        self.app = app
        self.host = urlsplit(issuer).netloc.encode('ascii')

    async def __call__(self, scope, receive, send):
        if scope['type'] == 'lifespan':
            return await self.app(scope, receive, send)
        if scope['type'] != 'http':
            if scope['type'] == 'websocket':
                return await send({'type':'websocket.close','code':1008})
            return
        # Uvicorn reports None for Unix sockets. A TCP request is never trusted,
        # even from loopback. Launch-time FD verification is equally essential.
        if scope.get('client') is not None:
            return await JSONResponse({'error':'Socket ingress required.'}, status_code=403)(scope, receive, send)
        headers = scope.get('headers', [])
        hosts = [v for k, v in headers if k.lower() == b'host']
        schemes = [v for k, v in headers if k.lower() == b'x-forwarded-proto']
        if hosts != [self.host] or schemes != [b'https']:
            return await JSONResponse({'error':'Invalid HTTPS origin.'}, status_code=400)(scope, receive, send)
        path, method = scope.get('path'), scope.get('method')
        allowed = {'GET', 'HEAD'} if path in _READ_PATHS else _OAUTH_PATHS.get(path, set())
        if method not in allowed:
            return await JSONResponse({'error':'Not found.'}, status_code=404)(scope, receive, send)
        forwarded = dict(scope)
        forwarded['scheme'] = 'https'
        # Keep the real socket/client provenance and every other field unchanged.
        # In particular XFF, Forwarded, and X-Forwarded-Host have no authority.
        return await self.app(forwarded, receive, send)


def _verify_socket():
    try:
        fd = int(os.environ['DREDGE_TUNNEL_SOCKET_FD'])
        with socket.socket(fileno=os.dup(fd)) as inherited:
            if (inherited.family != socket.AF_UNIX
                    or inherited.getsockname() != SOCKET_PATH
                    or not inherited.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN)):
                raise ValueError('Not the configured listening Unix socket.')
    except (KeyError, ValueError, OSError) as error:
        raise RuntimeError('The tunnel launcher must supply its restricted Unix socket.') from error


def create_tunnel_app():
    """Construct only for the dedicated launcher, never as a default deployment."""
    if (os.environ.get('DREDGE_CHATGPT_TUNNEL_ENABLED') != 'true'
            or os.environ.get('DREDGE_CHATGPT_ENABLED') != 'true'):
        raise RuntimeError('The dedicated tunnel and ChatGPT integration must be explicitly enabled.')
    _verify_socket()
    from .server import create_app
    from .chatgpt_transport import create_chatgpt_app
    from .auth import get_oauth
    flask_app = create_app()

    def tunnel_login():
        # Reuse existing provider flows and account binding, with an intentionally
        # narrow login page: no billing, execution UI, or unrelated website links.
        oauth = get_oauth()
        providers = [name for name in ('github', 'google')
                     if oauth and oauth.create_client(name)]
        response = flask_app.make_response(render_template_string('''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Connect DREDGE</title><body><h1>Connect your DREDGE account</h1>
<p>Sign in, then {% if controls_enabled %}review requested Studio permissions{% else %}review the request for read-only Studio access{% endif %}.</p>
{% for provider in providers %}<p><a href="/auth/{{ provider }}">Continue with {{ provider|title }}</a></p>
{% else %}<p>Sign-in is not configured for this staging service.</p>{% endfor %}
<p>Return to ChatGPT to start or cancel the connection.</p></body></html>''', providers=providers, controls_enabled=flask_app.config.get('DREDGE_CHATGPT_CONTROLS_ENABLED') is True))
        response.headers['Cache-Control'] = 'no-store'
        response.headers['Content-Security-Policy'] = "default-src 'none'; frame-ancestors 'none'; base-uri 'none'"
        response.headers['Referrer-Policy'] = 'no-referrer'
        return response

    flask_app.view_functions['auth.login'] = tunnel_login
    return SocketIngress(create_chatgpt_app(flask_app), flask_app.config['CHATGPT_ISSUER'])
