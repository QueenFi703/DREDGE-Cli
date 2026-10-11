"""Content-free Railway readiness listener; never forwards an incoming request.

This is the only TCP listener in the dedicated tunnel image. All probe targets,
paths, and headers are fixed from trusted startup configuration, not HTTP input.
"""
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlsplit

import httpx

SOCKET_PATH = '/run/dredge/gateway.sock'
TUNNEL_READY = 'http://127.0.0.1:2000/ready'


def validate_issuer(issuer):
    value = urlsplit(issuer)
    if (value.scheme != 'https' or not value.hostname or value.netloc != value.hostname
            or value.path or value.query or value.fragment or not issuer.isascii()
            or issuer != 'https://' + value.hostname):
        raise RuntimeError('Health probe requires the canonical HTTPS issuer.')
    return issuer


def ready(issuer):
    try:
        with httpx.Client(transport=httpx.HTTPTransport(uds=SOCKET_PATH),
                          timeout=1.0, trust_env=False, follow_redirects=False) as app:
            # TLS ends at Cloudflare; the restricted local Unix socket uses HTTP.
            response = app.get('http://' + urlsplit(issuer).netloc + '/health',
                               headers={'X-Forwarded-Proto':'https'})
            if response.status_code != 200:
                return False
        with httpx.Client(timeout=1.0, trust_env=False, follow_redirects=False) as tunnel:
            return tunnel.get(TUNNEL_READY).status_code == 200
    except (httpx.HTTPError, OSError):
        return False


def handler_for(issuer):
    validate_issuer(issuer)

    class HealthHandler(BaseHTTPRequestHandler):
        server_version = 'DREDGE-Readiness'
        sys_version = ''

        def setup(self):
            super().setup()
            self.connection.settimeout(2.0)

        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path != '/health':
                status = 404
            else:
                status = 200 if ready(issuer) else 503
            self.send_response(status)
            self.send_header('Content-Length', '0')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()

        def do_HEAD(self):
            self.do_GET()

        def do_POST(self):
            self.send_response(404)
            self.send_header('Content-Length', '0')
            self.end_headers()

    return HealthHandler


def main():
    issuer = validate_issuer(os.environ.get('CHATGPT_ISSUER', ''))
    try:
        port = int(os.environ['PORT'])
        if not 1024 <= port <= 65535 or port == 2000:
            raise ValueError('Invalid readiness port.')
    except (KeyError, ValueError) as error:
        raise RuntimeError('PORT must be a nonprivileged readiness port other than 2000.') from error
    with HTTPServer(('0.0.0.0', port), handler_for(issuer)) as server:
        server.serve_forever()


if __name__ == '__main__':
    main()
