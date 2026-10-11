"""Opt-in DREDGE authorization server for one registered public ChatGPT client.

Authlib owns OAuth grant, PKCE and revocation protocol handling. These adapters
supply exact client registration, account-bound consent and durable SQLite state.
Bearer credentials and authorization codes are stored only as SHA-256 digests.
No refresh tokens, dynamic registration, outbound requests or deployment steps.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
import secrets
import time
from contextlib import contextmanager
from dataclasses import dataclass
from urllib.parse import urlsplit

from authlib.integrations.flask_oauth2 import AuthorizationServer
from authlib.integrations.flask_oauth2.requests import FlaskOAuth2Request
from authlib.oauth2 import OAuth2Error
from authlib.oauth2.rfc6749 import AuthorizationCodeMixin, ClientMixin, OAuth2Request
from authlib.oauth2.rfc6749.errors import InvalidGrantError, InvalidRequestError
from authlib.oauth2.rfc6749.grants import AuthorizationCodeGrant
from authlib.oauth2.rfc6750 import BearerTokenGenerator
from authlib.oauth2.rfc7009 import RevocationEndpoint
from authlib.oauth2.rfc7636 import CodeChallenge
from authlib.oauth2.rfc9207 import IssuerParameter
from flask import Blueprint, current_app, g, jsonify, redirect, render_template_string, request, session
from flask_login import current_user
from werkzeug.wrappers import Request

SCOPE = 'studio:read'
CODE_TTL = 300
TOKEN_TTL = 3600
CONSENT_TTL = 600
AUTHORIZE_PATH = '/chatgpt/oauth/authorize'
OWNER_PATTERN = re.compile(r'(?:github|google):[A-Za-z0-9_-]{1,255}\Z')
OPAQUE_PATTERN = re.compile(r'[A-Za-z0-9_-]{43}\Z')
VERIFIER_PATTERN = re.compile(r'[A-Za-z0-9._~-]{43,128}\Z')
AUTHORIZATION_FIELDS = frozenset({
    'response_type', 'client_id', 'redirect_uri', 'scope', 'state', 'resource',
    'code_challenge', 'code_challenge_method',
})
TOKEN_FIELDS = frozenset({
    'grant_type', 'client_id', 'code', 'redirect_uri', 'resource', 'code_verifier',
})

CONSENT_TEMPLATE = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Connect ChatGPT to DREDGE</title></head><body><main>
<h1>Connect ChatGPT to DREDGE?</h1>
<p>Signed in as {{ owner }}.</p>
<p>Allow the registered client <strong>{{ client_id }}</strong> to read your DREDGE
access configuration and aggregate activity for your own Studio runs.</p>
<p>Scope: {{ scope }}. Resource: {{ resource }}.</p>
<p>This access cannot execute runs, change your workspace, purchase anything, or read
case records. Access expires after one hour. Your sign-in alone does not grant access.</p>
<form action="/chatgpt/oauth/authorize" method="post">
<input type="hidden" name="transaction" value="{{ transaction }}">
<input type="hidden" name="csrf_token" value="{{ csrf_token }}">
<button name="decision" value="allow" type="submit">Allow read-only access</button>
<button name="decision" value="deny" type="submit">Deny</button>
</form></main></body></html>'''


def _digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _valid_owner(value):
    return isinstance(value, str) and OWNER_PATTERN.fullmatch(value) is not None


def _error(error='invalid_request', description='Invalid OAuth request.', status=400):
    return jsonify(error=error, error_description=description), status


def _single_fields(values, allowed):
    """Reject parameter pollution, unbounded inputs and unrecognized auth material."""
    return (set(values) <= allowed and all(
        len(values.getlist(key)) == 1 and len(values[key]) <= 2048 for key in values
    ))


@contextmanager
def _transaction():
    """One connection and write lock span grant validation and token persistence.

    Without this boundary, Authlib's query/save/delete callbacks could allow two
    concurrent exchanges to redeem one code. Failures roll back all token writes.
    """
    with current_app.extensions['studio_store'].connect() as db:
        db.execute('BEGIN IMMEDIATE')
        g.chatgpt_oauth_db = db
        try:
            yield db
        finally:
            g.pop('chatgpt_oauth_db', None)


@dataclass(frozen=True)
class _Owner:
    id: str

    def get_id(self):
        return self.id


class _Client(ClientMixin):
    def __init__(self, app):
        self.client_id = app.config['CHATGPT_CLIENT_ID']
        self.redirect_uri = app.config['CHATGPT_REDIRECT_URI']

    def get_client_id(self):
        return self.client_id

    def get_default_redirect_uri(self):
        return self.redirect_uri

    def get_allowed_scope(self, scope):
        return SCOPE if scope == SCOPE else None

    def check_redirect_uri(self, redirect_uri):
        return redirect_uri == self.redirect_uri

    def check_client_secret(self, client_secret):
        return False

    def check_endpoint_auth_method(self, method, endpoint):
        return method == 'none' and endpoint in {'token', 'revocation'}

    def check_response_type(self, response_type):
        return response_type == 'code'

    def check_grant_type(self, grant_type):
        return grant_type == 'authorization_code'


class _AuthorizationCode(AuthorizationCodeMixin):
    def __init__(self, row):
        self.row = dict(row)
        self.code_challenge = row['code_challenge']
        self.code_challenge_method = row['code_challenge_method']

    def get_redirect_uri(self):
        return self.row['redirect_uri']

    def get_scope(self):
        return self.row['scope']


class _S256Challenge(CodeChallenge):
    SUPPORTED_CODE_CHALLENGE_METHOD = ['S256']

    def validate_code_challenge(self, grant, redirect_uri):
        data = grant.request.payload.data
        if (data.get('code_challenge_method') != 'S256'
                or not OPAQUE_PATTERN.fullmatch(data.get('code_challenge', ''))):
            raise InvalidRequestError('PKCE with an S256 code challenge is required.')
        return super().validate_code_challenge(grant, redirect_uri)

    def validate_code_verifier(self, grant, result):
        # Authlib's dollar-anchored regex accepts a trailing newline; use a full
        # RFC 7636 shape check while retaining Authlib's cryptographic validation.
        if not VERIFIER_PATTERN.fullmatch(grant.request.form.get('code_verifier', '')):
            raise InvalidRequestError('A valid PKCE code verifier is required.')
        return super().validate_code_verifier(grant, result)


class _Grant(AuthorizationCodeGrant):
    TOKEN_ENDPOINT_AUTH_METHODS = ['none']

    def validate_authorization_request(self):
        data = self.request.payload.data
        if data.get('resource') != current_app.config['CHATGPT_RESOURCE']:
            raise InvalidRequestError('The configured resource is required.', error='invalid_target')
        # Avoid reflecting a malicious redirect in Authlib's ASCII-only error text.
        if data.get('redirect_uri') != current_app.config['CHATGPT_REDIRECT_URI']:
            raise InvalidRequestError('The exact registered redirect URI is required.')
        if not data.get('state') or not data.get('redirect_uri'):
            raise InvalidRequestError('State and the exact registered redirect URI are required.')
        return super().validate_authorization_request()

    def save_authorization_code(self, code, oauth_request):
        now = int(time.time())
        data = oauth_request.payload.data
        g.chatgpt_oauth_db.execute('''
            INSERT INTO chatgpt_oauth_codes
            (code_hash,issuer,resource,owner,client_id,scope,redirect_uri,
             code_challenge,code_challenge_method,created_at,expires_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?)''', (
            _digest(code), current_app.config['CHATGPT_ISSUER'], data['resource'],
            oauth_request.user.get_id(), oauth_request.client.get_client_id(),
            oauth_request.scope, oauth_request.payload.redirect_uri,
            data['code_challenge'], data['code_challenge_method'], now, now + CODE_TTL,
        ))

    def query_authorization_code(self, code, client):
        row = g.chatgpt_oauth_db.execute('''
            SELECT * FROM chatgpt_oauth_codes WHERE code_hash=? AND client_id=?
              AND issuer=? AND resource=?''', (
            _digest(code), client.get_client_id(), current_app.config['CHATGPT_ISSUER'],
            current_app.config['CHATGPT_RESOURCE'],
        )).fetchone()
        if row is None:
            return None
        if row['consumed_at'] is not None:
            # RFC 6749 recommends invalidating credentials derived from a replayed code.
            g.chatgpt_oauth_db.execute(
                'UPDATE chatgpt_oauth_tokens SET revoked_at=? WHERE code_hash=? AND revoked_at IS NULL',
                (int(time.time()), row['code_hash']),
            )
            return None
        if row['expires_at'] <= time.time() or row['scope'] != SCOPE:
            return None
        return _AuthorizationCode(row)

    def delete_authorization_code(self, authorization_code):
        # save_token atomically consumes the code with token persistence. Retain the
        # digest as a replay tombstone; never persist the raw credential.
        pass

    def authenticate_user(self, authorization_code):
        owner = authorization_code.row['owner']
        return _Owner(owner) if _valid_owner(owner) else None


class _Issuer(IssuerParameter):
    def get_issuer(self):
        return current_app.config['CHATGPT_ISSUER']


class _Server(AuthorizationServer):
    def create_oauth2_request(self, oauth_request=None):
        if isinstance(oauth_request, OAuth2Request):
            return oauth_request
        return super().create_oauth2_request(oauth_request)


class _Revocation(RevocationEndpoint):
    CLIENT_AUTH_METHODS = ['none']
    SUPPORTED_TOKEN_TYPES = ('access_token',)

    def query_token(self, token_string, token_type_hint):
        row = g.chatgpt_oauth_db.execute('''
            SELECT * FROM chatgpt_oauth_tokens WHERE token_hash=? AND issuer=? AND resource=?''',
            (_digest(token_string), current_app.config['CHATGPT_ISSUER'],
             current_app.config['CHATGPT_RESOURCE']),
        ).fetchone()
        return _Token(row) if row else None

    def revoke_token(self, token, oauth_request):
        g.chatgpt_oauth_db.execute(
            'UPDATE chatgpt_oauth_tokens SET revoked_at=? WHERE token_hash=?',
            (int(time.time()), token.row['token_hash']),
        )


class _Token:
    def __init__(self, row):
        self.row = row

    def check_client(self, client):
        return self.row['client_id'] == client.get_client_id()


class _HideTokenDebug(logging.Filter):
    """Authlib's grant debug message otherwise includes the raw token dictionary."""
    def filter(self, record):
        return not str(record.msg).startswith('Issue token ')


def _saved_request(params):
    return FlaskOAuth2Request(Request.from_values(
        base_url=current_app.config['CHATGPT_ISSUER'], path=AUTHORIZE_PATH,
        method='GET', query_string=params,
    ))


def _validate_configuration(app):
    for key in ('CHATGPT_ISSUER', 'CHATGPT_RESOURCE', 'CHATGPT_REDIRECT_URI'):
        value = app.config.get(key)
        if not isinstance(value, str) or len(value) > 2048:
            raise RuntimeError(f'{key} must be a configured HTTPS URL.')
        try:
            parsed = urlsplit(value)
            valid = (parsed.scheme == 'https' and parsed.hostname and not parsed.username
                     and not parsed.password and not parsed.fragment and not parsed.query
                     and not any(c.isspace() for c in value) and parsed.port in (None, 443))
        except ValueError:
            valid = False
        if not valid:
            raise RuntimeError(f'{key} must be a configured HTTPS URL without credentials or query.')
    issuer = app.config['CHATGPT_ISSUER']
    parsed = urlsplit(issuer)
    if parsed.path or issuer != f'https://{parsed.netloc}':
        raise RuntimeError('CHATGPT_ISSUER must be a canonical HTTPS origin with no trailing slash.')
    resource = urlsplit(app.config['CHATGPT_RESOURCE'])
    if resource.netloc != parsed.netloc or not resource.path:
        raise RuntimeError('CHATGPT_RESOURCE must be an endpoint under CHATGPT_ISSUER.')
    client_id = app.config.get('CHATGPT_CLIENT_ID')
    if not isinstance(client_id, str) or not client_id or len(client_id) > 256:
        raise RuntimeError('CHATGPT_CLIENT_ID must be explicitly configured.')
    if not app.secret_key:
        raise RuntimeError('A secure Flask SECRET_KEY is required for OAuth consent.')


def register_chatgpt_oauth(app):
    """Register only after the caller explicitly enables and configures the boundary."""
    if app.config.get('DREDGE_CHATGPT_ENABLED') is not True:
        return
    _validate_configuration(app)
    app.config.update(SESSION_COOKIE_SECURE=True, SESSION_COOKIE_HTTPONLY=True,
                      SESSION_COOKIE_SAMESITE='Lax')
    with app.extensions['studio_store'].connect() as db:
        db.executescript('''
            CREATE TABLE IF NOT EXISTS chatgpt_oauth_consents (
                transaction_hash TEXT PRIMARY KEY, binding_hash TEXT NOT NULL,
                owner TEXT, parameters TEXT NOT NULL, expires_at INTEGER NOT NULL,
                consumed_at INTEGER);
            CREATE TABLE IF NOT EXISTS chatgpt_oauth_codes (
                code_hash TEXT PRIMARY KEY, issuer TEXT NOT NULL, resource TEXT NOT NULL,
                owner TEXT NOT NULL, client_id TEXT NOT NULL, scope TEXT NOT NULL,
                redirect_uri TEXT NOT NULL, code_challenge TEXT NOT NULL,
                code_challenge_method TEXT NOT NULL, created_at INTEGER NOT NULL,
                expires_at INTEGER NOT NULL, consumed_at INTEGER);
            CREATE TABLE IF NOT EXISTS chatgpt_oauth_tokens (
                token_hash TEXT PRIMARY KEY, code_hash TEXT NOT NULL,
                issuer TEXT NOT NULL, resource TEXT NOT NULL, owner TEXT NOT NULL,
                client_id TEXT NOT NULL, scope TEXT NOT NULL, created_at INTEGER NOT NULL,
                expires_at INTEGER NOT NULL, revoked_at INTEGER);
            CREATE INDEX IF NOT EXISTS chatgpt_oauth_tokens_code
                ON chatgpt_oauth_tokens(code_hash);
        ''')

    def query_client(client_id):
        client = _Client(app)
        return client if client_id == client.client_id else None

    def save_token(token, oauth_request):
        code = oauth_request.authorization_code.row
        now = int(time.time())
        consumed = g.chatgpt_oauth_db.execute('''
            UPDATE chatgpt_oauth_codes SET consumed_at=?
            WHERE code_hash=? AND consumed_at IS NULL AND expires_at>?''',
            (now, code['code_hash'], now),
        )
        if consumed.rowcount != 1:
            raise InvalidGrantError('The authorization code is no longer valid.')
        g.chatgpt_oauth_db.execute('''
            INSERT INTO chatgpt_oauth_tokens
            (token_hash,code_hash,issuer,resource,owner,client_id,scope,created_at,expires_at)
            VALUES(?,?,?,?,?,?,?,?,?)''', (
            _digest(token['access_token']), code['code_hash'], code['issuer'], code['resource'],
            code['owner'], code['client_id'], token['scope'], now, now + token['expires_in'],
        ))

    server = _Server(app, query_client=query_client, save_token=save_token)
    server.scopes_supported = [SCOPE]
    server.register_token_generator('default', BearerTokenGenerator(
        lambda **kwargs: secrets.token_urlsafe(32), expires_generator=TOKEN_TTL,
    ))
    server.register_grant(_Grant, [_S256Challenge(required=True)])
    server.register_endpoint(_Revocation)
    server.register_extension(_Issuer())
    app.extensions['chatgpt_oauth_server'] = server
    log = logging.getLogger('authlib.oauth2.rfc6749.grants.authorization_code')
    if not any(isinstance(f, _HideTokenDebug) for f in log.filters):
        log.addFilter(_HideTokenDebug())
    bp = Blueprint('chatgpt_oauth', __name__)

    @bp.before_request
    def boundary():
        if app.config.get('DREDGE_CHATGPT_ENABLED') is not True:
            return _error('temporarily_unavailable', 'ChatGPT access is disabled.', 503)
        if request.host_url.rstrip('/') != app.config['CHATGPT_ISSUER']:
            return _error('invalid_request', 'Use the configured HTTPS issuer origin.')
        if request.content_length is not None and request.content_length > 16384:
            return _error('invalid_request', 'OAuth request too large.', 413)
        request.max_content_length = 16384

    @bp.after_request
    def security_headers(response):
        response.headers['Cache-Control'] = 'no-store'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['Content-Security-Policy'] = "default-src 'none'; form-action 'self'; frame-ancestors 'none'; base-uri 'none'"
        return response

    @bp.get('/.well-known/oauth-authorization-server')
    def metadata():
        issuer = app.config['CHATGPT_ISSUER']
        return jsonify(
            issuer=issuer, authorization_endpoint=issuer + AUTHORIZE_PATH,
            token_endpoint=issuer + '/chatgpt/oauth/token',
            revocation_endpoint=issuer + '/chatgpt/oauth/revoke',
            response_types_supported=['code'], grant_types_supported=['authorization_code'],
            scopes_supported=[SCOPE], token_endpoint_auth_methods_supported=['none'],
            revocation_endpoint_auth_methods_supported=['none'],
            code_challenge_methods_supported=['S256'],
            authorization_response_iss_parameter_supported=True,
        )

    @bp.route(AUTHORIZE_PATH, methods=['GET', 'POST'])
    def authorize():
        if request.method == 'GET' and 'transaction' not in request.args:
            if not _single_fields(request.args, AUTHORIZATION_FIELDS):
                return _error()
            if request.args.get('response_type') != 'code':
                return _error('unsupported_response_type', 'Only code responses are supported.')
            try:
                server.get_consent_grant()
            except OAuth2Error as error:
                # Invalid initial requests never redirect, even for a supplied URI.
                return _error(error.error, 'The authorization request is not valid.')
            binding = session.setdefault('chatgpt_oauth_binding', secrets.token_urlsafe(32))
            session.setdefault('chatgpt_oauth_csrf', secrets.token_urlsafe(32))
            transaction_id = secrets.token_urlsafe(32)
            params = json.dumps(request.args.to_dict(), sort_keys=True)
            owner = current_user.get_id() if current_user.is_authenticated else None
            if owner is not None and not _valid_owner(owner):
                return _error('access_denied', 'Sign in with a supported DREDGE account.', 403)
            with _transaction() as db:
                db.execute('''INSERT INTO chatgpt_oauth_consents
                    (transaction_hash,binding_hash,owner,parameters,expires_at)
                    VALUES(?,?,?,?,?)''', (_digest(transaction_id), _digest(binding), owner,
                                          params, int(time.time()) + CONSENT_TTL))
        else:
            values = request.args if request.method == 'GET' else request.form
            allowed = {'transaction'} if request.method == 'GET' else {'transaction', 'csrf_token', 'decision'}
            if (not _single_fields(values, allowed)
                    or (request.method == 'POST' and request.args)
                    or not OPAQUE_PATTERN.fullmatch(values.get('transaction', ''))):
                return _error()
            transaction_id = values['transaction']

        binding = session.get('chatgpt_oauth_binding', '')
        if not binding:
            return _error('access_denied', 'Restart the connection in this browser.', 403)
        if request.method == 'POST':
            expected = session.get('chatgpt_oauth_csrf', '')
            supplied = request.form.get('csrf_token', '')
            if (not expected or not OPAQUE_PATTERN.fullmatch(supplied)
                    or not hmac.compare_digest(expected, supplied)):
                return _error('access_denied', 'Refresh the consent page before continuing.', 403)
            if request.form.get('decision') not in {'allow', 'deny'}:
                return _error()

        with _transaction() as db:
            row = db.execute('SELECT * FROM chatgpt_oauth_consents WHERE transaction_hash=?',
                             (_digest(transaction_id),)).fetchone()
            if (not row or row['expires_at'] <= time.time() or row['consumed_at'] is not None
                    or not hmac.compare_digest(row['binding_hash'], _digest(binding))):
                return _error('access_denied', 'This consent request expired or is no longer valid.', 403)
            if not current_user.is_authenticated:
                if request.method == 'POST':
                    return _error('access_denied', 'Sign in again before granting access.', 401)
                session['chatgpt_return_to'] = AUTHORIZE_PATH + '?transaction=' + transaction_id
                return redirect('/auth/login')
            owner = current_user.get_id()
            if not _valid_owner(owner) or (row['owner'] is not None and row['owner'] != owner):
                return _error('access_denied', 'The signed-in account changed. Restart the connection.', 403)
            if row['owner'] is None:
                db.execute('UPDATE chatgpt_oauth_consents SET owner=? WHERE transaction_hash=?',
                           (owner, _digest(transaction_id)))
            oauth_request = _saved_request(json.loads(row['parameters']))
            try:
                grant = server.get_consent_grant(oauth_request, end_user=_Owner(owner))
            except OAuth2Error:
                return _error('access_denied', 'The connection configuration changed. Restart it.', 403)
            if request.method == 'GET':
                return render_template_string(
                    CONSENT_TEMPLATE, owner=owner, client_id=app.config['CHATGPT_CLIENT_ID'],
                    resource=app.config['CHATGPT_RESOURCE'], scope=SCOPE,
                    transaction=transaction_id, csrf_token=session['chatgpt_oauth_csrf'],
                )
            db.execute('UPDATE chatgpt_oauth_consents SET consumed_at=? WHERE transaction_hash=?',
                       (int(time.time()), _digest(transaction_id)))
            return server.create_authorization_response(
                oauth_request, grant_user=_Owner(owner) if request.form['decision'] == 'allow' else None,
                grant=grant,
            )

    @bp.post('/chatgpt/oauth/token')
    def token():
        if (request.args or request.mimetype != 'application/x-www-form-urlencoded'
                or not _single_fields(request.form, TOKEN_FIELDS)
                or request.headers.get('Authorization')):
            return _error()
        if request.form.get('grant_type') != 'authorization_code':
            return _error('unsupported_grant_type', 'Only authorization_code is supported.')
        if request.form.get('resource') != app.config['CHATGPT_RESOURCE']:
            return _error('invalid_target', 'The configured resource is required.')
        with _transaction():
            return server.create_token_response()

    @bp.post('/chatgpt/oauth/revoke')
    def revoke():
        if (request.args or request.mimetype != 'application/x-www-form-urlencoded'
                or not _single_fields(request.form, {'client_id', 'token', 'token_type_hint'})
                or request.headers.get('Authorization')):
            return _error()
        with _transaction():
            return server.create_endpoint_response('revocation')

    app.register_blueprint(bp)


def verify_access_token(app, token):
    """Resolve a token on every call; never treat a browser cookie as MCP auth.

    Issuer, resource, client, scope, expiry, revocation and account identity are
    checked against current server policy, including after a process restart.
    """
    if (app.config.get('DREDGE_CHATGPT_ENABLED') is not True
            or 'chatgpt_oauth_server' not in app.extensions
            or not isinstance(token, str) or not OPAQUE_PATTERN.fullmatch(token)):
        return None
    with app.extensions['studio_store'].connect() as db:
        row = db.execute('''SELECT owner,client_id,scope,expires_at,resource
            FROM chatgpt_oauth_tokens WHERE token_hash=? AND issuer=? AND resource=?
              AND client_id=? AND scope=? AND revoked_at IS NULL AND expires_at>?''', (
            _digest(token), app.config['CHATGPT_ISSUER'], app.config['CHATGPT_RESOURCE'],
            app.config['CHATGPT_CLIENT_ID'], SCOPE, time.time(),
        )).fetchone()
    if row is None or not _valid_owner(row['owner']):
        return None
    return dict(row)
