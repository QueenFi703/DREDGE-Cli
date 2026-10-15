"""Offline tests only: no real credentials, provider calls or deployment."""
import base64
import hashlib
import json
import re
import time
from urllib.parse import parse_qs, urlsplit

import pytest
from flask_login import login_user
from werkzeug.security import generate_password_hash

from dredge.server import create_app
from dredge.auth import User
from dredge.reviewer_login import STAGING_ORIGIN, active_review_owner, revoke_review_access
from dredge.chatgpt_oauth import verify_access_token
from dredge.chatgpt_foundation import call_read_tool, supported_scopes

PASSWORD = 'fictional-offline-test-password-only'
VERIFIER = 'f' * 64
REDIRECT = 'https://chatgpt.com/connector_platform_oauth_redirect'


@pytest.fixture
def settings(tmp_path, monkeypatch):
    env = dict(SECRET_KEY='offline-session-signing-key-not-for-use-' * 2,
               STUDIO_DB_PATH=str(tmp_path / 'ordinary.sqlite3'),
               DREDGE_CHATGPT_ENABLED='true', DREDGE_CHATGPT_TUNNEL_ENABLED='true',
               DREDGE_CHATGPT_RUN_TOOLS_ENABLED='true', DREDGE_CHATGPT_CONTROLS_ENABLED='true',
               CHATGPT_ISSUER=STAGING_ORIGIN, CHATGPT_RESOURCE=STAGING_ORIGIN + '/chatgpt/mcp',
               CHATGPT_CLIENT_ID='fictional-review-client', CHATGPT_REDIRECT_URI=REDIRECT,
               DREDGE_REVIEWER_LOGIN_ENABLED='true', DREDGE_REVIEWER_GENERATION='1', DREDGE_REVIEWER_USERNAME='offline-reviewer',
               DREDGE_REVIEWER_PASSWORD_HASH=generate_password_hash(PASSWORD, method='scrypt:32768:8:1'),
               DREDGE_REVIEWER_DB_PATH=str(tmp_path / 'review.sqlite3'),
               DREDGE_REVIEWER_EXPIRES_AT=str(int(time.time()) + 3600))
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    # Explicitly avoid ambient provider credentials in tests.
    for key in ('OPENAI_API_KEY', 'GITHUB_CLIENT_SECRET', 'GOOGLE_CLIENT_SECRET', 'STRIPE_RESTRICTED_KEY'):
        monkeypatch.delenv(key, raising=False)
    return env


@pytest.fixture
def app(settings):
    return create_app()


def form(client):
    r = client.get('/auth/reviewer', base_url=STAGING_ORIGIN)
    assert r.status_code == 200
    return re.search(r'name="csrf_token" value="([^"]+)"', r.text)[1]


def sign_in(client, password=PASSWORD, username='offline-reviewer'):
    csrf = form(client)
    return client.post('/auth/reviewer', data=dict(csrf_token=csrf, username=username, password=password), base_url=STAGING_ORIGIN)


def grant(client, app):
    query = dict(response_type='code', client_id='fictional-review-client', redirect_uri=REDIRECT,
                 resource=STAGING_ORIGIN + '/chatgpt/mcp', state='fictional-state',
                 scope=' '.join(sorted(supported_scopes(app))), code_challenge_method='S256',
                 code_challenge=base64.urlsafe_b64encode(hashlib.sha256(VERIFIER.encode()).digest()).rstrip(b'=').decode())
    r = client.get('/chatgpt/oauth/authorize', query_string=query, base_url=STAGING_ORIGIN)
    assert r.status_code == 200, r.text
    fields = dict(re.findall(r'name="([^"]+)" value="([^"]*)"', r.text))
    r = client.post('/chatgpt/oauth/authorize', data={**fields, 'decision': 'allow'}, base_url=STAGING_ORIGIN)
    assert r.status_code == 302, r.text
    return parse_qs(urlsplit(r.location).query)['code'][0]


def exchange(client, code):
    return client.post('/chatgpt/oauth/token', base_url=STAGING_ORIGIN,
                       data=dict(grant_type='authorization_code', client_id='fictional-review-client',
                                 code=code, redirect_uri=REDIRECT, resource=STAGING_ORIGIN + '/chatgpt/mcp', code_verifier=VERIFIER))


def test_password_session_csrf_and_no_credential_disclosure(app):
    c = app.test_client()
    page = c.get('/auth/reviewer', base_url=STAGING_ORIGIN)
    assert 'no-store' in page.headers['Cache-Control']
    assert PASSWORD not in page.text
    r = c.post('/auth/reviewer', data={'username':'offline-reviewer','password':PASSWORD}, base_url=STAGING_ORIGIN)
    assert r.status_code == 403
    with c.session_transaction(base_url=STAGING_ORIGIN) as s:
        s['untrusted_previous_session_value'] = 'discard me'
    assert sign_in(c).status_code == 302
    with c.session_transaction(base_url=STAGING_ORIGIN) as s:
        assert s['_user_id'].startswith('review:')
        assert 'untrusted_previous_session_value' not in s
        assert PASSWORD not in str(dict(s))
    csrf = form(c)
    assert c.post('/auth/reviewer/logout', data={'csrf_token':csrf}, base_url=STAGING_ORIGIN).status_code == 302
    with c.session_transaction(base_url=STAGING_ORIGIN) as s:
        assert '_user_id' not in s


def test_oauth_pending_transaction_survives_login_without_open_redirect(app):
    c = app.test_client()
    query = dict(response_type='code', client_id='fictional-review-client', redirect_uri=REDIRECT,
                 resource=STAGING_ORIGIN + '/chatgpt/mcp', state='fictional-state',scope='studio:read',
                 code_challenge_method='S256',code_challenge=base64.urlsafe_b64encode(hashlib.sha256(VERIFIER.encode()).digest()).rstrip(b'=').decode())
    assert c.get('/chatgpt/oauth/authorize', query_string=query, base_url=STAGING_ORIGIN).status_code == 302
    r = sign_in(c)
    assert r.location.startswith('/chatgpt/oauth/authorize?transaction=')
    consent = c.get(r.location, base_url=STAGING_ORIGIN)
    assert consent.status_code == 200
    assert 'Allow read-only access' in consent.text
    assert c.get('/auth/reviewer?next=https://evil.example', base_url=STAGING_ORIGIN).status_code == 400


def test_all_tools_use_fictional_store_and_no_paid_status(app, monkeypatch):
    owner = app.config['REVIEWER_ACCOUNT']['owner']
    with app.extensions['studio_store'].connect() as db:
        db.execute('INSERT INTO runs(id,owner,status,payload,result,created) VALUES(?,?,?,?,?,?)',
                   ('00000000-0000-0000-0000-000000000999', owner, 'completed',
                    json.dumps({'data_scope':'public_or_fictional'}), json.dumps({'answer':'ORDINARY STORE SENTINEL'}), time.time()))
    with app.test_request_context():
        login_user(User(owner,'review','','review'))
        scopes = supported_scopes(app)
        def call(name, args=None):
            return call_read_tool(name, args or {}, scopes=scopes)
        access = call('get_dredge_access')
        assert access['workspace_role'] == 'operator' and not access['paid_access']
        assert not access['billing_configured'] and not access['in_chat_purchases_enabled']
        runs = call('list_my_studio_runs')['runs']
        assert len(runs) == 2
        for run in runs:
            result = call('get_my_studio_run_result', {'run_id':run['id']})
            assert result['summary'].startswith('FICTIONAL REVIEW FIXTURE')
            assert result['cost_usd'] is None and result['token_usage'] is None
            assert 'ORDINARY STORE SENTINEL' not in str(result)
            call('get_my_studio_execution_graph', {'run_id':run['id']})
            call('get_my_studio_source_summary', {'run_id':run['id']})
        with pytest.raises(ValueError, match='Run not found'):
            call('get_my_studio_run_result', {'run_id':'00000000-0000-0000-0000-000000000999'})
        call('get_studio_activity_report')
        call('list_approved_public_sources')
        budget = call('get_my_studio_budget')
        assert budget['enabled']
        call('update_my_studio_budget', {'currency':'USD','enabled':True,'limit_micro_usd':0})


def test_expiry_revocation_rotation_invalidate_tokens_and_sessions(app, settings, monkeypatch):
    c = app.test_client()
    assert sign_in(c).status_code == 302
    pending = grant(c, app)
    token = exchange(c, grant(c, app)).get_json()['access_token']
    assert verify_access_token(app, token)
    second_app = create_app()
    revoke_review_access(second_app)
    assert verify_access_token(app, token) is None
    assert exchange(app.test_client(), pending).status_code == 400
    assert c.get('/auth/reviewer', base_url=STAGING_ORIGIN).status_code == 401
    # Restart never unrevokes a credential generation.
    assert not active_review_owner(create_app(), app.config['REVIEWER_ACCOUNT']['owner'])
    monkeypatch.setenv('DREDGE_REVIEWER_PASSWORD_HASH', generate_password_hash('different-offline-password',method='scrypt:32768:8:1'))
    monkeypatch.setenv('DREDGE_REVIEWER_GENERATION','2')
    rotated = create_app()
    assert rotated.config['REVIEWER_ACCOUNT']['owner'] != app.config['REVIEWER_ACCOUNT']['owner']
    assert verify_access_token(rotated, token) is None
    rotated.config['REVIEWER_ACCOUNT']['expires_at'] = time.time() - 1
    assert not active_review_owner(rotated, rotated.config['REVIEWER_ACCOUNT']['owner'])


def test_durable_global_throttle(app):
    for _ in range(5):
        assert sign_in(app.test_client(), password='wrong').status_code == 401
    assert sign_in(create_app().test_client()).status_code == 429


@pytest.mark.parametrize('path', ['/api/studio/runs','/api/billing/status','/billing','/advanced',
                                   '/api/architecture/run','/api/casework/cases','/client','/auth/github'])
@pytest.mark.parametrize('method', ['get','post'])
def test_review_principal_cannot_access_other_routes(app, path, method):
    c = app.test_client()
    assert sign_in(c).status_code == 302
    assert getattr(c,method)(path, base_url=STAGING_ORIGIN).status_code == 403


def test_default_off_and_personal_login_unchanged(settings, monkeypatch):
    monkeypatch.setenv('DREDGE_REVIEWER_LOGIN_ENABLED','false')
    app = create_app()
    assert app.test_client().get('/auth/reviewer', base_url=STAGING_ORIGIN).status_code == 404
    with app.test_request_context():
        from dredge.studio import store,role
        login_user(User('github:fictional-personal','Personal','','github'))
        assert store() is app.extensions['studio_store']
        assert role() == 'viewer'


@pytest.mark.parametrize('key,value', [
    ('DREDGE_CHATGPT_TUNNEL_ENABLED','false'), ('CHATGPT_ISSUER','https://production.example'),
    ('DREDGE_REVIEWER_PASSWORD_HASH','plain-text'), ('DREDGE_REVIEWER_USERNAME','admin'),
    ('DREDGE_REVIEWER_EXPIRES_AT','0'), ('DREDGE_REVIEWER_LOGIN_ENABLED','1')])
def test_invalid_configuration_fails_closed(settings, monkeypatch,key,value):
    monkeypatch.setenv(key,value)
    with pytest.raises(RuntimeError): create_app()


def test_database_alias_rejected(settings, monkeypatch):
    monkeypatch.setenv('DREDGE_REVIEWER_DB_PATH',settings['STUDIO_DB_PATH'])
    with pytest.raises(RuntimeError,match='distinct'): create_app()


def test_real_oauth_to_all_eleven_mcp_tools(app, monkeypatch):
    from starlette.testclient import TestClient
    from dredge.chatgpt_transport import create_chatgpt_app
    from dredge.public_source_collector import CATALOG, VERSION
    from test_chatgpt_transport import rpc
    c = app.test_client()
    assert sign_in(c).status_code == 302
    token = exchange(c, grant(c, app)).get_json()['access_token']
    collected = []
    def collect(key):
        collected.append(key)
        return dict(url=CATALOG[key][1], title='Offline mock source', observed_at=time.time(),
                    digest='a'*64, collector_version=VERSION, bytes_observed=10, text='Mock source')
    monkeypatch.setattr('dredge.public_source_revalidation.collect', collect)
    with TestClient(create_chatgpt_app(app), base_url=STAGING_ORIGIN) as client:
        assert rpc(client, None, 'tools/list').status_code == 401
        tools = rpc(client, token, 'tools/list').json()['result']['tools']
        assert len(tools) == 11
        def call(name, args=None):
            r = rpc(client, token, 'tools/call', {'name':name, 'arguments':args or {}})
            assert r.status_code == 200, r.text
            out = r.json()['result']
            assert out['isError'] is False, out
            return out['structuredContent']
        call('get_dredge_access')
        call('get_studio_activity_report')
        runs = call('list_my_studio_runs')['runs']
        found = call('find_my_saved_research', {'query':'fictional library'})
        assert len(found['candidates']) == 1
        source_run = None
        for run in runs:
            result = call('get_my_studio_run_result', {'run_id':run['id']})
            graph = call('get_my_studio_execution_graph', {'run_id':run['id']})
            assert graph['graph_status'] == 'unavailable' and not graph['nodes']
            if 'catalog page' in result['summary']:
                source_run = run['id']
        assert source_run
        call('get_my_studio_budget')
        call('update_my_studio_budget', {'currency':'USD','enabled':True,'limit_micro_usd':0})
        call('list_approved_public_sources')
        call('check_my_studio_public_sources', {'run_id':source_run,'catalog_ids':['mo-dss-child-support']})
        assert collected == ['mo-dss-child-support']
        assert len(call('get_my_studio_source_summary', {'run_id':source_run})['checks']) == 1
        assert rpc(client, token, 'tools/call', {'name':'execute_run','arguments':{}}).json()['result']['isError']
        revoke_review_access(app)
        assert rpc(client, token, 'tools/list').status_code == 401


def test_password_rotation_rejects_old_worker_and_rollback(app, settings, monkeypatch):
    c = app.test_client()
    assert sign_in(c).status_code == 302
    code = grant(c, app)
    token = exchange(c, grant(c, app)).get_json()['access_token']
    monkeypatch.setenv('DREDGE_REVIEWER_PASSWORD_HASH',generate_password_hash('rotated-offline-test-only',method='scrypt:32768:8:1'))
    monkeypatch.setenv('DREDGE_REVIEWER_GENERATION','2')
    create_app()
    assert verify_access_token(app, token) is None
    assert exchange(app.test_client(), code).status_code == 400
    assert c.get('/auth/reviewer', base_url=STAGING_ORIGIN).status_code == 401
    monkeypatch.setenv('DREDGE_REVIEWER_GENERATION','1')
    monkeypatch.setenv('DREDGE_REVIEWER_PASSWORD_HASH',settings['DREDGE_REVIEWER_PASSWORD_HASH'])
    with pytest.raises(RuntimeError, match='rollback'): create_app()


@pytest.mark.parametrize('attack', ['unicode_csrf','unicode_username','duplicate','foreign_origin','oversized','json'])
def test_malformed_login_fails_without_exception(app, attack):
    from werkzeug.datastructures import MultiDict
    c = app.test_client()
    data = dict(csrf_token=form(c), username='offline-reviewer',password=PASSWORD)
    kwargs = {}
    if attack == 'unicode_csrf': data['csrf_token'] = 'é' * 43
    if attack == 'unicode_username': data['username'] = 'é' * 10
    if attack == 'duplicate': data = MultiDict(list(data.items())+[('username','other')])
    if attack == 'foreign_origin': kwargs['headers']={'Origin':'https://foreign.example'}
    if attack == 'oversized': data['password'] = 'x' * 5000
    if attack == 'json': kwargs['json']=data
    else: kwargs['data']=data
    r = c.post('/auth/reviewer',base_url=STAGING_ORIGIN,**kwargs)
    assert 400 <= r.status_code < 500


def test_natural_expiry_restart_preserves_ordinary_service(app, settings, monkeypatch):
    c = app.test_client()
    sign_in(c)
    token = exchange(c, grant(c, app)).get_json()['access_token']
    with app.extensions['reviewer_store'].connect() as db:
        count = db.execute('SELECT count(*) FROM runs').fetchone()[0]
    monkeypatch.setenv('DREDGE_REVIEWER_EXPIRES_AT', str(int(time.time()) - 1))
    expired = create_app()
    assert 'reviewer_store' not in expired.extensions
    assert verify_access_token(expired, token) is None
    assert expired.test_client().get('/auth/login',base_url=STAGING_ORIGIN).status_code == 200
    assert expired.test_client().get('/auth/reviewer',base_url=STAGING_ORIGIN).status_code == 404
    with app.extensions['reviewer_store'].connect() as db:
        assert db.execute('SELECT count(*) FROM runs').fetchone()[0] == count
