"""Real local OAuth + official MCP HTTP protocol; no provider or network calls."""
import base64
import hashlib
import json
import re
import time
from urllib.parse import parse_qs, urlsplit

import pytest
from flask import session
from starlette.testclient import TestClient

from dredge.auth import User, _users, studio_return_path
from dredge.chatgpt_transport import create_chatgpt_app
from dredge.server import create_app

ISSUER = 'https://dredge.example.test'
CALLBACK = 'https://chatgpt.com/connector/oauth/test-callback'
VERIFIER = 'a' * 64
CHALLENGE = base64.urlsafe_b64encode(hashlib.sha256(VERIFIER.encode()).digest()).rstrip(b'=').decode()


@pytest.fixture
def app(tmp_path, monkeypatch):
    for key, value in {
        'DREDGE_CHATGPT_ENABLED':'true', 'CHATGPT_ISSUER':ISSUER,
        'CHATGPT_RESOURCE':ISSUER+'/chatgpt/mcp','CHATGPT_CLIENT_ID':'test-client',
        'CHATGPT_REDIRECT_URI':CALLBACK,'SECRET_KEY':'fictional-test-secret-key-with-32-bytes',
        'STUDIO_DB_PATH':str(tmp_path/'studio.sqlite3'),
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv('STRIPE_RESTRICTED_KEY', raising=False)
    app = create_app()
    app.config['TESTING'] = True
    yield app
    _users.clear()


def oauth_token(app, owner):
    """Simulate an already signed-in fictional account; exercise real consent/grant."""
    _users[owner] = User(owner, 'Fictional test user', 'fictional@example.test', 'github')
    browser = app.test_client()
    with browser.session_transaction(base_url=ISSUER) as sess:
        sess['_user_id'] = owner
        sess['_fresh'] = True
    consent = browser.get('/chatgpt/oauth/authorize', base_url=ISSUER, query_string={
        'response_type':'code','client_id':'test-client','redirect_uri':CALLBACK,
        'scope':'studio:read','state':'test-state','resource':ISSUER+'/chatgpt/mcp',
        'code_challenge':CHALLENGE,'code_challenge_method':'S256',
    })
    assert consent.status_code == 200, consent.text
    fields = dict(re.findall(r'name="([^"]+)"\s+value="([^"]+)"', consent.text))
    authorized = browser.post('/chatgpt/oauth/authorize', base_url=ISSUER,
        data={'transaction':fields['transaction'],'csrf_token':fields['csrf_token'],'decision':'allow'})
    assert authorized.status_code == 302, authorized.text
    code = parse_qs(urlsplit(authorized.location).query)['code'][0]
    token = browser.post('/chatgpt/oauth/token', base_url=ISSUER, data={
        'grant_type':'authorization_code','client_id':'test-client','code':code,
        'redirect_uri':CALLBACK,'resource':ISSUER+'/chatgpt/mcp','code_verifier':VERIFIER,
    })
    assert token.status_code == 200, token.text
    return token.json['access_token']


def rpc(client, token, method, params=None, **headers):
    hdr = {'Accept':'application/json, text/event-stream','MCP-Protocol-Version':'2025-11-25'}
    if token:
        hdr['Authorization'] = 'Bearer '+token
    hdr.update(headers)
    return client.post('/chatgpt/mcp', json={'jsonrpc':'2.0','id':1,'method':method,'params':params or {}}, headers=hdr)


def test_discovery_auth_gate_initialize_and_two_tools(app):
    token = oauth_token(app, 'github:owner')
    with TestClient(create_chatgpt_app(app), base_url=ISSUER) as client:
        metadata = client.get('/.well-known/oauth-protected-resource/chatgpt/mcp').json()
        assert metadata['resource'] == ISSUER+'/chatgpt/mcp'
        assert metadata['authorization_servers'] == [ISSUER]
        assert metadata['scopes_supported'] == ['studio:read']
        auth = client.get('/.well-known/oauth-authorization-server').json()
        assert auth['code_challenge_methods_supported'] == ['S256']
        unauthenticated = rpc(client, None, 'tools/list')
        assert unauthenticated.status_code == 401
        assert 'resource_metadata=' in unauthenticated.headers['WWW-Authenticate']
        assert rpc(client, 'invalid', 'tools/list').status_code == 401
        init = rpc(client, token, 'initialize', {'protocolVersion':'2025-11-25','capabilities':{},'clientInfo':{'name':'test','version':'1'}})
        assert init.status_code == 200, init.text
        assert init.json()['result']['serverInfo']['name'] == 'DREDGE Studio'
        tools = rpc(client, token, 'tools/list').json()['result']['tools']
        assert {t['name'] for t in tools} == {'get_dredge_access','get_studio_activity_report'}
        for tool in tools:
            assert tool['annotations']['readOnlyHint'] is True
            assert tool['inputSchema']['additionalProperties'] is False
            assert tool['securitySchemes'] == [{'type':'oauth2','scopes':['studio:read']}]
        for name in ('get_dredge_access','get_studio_activity_report'):
            result = rpc(client, token, 'tools/call', {'name':name,'arguments':{}})
            assert result.status_code == 200, result.text
            assert result.json()['result']['isError'] is False
            assert result.json()['result']['structuredContent']
        assert not client.cookies.get('session')
        for path in ('/api/studio/runs', '/api/billing/checkout'):
            assert client.post(path, json={}, headers={'Authorization':'Bearer '+token}).status_code == 401


def test_real_tokens_enforce_owner_and_scope_without_role_escalation(app):
    own_token = oauth_token(app, 'github:owner')
    other_token = oauth_token(app, 'google:other')
    app.config['STUDIO_ROLES'] = {'github:owner':'admin'}
    with app.extensions['studio_store'].connect() as db:
        for owner, ident, tokens in [('github:owner','one',11),('google:other','two',93)]:
            db.execute('INSERT INTO runs(id,owner,status,payload,result,created,started,ended) VALUES(?,?,?,?,?,?,?,?)',
                (ident,owner,'completed',json.dumps({'query':'private','pipeline_type':'standard'}),
                 json.dumps({'provider_calls':1,'token_usage':tokens}),time.time(),1,2))
    with TestClient(create_chatgpt_app(app), base_url=ISSUER) as client:
        for token, expected in ((own_token,11),(other_token,93)):
            result = rpc(client, token, 'tools/call', {'name':'get_studio_activity_report'}).json()['result']
            assert result['structuredContent']['token_usage'] == expected
            assert result['structuredContent']['proposed_runs'] == 1
            assert result['structuredContent']['cost_usd'] is None
            assert 'private' not in json.dumps(result)
        from concurrent.futures import ThreadPoolExecutor
        def concurrent_report(pair):
            token, expected = pair
            result = rpc(client, token, 'tools/call', {'name':'get_studio_activity_report'}).json()['result']
            return result['structuredContent']['token_usage'] == expected
        with ThreadPoolExecutor(max_workers=4) as pool:
            assert all(pool.map(concurrent_report, [(own_token,11),(other_token,93)] * 8))
        for params in ({'name':'get_studio_activity_report','arguments':{'owner':'google:other'}}, {'name':'execute_run'}):
            assert rpc(client,own_token,'tools/call',params).json()['result']['isError'] is True
        # Browser session cookies alone cannot authorize the MCP resource.
        client.cookies.set('session','untrusted-cookie')
        assert rpc(client,None,'tools/list').status_code == 401
        # Protocol host/origin protections are independent of a valid bearer token.
        assert rpc(client,own_token,'tools/list', Origin='https://evil.example').status_code == 403
        assert rpc(client,own_token,'tools/list', Host='evil.example').status_code == 421


def test_revocation_takes_effect_on_next_mcp_request(app):
    token = oauth_token(app, 'github:owner')
    with TestClient(create_chatgpt_app(app), base_url=ISSUER) as client:
        assert rpc(client,token,'tools/list').status_code == 200
        revoked = client.post('/chatgpt/oauth/revoke',data={'token':token,'client_id':'test-client'})
        assert revoked.status_code == 200, revoked.text
        assert rpc(client,token,'tools/list').status_code == 401


def test_login_return_is_local_validated_and_one_use(app):
    valid = '/chatgpt/oauth/authorize?transaction=' + 'a'*43
    with app.test_request_context():
        session['chatgpt_return_to'] = valid
        assert studio_return_path() == valid
        assert studio_return_path() == '/advanced'
        for bad in ('https://evil.example', '//evil.example', valid+'&redirect_uri=https://evil.example', valid+'a'):
            session['chatgpt_return_to'] = bad
            assert studio_return_path() == '/advanced'
        app.config['DREDGE_CHATGPT_ENABLED'] = False
        session['chatgpt_return_to'] = valid
        assert studio_return_path() == '/advanced'


@pytest.mark.parametrize('key,value', [
    ('CHATGPT_ISSUER','http://dredge.example.test'),
    ('CHATGPT_ISSUER','https://dredge.example.test:443'),
    ('CHATGPT_ISSUER','https://dredge.example.test:'),
    ('CHATGPT_ISSUER','https://exämple.test'),
    ('CHATGPT_ISSUER','HTTPS://dredge.example.test'),
    ('CHATGPT_ISSUER','https://127.1'),
    ('CHATGPT_ISSUER','https://0x7f000001'),
    ('CHATGPT_ISSUER','https://127.0.0.01'),
    ('CHATGPT_RESOURCE','https://evil.example/mcp'),
    ('CHATGPT_REDIRECT_URI','https://chatgpt.com.evil.example/connector/oauth/test'),
    ('CHATGPT_REDIRECT_URI',CALLBACK+'?next=evil'),
    ('SECRET_KEY','short'),('DREDGE_CHATGPT_ENABLED','maybe'),
])
def test_bad_configuration_fails_closed(app, monkeypatch, key, value):
    monkeypatch.setenv(key,value)
    if key == 'CHATGPT_ISSUER':
        monkeypatch.setenv('CHATGPT_RESOURCE', value + '/chatgpt/mcp')
    with pytest.raises((RuntimeError,ValueError)):
        create_app()


def test_mcp_malformed_requests_fail_without_writes_or_token_leaks(app):
    token = oauth_token(app, 'github:owner')
    with app.extensions['studio_store'].connect() as db:
        before = '\n'.join(db.iterdump())
    with TestClient(create_chatgpt_app(app), base_url=ISSUER) as client:
        headers = {'Accept':'application/json, text/event-stream','Authorization':'Bearer '+token,
                   'Content-Type':'application/json'}
        for payload in ('{broken', '[]', 'null', '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_dredge_access","arguments":[]}}'):
            response = client.post('/chatgpt/mcp',content=payload,headers=headers)
            if response.status_code == 200:
                assert response.json()['error']['code'] == -32602
            else:
                assert response.status_code in (400, 422), response.text
            assert token not in response.text
        for payload in (b'\xff', b'{"jsonrpc":"2.0","id":"\\ud800","method":"ping"}',
                        b'{"jsonrpc":"2.0","id":1,"method":"ping","params":{"x":NaN}}'):
            response = client.post('/chatgpt/mcp',content=payload,headers=headers)
            assert response.status_code == 400, response.text
            assert token not in response.text
        oversized = client.post('/chatgpt/mcp',content=' '*17000,headers=headers)
        assert oversized.status_code == 413
        invalid_protocol = rpc(client,token,'tools/list',**{'MCP-Protocol-Version':'made-up'})
        assert invalid_protocol.status_code == 400
    with app.extensions['studio_store'].connect() as db:
        assert '\n'.join(db.iterdump()) == before


@pytest.mark.parametrize('field,value', [('issuer','https://evil.example'),
    ('resource','https://evil.example/mcp'),('expires_at',0),('revoked_at',1),
    ('owner','unbound-user'),('scope','studio:write')])
def test_invalid_token_records_are_rejected_at_http_gate(app, field, value):
    token = oauth_token(app, 'github:owner')
    # Only fixed test field names are interpolated; this is fixture corruption,
    # never an API accepted from the MCP client.
    with app.extensions['studio_store'].connect() as db:
        db.execute(f'UPDATE chatgpt_oauth_tokens SET {field}=?', (value,))
    with TestClient(create_chatgpt_app(app), base_url=ISSUER) as client:
        response = rpc(client,token,'tools/list')
        assert response.status_code == 401
        assert token not in response.text
