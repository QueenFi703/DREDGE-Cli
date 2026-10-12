"""Optional run scope through real OAuth and local MCP; all accounts are fictional."""
import base64
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import parse_qs, urlsplit

import pytest
from starlette.testclient import TestClient

from dredge.auth import User, _users
from dredge.chatgpt_oauth import verify_access_token
from dredge.chatgpt_transport import create_chatgpt_app
from dredge import chatgpt_tunnel_supervisor as supervisor
from dredge.server import create_app

ISSUER = 'https://dredge.example.test'
RESOURCE = ISSUER + '/chatgpt/mcp'
CALLBACK = 'https://chatgpt.com/connector/oauth/fictional-run-scopes'
VERIFIER = 'a' * 64
CHALLENGE = base64.urlsafe_b64encode(hashlib.sha256(VERIFIER.encode()).digest()).rstrip(b'=').decode()
BASE_SCOPE = 'studio:read'
RUN_SCOPE = 'studio:runs:read'
EXPANDED_SCOPE = BASE_SCOPE + ' ' + RUN_SCOPE
OLD_TOOLS = {'get_dredge_access', 'get_studio_activity_report'}
NEW_TOOLS = {'list_my_studio_runs', 'get_my_studio_run_result'}


@pytest.fixture
def make_app(tmp_path, monkeypatch):
    for key, value in {
        'DREDGE_CHATGPT_ENABLED': 'true', 'CHATGPT_ISSUER': ISSUER,
        'CHATGPT_RESOURCE': RESOURCE, 'CHATGPT_CLIENT_ID': 'fictional-client',
        'CHATGPT_REDIRECT_URI': CALLBACK, 'SECRET_KEY': 'fictional-test-secret-key-with-32-bytes',
        'STUDIO_DB_PATH': str(tmp_path / 'studio.sqlite3'),
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv('STRIPE_RESTRICTED_KEY', raising=False)
    monkeypatch.delenv('DREDGE_CHATGPT_RUN_TOOLS_ENABLED', raising=False)

    def create(flag=None):
        if flag is None:
            monkeypatch.delenv('DREDGE_CHATGPT_RUN_TOOLS_ENABLED', raising=False)
        else:
            monkeypatch.setenv('DREDGE_CHATGPT_RUN_TOOLS_ENABLED', flag)
        app = create_app()
        app.config['TESTING'] = True
        return app

    yield create
    _users.clear()


def signed_browser(app, owner='github:fictional-owner'):
    _users[owner] = User(owner, 'Fictional user', 'fictional@example.test', owner.split(':')[0])
    browser = app.test_client()
    with browser.session_transaction(base_url=ISSUER) as session:
        session['_user_id'] = owner
        session['_fresh'] = True
    return browser


def request_parameters(scope=BASE_SCOPE):
    return {
        'response_type': 'code', 'client_id': 'fictional-client', 'redirect_uri': CALLBACK,
        'scope': scope, 'state': 'fictional-state', 'resource': RESOURCE,
        'code_challenge': CHALLENGE, 'code_challenge_method': 'S256', 'ui_locales': 'en-US',
    }


def consent(browser, scope=BASE_SCOPE):
    response = browser.get('/chatgpt/oauth/authorize', base_url=ISSUER,
                           query_string=request_parameters(scope))
    assert response.status_code == 200, response.text
    fields = dict(re.findall(r'name="([^"]+)"\s+value="([^"]+)"', response.text))
    return response, fields


def issue_code(browser, scope=BASE_SCOPE):
    _, fields = consent(browser, scope)
    response = browser.post('/chatgpt/oauth/authorize', base_url=ISSUER,
                            data={**fields, 'decision': 'allow'})
    assert response.status_code == 302, response.text
    query = parse_qs(urlsplit(response.location).query)
    assert query['state'] == ['fictional-state']
    assert query['iss'] == [ISSUER]
    return query['code'][0]


def exchange(browser, code, **changes):
    data = {'grant_type': 'authorization_code', 'client_id': 'fictional-client',
            'code': code, 'redirect_uri': CALLBACK, 'resource': RESOURCE,
            'code_verifier': VERIFIER}
    data.update(changes)
    return browser.post('/chatgpt/oauth/token', base_url=ISSUER, data=data)


def issue_token(app, scope=BASE_SCOPE, owner='github:fictional-owner'):
    browser = signed_browser(app, owner)
    response = exchange(browser, issue_code(browser, scope))
    assert response.status_code == 200, response.text
    return response.json['access_token']


def rpc(client, token, method='tools/list', params=None):
    return client.post('/chatgpt/mcp', json={
        'jsonrpc': '2.0', 'id': 1, 'method': method, 'params': params or {},
    }, headers={'Authorization': 'Bearer ' + token,
                'Accept': 'application/json, text/event-stream',
                'MCP-Protocol-Version': '2025-11-25'})


def call(client, token, name, arguments=None):
    return rpc(client, token, 'tools/call', {'name': name, 'arguments': arguments or {}})


@pytest.mark.parametrize('flag', [None, '', '0', 'false', 'off', 'no', ' FALSE '])
def test_run_flag_is_disabled_by_default_and_explicit_false(make_app, flag):
    app = make_app(flag)
    assert app.config['DREDGE_CHATGPT_RUN_TOOLS_ENABLED'] is False
    browser = signed_browser(app)
    rejected = browser.get('/chatgpt/oauth/authorize', base_url=ISSUER,
                           query_string=request_parameters(EXPANDED_SCOPE))
    assert rejected.status_code == 400
    assert rejected.json['error'] == 'invalid_scope'
    token = issue_token(app)
    with TestClient(create_chatgpt_app(app), base_url=ISSUER) as client:
        for path in ('/.well-known/oauth-authorization-server',
                     '/.well-known/oauth-protected-resource/chatgpt/mcp'):
            assert client.get(path).json()['scopes_supported'] == [BASE_SCOPE]
        assert {t['name'] for t in rpc(client, token).json()['result']['tools']} == OLD_TOOLS
        assert call(client, token, 'list_my_studio_runs').json()['result']['isError'] is True


@pytest.mark.parametrize('flag', ['1', 'yes', 'on', 'maybe', 'true false'])
def test_ambiguous_run_flag_fails_closed(make_app, flag):
    with pytest.raises(RuntimeError, match='DREDGE_CHATGPT_RUN_TOOLS_ENABLED'):
        make_app(flag)


def test_expanded_consent_and_grant_preserve_exact_requested_set(make_app):
    app = make_app('true')
    browser = signed_browser(app)
    base, _ = consent(browser)
    assert 'stored result summaries' not in base.text
    assert 'Scope: studio:read.' in base.text
    expanded, _ = consent(browser, RUN_SCOPE + ' ' + BASE_SCOPE)
    assert 'stored result summaries' in expanded.text
    assert 'run metadata' in expanded.text
    assert 'public_or_fictional' in expanded.text
    assert 'not automatically redacted' in expanded.text
    assert 'Scope: ' + EXPANDED_SCOPE + '.' in expanded.text
    assert f"form-action 'self' {CALLBACK};" in expanded.headers['Content-Security-Policy']
    token_response = exchange(browser, issue_code(browser, RUN_SCOPE + ' ' + BASE_SCOPE))
    assert token_response.status_code == 200, token_response.text
    assert token_response.json['scope'] == EXPANDED_SCOPE
    assert verify_access_token(app, token_response.json['access_token'])['scope'] == EXPANDED_SCOPE
    with app.extensions['studio_store'].connect() as db:
        assert {row[0] for row in db.execute('SELECT scope FROM chatgpt_oauth_codes')} == {EXPANDED_SCOPE}


@pytest.mark.parametrize('scope', ['', RUN_SCOPE, 'studio:write', BASE_SCOPE + ' studio:write',
                                  BASE_SCOPE + ' ' + BASE_SCOPE, ' ' + EXPANDED_SCOPE,
                                  EXPANDED_SCOPE + ' ', BASE_SCOPE + '  ' + RUN_SCOPE,
                                  BASE_SCOPE + '\t' + RUN_SCOPE, BASE_SCOPE + '\n' + RUN_SCOPE])
def test_scope_syntax_unknown_or_missing_base_rejected_before_consent(make_app, scope):
    app = make_app('true')
    response = signed_browser(app).get('/chatgpt/oauth/authorize', base_url=ISSUER,
                                      query_string=request_parameters(scope))
    assert response.status_code == 400
    assert response.json['error'] == 'invalid_scope'
    assert 'Location' not in response.headers
    with app.extensions['studio_store'].connect() as db:
        assert db.execute('SELECT COUNT(*) FROM chatgpt_oauth_consents').fetchone()[0] == 0


def test_existing_token_retains_two_tools_after_opt_in_and_new_consent_expands(make_app):
    original = make_app()
    old_token = issue_token(original)
    app = make_app(' TRUE ')
    assert app.config['DREDGE_CHATGPT_RUN_TOOLS_ENABLED'] is True
    new_token = issue_token(app, EXPANDED_SCOPE)
    assert verify_access_token(app, old_token)['scope'] == BASE_SCOPE
    with TestClient(create_chatgpt_app(app), base_url=ISSUER) as client:
        for path in ('/.well-known/oauth-authorization-server',
                     '/.well-known/oauth-protected-resource/chatgpt/mcp'):
            assert client.get(path).json()['scopes_supported'] == [BASE_SCOPE, RUN_SCOPE]
        new_tools = rpc(client, new_token).json()['result']['tools']
        assert {tool['name'] for tool in new_tools} == OLD_TOOLS | NEW_TOOLS
        for tool in new_tools:
            expected = [BASE_SCOPE, RUN_SCOPE] if tool['name'] in NEW_TOOLS else [BASE_SCOPE]
            assert tool['securitySchemes'] == [{'type': 'oauth2', 'scopes': expected}]
        for _ in range(2):
            assert {t['name'] for t in rpc(client, old_token).json()['result']['tools']} == OLD_TOOLS
            assert call(client, old_token, 'list_my_studio_runs').json()['result']['isError'] is True
            assert call(client, old_token, 'get_my_studio_run_result', {
                'run_id': '00000000-0000-0000-0000-000000000001',
            }).json()['result']['isError'] is True
            for name in OLD_TOOLS:
                assert call(client, old_token, name).json()['result']['isError'] is False
            assert call(client, new_token, 'list_my_studio_runs').json()['result']['isError'] is False


@pytest.mark.parametrize('invalidation', ['revoked', 'expired'])
def test_expanded_tokens_rejected_after_revocation_or_expiry(make_app, invalidation):
    app = make_app('true')
    token = issue_token(app, EXPANDED_SCOPE)
    with TestClient(create_chatgpt_app(app), base_url=ISSUER) as client:
        assert call(client, token, 'list_my_studio_runs').json()['result']['isError'] is False
        if invalidation == 'revoked':
            response = client.post('/chatgpt/oauth/revoke', data={
                'token': token, 'client_id': 'fictional-client',
            })
            assert response.status_code == 200
        else:
            with app.extensions['studio_store'].connect() as db:
                db.execute('UPDATE chatgpt_oauth_tokens SET expires_at=0')
        assert verify_access_token(app, token) is None
        assert rpc(client, token).status_code == 401
        assert call(client, token, 'list_my_studio_runs').status_code == 401


def test_opt_out_invalidates_expanded_codes_tokens_and_pending_consent(make_app):
    app = make_app('true')
    browser = signed_browser(app)
    _, pending = consent(browser, EXPANDED_SCOPE)
    code = issue_code(browser, EXPANDED_SCOPE)
    expanded = issue_token(app, EXPANDED_SCOPE)
    base = issue_token(app)
    app.config['DREDGE_CHATGPT_RUN_TOOLS_ENABLED'] = False
    assert browser.post('/chatgpt/oauth/authorize', base_url=ISSUER,
                        data={**pending, 'decision': 'allow'}).status_code == 403
    assert exchange(browser, code).status_code == 400
    assert verify_access_token(app, expanded) is None
    assert verify_access_token(app, base)['scope'] == BASE_SCOPE
    with TestClient(create_chatgpt_app(app), base_url=ISSUER) as client:
        assert client.get('/.well-known/oauth-protected-resource/chatgpt/mcp').json()[
            'scopes_supported'] == [BASE_SCOPE]
        assert client.get('/.well-known/oauth-authorization-server').json()[
            'scopes_supported'] == [BASE_SCOPE]
        assert rpc(client, expanded).status_code == 401
        assert {t['name'] for t in rpc(client, base).json()['result']['tools']} == OLD_TOOLS


@pytest.mark.parametrize('scope', [RUN_SCOPE, RUN_SCOPE + ' ' + BASE_SCOPE,
                                  EXPANDED_SCOPE + ' ', BASE_SCOPE + '  ' + RUN_SCOPE,
                                  EXPANDED_SCOPE + ' ' + RUN_SCOPE, EXPANDED_SCOPE + ' studio:write'])
def test_noncanonical_or_ungranted_stored_scope_is_rejected(make_app, scope):
    app = make_app('true')
    browser = signed_browser(app)
    code = issue_code(browser, EXPANDED_SCOPE)
    token = issue_token(app, EXPANDED_SCOPE)
    with app.extensions['studio_store'].connect() as db:
        db.execute('UPDATE chatgpt_oauth_tokens SET scope=?', (scope,))
        db.execute('UPDATE chatgpt_oauth_codes SET scope=? WHERE consumed_at IS NULL', (scope,))
    assert verify_access_token(app, token) is None
    assert exchange(browser, code).status_code == 400


def test_expanded_grant_cannot_swap_owner_client_resource_or_pkce(make_app):
    app = make_app('true')
    browser = signed_browser(app)
    _, pending = consent(browser, EXPANDED_SCOPE)
    with browser.session_transaction(base_url=ISSUER) as session:
        other = 'google:other-fictional-owner'
        _users[other] = User(other, 'Other', 'other@example.test', 'google')
        session['_user_id'] = other
    denied = browser.post('/chatgpt/oauth/authorize', base_url=ISSUER,
                          data={**pending, 'decision': 'allow'})
    assert denied.status_code == 403
    browser = signed_browser(app)
    code = issue_code(browser, EXPANDED_SCOPE)
    for changes in ({'client_id': 'other-client'}, {'resource': 'https://evil.example.test/mcp'},
                    {'redirect_uri': CALLBACK + '-other'}, {'code_verifier': 'b' * 64},
                    {'scope': BASE_SCOPE}):
        assert exchange(browser, code, **changes).status_code in {400, 401}
    response = exchange(browser, code)
    assert response.status_code == 200
    record = verify_access_token(app, response.json['access_token'])
    assert record['owner'] == 'github:fictional-owner'
    assert record['scope'] == EXPANDED_SCOPE


def supervisor_env(**changes):
    values = {
        'DREDGE_CHATGPT_TUNNEL_ENABLED': 'true', 'DREDGE_CHATGPT_ENABLED': 'true',
        'CHATGPT_ISSUER': ISSUER, 'CHATGPT_RESOURCE': RESOURCE,
        'CHATGPT_CLIENT_ID': 'fictional-client', 'CHATGPT_REDIRECT_URI': CALLBACK,
        'SECRET_KEY': 'fictional-test-key-at-least-32-bytes',
        'STUDIO_DB_PATH': '/data/studio/studio.sqlite3',
        'TUNNEL_TOKEN': 'fictional-not-a-real-tunnel-token',
    }
    values.update(changes)
    return values


@pytest.mark.parametrize('flag,expected', [(None, 'false'), ('false', 'false'),
                                         ('off', 'false'), ('true', 'true'), (' TRUE ', 'true')])
def test_supervisor_passes_only_normalized_optional_flag_to_app(flag, expected):
    source = supervisor_env()
    if flag is not None:
        source['DREDGE_CHATGPT_RUN_TOOLS_ENABLED'] = flag
    app, connector, health = supervisor.child_specs(supervisor.read_configuration(source), 42)
    assert app.environment['DREDGE_CHATGPT_RUN_TOOLS_ENABLED'] == expected
    assert 'DREDGE_CHATGPT_RUN_TOOLS_ENABLED' not in connector.environment
    assert 'DREDGE_CHATGPT_RUN_TOOLS_ENABLED' not in health.environment


@pytest.mark.parametrize('flag', ['yes', 'on', '1', 'maybe'])
def test_supervisor_rejects_ambiguous_optional_run_flag(flag):
    with pytest.raises(supervisor.ConfigurationError, match='DREDGE_CHATGPT_RUN_TOOLS_ENABLED'):
        supervisor.read_configuration(supervisor_env(DREDGE_CHATGPT_RUN_TOOLS_ENABLED=flag))


def test_expanded_calls_keep_owner_immutable_and_leave_store_unchanged(make_app):
    app = make_app('true')
    owner = 'github:fictional-owner'
    other = 'google:other-fictional-owner'
    own_token = issue_token(app, EXPANDED_SCOPE, owner)
    other_token = issue_token(app, EXPANDED_SCOPE, other)
    own_id = '00000000-0000-0000-0000-000000000001'
    other_id = '00000000-0000-0000-0000-000000000002'
    missing_id = '00000000-0000-0000-0000-000000000003'
    app.config['STUDIO_ROLES'] = {owner: 'admin'}
    with app.extensions['studio_store'].connect() as db:
        for principal, run_id, answer in ((owner, own_id, 'Fictional owner summary'),
                                          (other, other_id, 'Fictional other summary')):
            db.execute('''INSERT INTO runs(id,owner,status,payload,result,created,started,ended)
                       VALUES(?,?,?,?,?,?,?,?)''', (
                run_id, principal, 'completed',
                json.dumps({'execution_mode': 'live', 'data_scope': 'public_or_fictional'}),
                json.dumps({'answer': answer, 'token_usage': 42}), time.time(), 1, 2,
            ))
        before = '\n'.join(db.iterdump())
    with TestClient(create_chatgpt_app(app), base_url=ISSUER) as client:
        def owned_read(pair):
            token, expected_id, excluded_id, expected_text = pair
            listed = call(client, token, 'list_my_studio_runs').json()['result']
            assert listed['isError'] is False
            assert expected_id in json.dumps(listed)
            assert excluded_id not in json.dumps(listed)
            result = call(client, token, 'get_my_studio_run_result', {
                'run_id': expected_id,
            }).json()['result']
            assert result['isError'] is False
            assert result['structuredContent']['summary'] == expected_text
            return True

        pairs = [(own_token, own_id, other_id, 'Fictional owner summary'),
                 (other_token, other_id, own_id, 'Fictional other summary')]
        with ThreadPoolExecutor(max_workers=4) as pool:
            assert all(pool.map(owned_read, pairs * 4))
        denied = call(client, own_token, 'get_my_studio_run_result', {
            'run_id': other_id,
        }).json()['result']
        missing = call(client, own_token, 'get_my_studio_run_result', {
            'run_id': missing_id,
        }).json()['result']
        assert denied == missing
        assert other_id not in json.dumps(denied)
        assert 'Fictional other summary' not in json.dumps(denied)
        for name, arguments in (
            ('list_my_studio_runs', {'owner': other}),
            ('get_my_studio_run_result', {'run_id': own_id, 'owner': other}),
            ('get_my_studio_run_result', {'run_id': own_id, 'scopes': [BASE_SCOPE, RUN_SCOPE]}),
        ):
            assert call(client, own_token, name, arguments).json()['result']['isError'] is True
    with app.extensions['studio_store'].connect() as db:
        assert '\n'.join(db.iterdump()) == before


def test_new_scope_flag_cannot_enable_disabled_chatgpt(monkeypatch):
    from flask import Flask
    from dredge.chatgpt_transport import configure_chatgpt

    monkeypatch.setenv('DREDGE_CHATGPT_ENABLED', 'false')
    monkeypatch.setenv('DREDGE_CHATGPT_RUN_TOOLS_ENABLED', 'true')
    app = Flask(__name__)
    configure_chatgpt(app)
    assert app.config['DREDGE_CHATGPT_ENABLED'] is False
    assert app.config['DREDGE_CHATGPT_RUN_TOOLS_ENABLED'] is False
    assert 'chatgpt_oauth_server' not in app.extensions


@pytest.mark.parametrize('field,value', [('revoked_at', 1), ('expires_at', 0)])
def test_run_calls_recheck_authorization_after_http_gate(make_app, monkeypatch, field, value):
    from dredge import chatgpt_foundation, chatgpt_oauth

    app = make_app('true')
    token = issue_token(app, EXPANDED_SCOPE)
    real_verify = chatgpt_oauth.verify_access_token
    armed = False
    checks = []
    dispatches = []

    def verify_then_invalidate(candidate_app, candidate_token):
        record = real_verify(candidate_app, candidate_token)
        if armed:
            checks.append(record)
            if len(checks) == 1:
                # Return the gate's already-read record, then expire/revoke it
                # before execution. No auth/cache mock substitutes for the store.
                with app.extensions['studio_store'].connect() as db:
                    db.execute(f'UPDATE chatgpt_oauth_tokens SET {field}=?', (value,))
        return record

    real_dispatch = chatgpt_foundation.call_read_tool

    def track_dispatch(*args, **kwargs):
        dispatches.append(args)
        return real_dispatch(*args, **kwargs)

    monkeypatch.setattr(chatgpt_oauth, 'verify_access_token', verify_then_invalidate)
    monkeypatch.setattr(chatgpt_foundation, 'call_read_tool', track_dispatch)
    with TestClient(create_chatgpt_app(app), base_url=ISSUER) as client:
        assert rpc(client, token).status_code == 200
        armed = True
        response = call(client, token, 'list_my_studio_runs')
        assert response.status_code == 200
        assert response.json()['result']['isError'] is True
    assert checks[0] is not None
    assert checks[-1] is None
    assert len(checks) >= 2
    assert dispatches == []


def test_supported_scope_sets_are_isolated_between_apps(make_app):
    from dredge.chatgpt_foundation import supported_scopes

    enabled = make_app('true')
    disabled = make_app()
    assert supported_scopes(enabled) == frozenset({BASE_SCOPE, RUN_SCOPE})
    assert supported_scopes(disabled) == frozenset({BASE_SCOPE})
    for app, scopes in ((enabled, [BASE_SCOPE, RUN_SCOPE]), (disabled, [BASE_SCOPE])):
        assert app.extensions['chatgpt_oauth_server'].scopes_supported == scopes
        assert app.test_client().get('/.well-known/oauth-authorization-server',
                                     base_url=ISSUER).json['scopes_supported'] == scopes
    assert issue_token(enabled, EXPANDED_SCOPE)
    rejected = signed_browser(disabled).get('/chatgpt/oauth/authorize', base_url=ISSUER,
                                           query_string=request_parameters(EXPANDED_SCOPE))
    assert rejected.status_code == 400
