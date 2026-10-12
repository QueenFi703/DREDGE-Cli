"""Local OAuth integration checks. Accounts, redirects and tokens are fictional."""
import base64
import hashlib
import re
from urllib.parse import parse_qs, urlsplit

import pytest
from flask import Flask
from flask_login import LoginManager

from dredge.auth import User
from dredge.studio import StudioStore
from dredge.chatgpt_oauth import register_chatgpt_oauth, verify_access_token

ISSUER = 'https://dredge.example.test'
RESOURCE = ISSUER + '/chatgpt/mcp'
REDIRECT = 'https://chatgpt.example.test/connector/callback'
VERIFIER = 'a' * 64


def make_app(path):
    app = Flask(__name__)
    app.config.update(TESTING=True, SECRET_KEY='fictional-test-key',
                      DREDGE_CHATGPT_ENABLED=True, CHATGPT_ISSUER=ISSUER,
                      CHATGPT_RESOURCE=RESOURCE, CHATGPT_CLIENT_ID='fictional-chatgpt',
                      CHATGPT_REDIRECT_URI=REDIRECT)
    app.extensions['studio_store'] = StudioStore(path)
    manager = LoginManager(app)
    manager.user_loader(lambda owner: User(owner, 'Fictional owner', '', owner.split(':')[0]))
    register_chatgpt_oauth(app)
    return app


@pytest.fixture
def app(tmp_path):
    return make_app(tmp_path / 'studio.sqlite3')


def sign_in(client, owner='github:fictional-owner'):
    with client.session_transaction(base_url=ISSUER) as session:
        session['_user_id'] = owner
        session['_fresh'] = True


def parameters(**changes):
    result = dict(response_type='code', client_id='fictional-chatgpt', redirect_uri=REDIRECT,
                  scope='studio:read', state='fictional-state', resource=RESOURCE,
                  code_challenge=base64.urlsafe_b64encode(hashlib.sha256(VERIFIER.encode()).digest()).rstrip(b'=').decode(),
                  code_challenge_method='S256')
    result.update(changes)
    return result


def begin(client, **changes):
    response = client.get('/chatgpt/oauth/authorize', query_string=parameters(**changes), base_url=ISSUER)
    assert response.status_code == 200, response.get_data(as_text=True)
    fields = dict(re.findall(r'name="([^"]+)" value="([^"]*)"', response.get_data(as_text=True)))
    assert fields['transaction'] and fields['csrf_token']
    return fields


def authorize(client, **changes):
    fields = begin(client, **changes)
    response = client.post('/chatgpt/oauth/authorize', data={**fields, 'decision':'allow'}, base_url=ISSUER)
    assert response.status_code == 302, response.get_data(as_text=True)
    params = parse_qs(urlsplit(response.location).query)
    assert params['state'] == ['fictional-state']
    return params['code'][0]


def exchange(client, authorization_code, **changes):
    data = dict(grant_type='authorization_code', client_id='fictional-chatgpt', code=authorization_code,
                redirect_uri=REDIRECT, resource=RESOURCE, code_verifier=VERIFIER)
    data.update(changes)
    return client.post('/chatgpt/oauth/token', data=data, base_url=ISSUER)


def test_authorization_code_pkce_end_to_end(app):
    client = app.test_client()
    sign_in(client)
    code = authorize(client)
    response = exchange(client, code)
    assert response.status_code == 200, response.get_json()
    token = response.get_json()
    assert token['scope'] == 'studio:read'
    assert 'refresh_token' not in token
    verified = verify_access_token(app, token['access_token'])
    assert verified['owner'] == 'github:fictional-owner'
    assert verified['client_id'] == 'fictional-chatgpt'
    assert verified['resource'] == RESOURCE
    assert exchange(client, code).status_code == 400


def test_consent_policy_permits_registered_callback_after_form_submission(app):
    client = app.test_client()
    sign_in(client)
    response = client.get('/chatgpt/oauth/authorize', query_string=parameters(), base_url=ISSUER)
    policy = response.headers['Content-Security-Policy']
    assert f"form-action 'self' {REDIRECT};" in policy
    assert "default-src 'none'" in policy
    assert "frame-ancestors 'none'" in policy
    fields = dict(re.findall(r'name="([^"]+)" value="([^"]*)"', response.get_data(as_text=True)))
    approved = client.post('/chatgpt/oauth/authorize',
                           data={**fields, 'decision': 'allow'}, base_url=ISSUER)
    assert approved.status_code == 302
    assert approved.location.startswith(REDIRECT + '?')
    assert approved.headers['Content-Security-Policy'] == policy


def test_discovery_reveals_only_supported_grant_and_pkce(app):
    data = app.test_client().get('/.well-known/oauth-authorization-server', base_url=ISSUER).get_json()
    assert data['issuer'] == ISSUER
    assert data['grant_types_supported'] == ['authorization_code']
    assert data['code_challenge_methods_supported'] == ['S256']
    assert data['token_endpoint_auth_methods_supported'] == ['none']
    assert 'registration_endpoint' not in data


def test_authorization_accepts_chatgpt_locale_hint(app):
    client = app.test_client()
    sign_in(client)
    code = authorize(client, ui_locales='en-US')
    response = exchange(client, code)
    assert response.status_code == 200
    verified = verify_access_token(app, response.get_json()['access_token'])
    assert verified['owner'] == 'github:fictional-owner'
    assert response.get_json()['scope'] == 'studio:read'


@pytest.mark.parametrize('extra', [
    [('ui_locales', 'en-US'), ('ui_locales', 'fr')],
    [('ui_locales', 'x' * 2049)],
    [('unknown_parameter', 'en-US')],
])
def test_locale_support_keeps_parameter_boundary(app, extra):
    client = app.test_client()
    sign_in(client)
    response = client.get('/chatgpt/oauth/authorize',
                          query_string=list(parameters().items()) + extra, base_url=ISSUER)
    assert response.status_code == 400
    with app.extensions['studio_store'].connect() as db:
        assert db.execute('SELECT COUNT(*) FROM chatgpt_oauth_consents').fetchone()[0] == 0


def test_registration_disabled_without_explicit_switch(tmp_path):
    app = Flask(__name__)
    register_chatgpt_oauth(app)
    assert app.test_client().get('/.well-known/oauth-authorization-server').status_code == 404
    assert verify_access_token(app, 'anything') is None


@pytest.mark.parametrize('change', [
    {'client_id':'another-client'}, {'redirect_uri':'https://evil.example.test/callback'},
    {'redirect_uri':REDIRECT + '/extra'}, {'redirect_uri':''},
    {'response_type':'token'}, {'scope':'studio:write'}, {'scope':''},
    {'scope':'studio:read studio:write'}, {'resource':'https://evil.example.test/mcp'},
    {'resource':''}, {'state':''}, {'code_challenge':''},
    {'code_challenge_method':'plain'}, {'code_challenge_method':''},
    {'code_challenge':'a' * 42}, {'code_challenge':'a' * 44},
])
def test_invalid_authorization_never_redirects_or_persists_consent(app, change):
    client = app.test_client()
    sign_in(client)
    response = client.get('/chatgpt/oauth/authorize', query_string=parameters(**change), base_url=ISSUER)
    assert response.status_code in {400, 401}
    assert 'Location' not in response.headers
    with app.extensions['studio_store'].connect() as db:
        assert db.execute('SELECT COUNT(*) FROM chatgpt_oauth_consents').fetchone()[0] == 0


@pytest.mark.parametrize('field', ['client_id','redirect_uri','resource','scope','state','code_challenge','code_challenge_method'])
def test_duplicate_authorization_parameters_rejected(app, field):
    client = app.test_client()
    sign_in(client)
    query = list(parameters().items()) + [(field, parameters()[field])]
    response = client.get('/chatgpt/oauth/authorize', query_string=query, base_url=ISSUER)
    assert response.status_code == 400


def test_consent_login_resume_is_browser_bound(app):
    client = app.test_client()
    response = client.get('/chatgpt/oauth/authorize', query_string=parameters(), base_url=ISSUER)
    assert response.location == '/auth/login'
    with client.session_transaction(base_url=ISSUER) as saved:
        resume = saved['chatgpt_return_to']
    assert re.fullmatch(r'/chatgpt/oauth/authorize\?transaction=[A-Za-z0-9_-]{43}', resume)
    other_browser = app.test_client()
    sign_in(other_browser)
    assert other_browser.get(resume, base_url=ISSUER).status_code == 403
    sign_in(client, 'google:fictional-owner')
    response = client.get(resume, base_url=ISSUER)
    assert response.status_code == 200
    assert 'google:fictional-owner' in response.get_data(as_text=True)


@pytest.mark.parametrize('change', [
    {'client_id':'another-client'}, {'resource':'https://evil.example.test/mcp'},
    {'redirect_uri':'https://evil.example.test/callback'}, {'scope':'studio:write'},
    {'owner':'github:other'}, {'state':'attacker-state'},
    {'code_challenge':'a' * 43},
])
def test_consent_cannot_swap_approved_parameters(app, change):
    client = app.test_client()
    sign_in(client)
    fields = begin(client)
    response = client.post('/chatgpt/oauth/authorize', data={**fields, 'decision':'allow', **change}, base_url=ISSUER)
    assert response.status_code == 400
    with app.extensions['studio_store'].connect() as db:
        assert db.execute('SELECT COUNT(*) FROM chatgpt_oauth_codes').fetchone()[0] == 0


@pytest.mark.parametrize('csrf', ['', 'wrong', 'é' * 43])
def test_consent_rejects_missing_mismatched_and_unicode_csrf(app, csrf):
    client = app.test_client()
    sign_in(client)
    fields = begin(client)
    response = client.post('/chatgpt/oauth/authorize', data={**fields, 'csrf_token':csrf, 'decision':'allow'}, base_url=ISSUER)
    assert response.status_code == 403


def test_consent_bound_to_shown_account_and_authentication(app):
    client = app.test_client()
    sign_in(client)
    fields = begin(client)
    sign_in(client, 'github:other')
    response = client.post('/chatgpt/oauth/authorize', data={**fields, 'decision':'allow'}, base_url=ISSUER)
    assert response.status_code == 403
    with client.session_transaction(base_url=ISSUER) as saved:
        saved.pop('_user_id')
    response = client.post('/chatgpt/oauth/authorize', data={**fields, 'decision':'allow'}, base_url=ISSUER)
    assert response.status_code == 401


@pytest.mark.parametrize('decision', ['allow','deny'])
def test_consent_consumed_once_and_deny_issues_no_code(app, decision):
    client = app.test_client()
    sign_in(client)
    fields = begin(client)
    response = client.post('/chatgpt/oauth/authorize', data={**fields, 'decision':decision}, base_url=ISSUER)
    assert response.status_code == 302
    params = parse_qs(urlsplit(response.location).query)
    assert params['state'] == ['fictional-state']
    assert params['iss'] == [ISSUER]
    if decision == 'deny':
        assert params['error'] == ['access_denied']
        assert 'code' not in params
    assert client.post('/chatgpt/oauth/authorize', data={**fields, 'decision':'allow'}, base_url=ISSUER).status_code == 403
    with app.extensions['studio_store'].connect() as db:
        assert db.execute('SELECT COUNT(*) FROM chatgpt_oauth_codes').fetchone()[0] == (1 if decision == 'allow' else 0)


@pytest.mark.parametrize('changes', [
    {'code_verifier':''}, {'code_verifier':'b' * 64}, {'code_verifier':'a' * 42},
    {'redirect_uri':REDIRECT + '/extra'}, {'redirect_uri':''}, {'client_id':'another-client'},
    {'resource':'https://evil.example.test/mcp'}, {'resource':''},
    {'scope':'studio:write'}, {'grant_type':'refresh_token'}, {'code':'not-a-code'},
    {'client_secret':'unrequested-secret'},
])
def test_token_exchange_fails_closed_without_consuming_valid_code(app, changes):
    client = app.test_client()
    sign_in(client)
    code = authorize(client)
    invalid = exchange(client, code, **changes)
    assert invalid.status_code in {400, 401}
    with app.extensions['studio_store'].connect() as db:
        assert db.execute('SELECT COUNT(*) FROM chatgpt_oauth_tokens').fetchone()[0] == 0
    assert exchange(client, code).status_code == 200


@pytest.mark.parametrize('field', ['client_id','code','redirect_uri','resource','code_verifier','grant_type'])
def test_duplicate_token_parameters_rejected(app, field):
    client = app.test_client()
    sign_in(client)
    code = authorize(client)
    data = dict(grant_type='authorization_code', client_id='fictional-chatgpt', code=code,
                redirect_uri=REDIRECT, resource=RESOURCE, code_verifier=VERIFIER)
    from werkzeug.datastructures import MultiDict
    payload = MultiDict([*data.items(), (field, data[field])])
    response = client.post('/chatgpt/oauth/token', data=payload, base_url=ISSUER)
    assert response.status_code == 400


def test_hashes_only_persist_and_tokens_survive_restart(app):
    client = app.test_client()
    sign_in(client)
    code = authorize(client)
    raw_token = exchange(client, code).get_json()['access_token']
    with app.extensions['studio_store'].connect() as db:
        stored_code = dict(db.execute('SELECT * FROM chatgpt_oauth_codes').fetchone())
        stored_token = dict(db.execute('SELECT * FROM chatgpt_oauth_tokens').fetchone())
    assert stored_code['code_hash'] == hashlib.sha256(code.encode()).hexdigest()
    assert stored_token['token_hash'] == hashlib.sha256(raw_token.encode()).hexdigest()
    assert raw_token not in str(stored_token) and code not in str(stored_code)
    restarted = make_app(app.extensions['studio_store'].path)
    assert verify_access_token(restarted, raw_token)['owner'] == 'github:fictional-owner'


def test_codes_survive_restart_and_expire(app):
    client = app.test_client()
    sign_in(client)
    code = authorize(client)
    restarted = make_app(app.extensions['studio_store'].path)
    assert exchange(restarted.test_client(), code).status_code == 200
    expired_code = authorize(client)
    with app.extensions['studio_store'].connect() as db:
        db.execute('UPDATE chatgpt_oauth_codes SET expires_at=1 WHERE consumed_at IS NULL')
    assert exchange(client, expired_code).status_code == 400


def test_consent_expires_and_configuration_change_invalidates_it(app):
    client = app.test_client()
    sign_in(client)
    fields = begin(client)
    with app.extensions['studio_store'].connect() as db:
        db.execute('UPDATE chatgpt_oauth_consents SET expires_at=1')
    assert client.post('/chatgpt/oauth/authorize', data={**fields, 'decision':'allow'}, base_url=ISSUER).status_code == 403
    fields = begin(client)
    app.config['CHATGPT_REDIRECT_URI'] = 'https://chatgpt.example.test/different-callback'
    assert client.post('/chatgpt/oauth/authorize', data={**fields, 'decision':'allow'}, base_url=ISSUER).status_code == 403


@pytest.mark.parametrize('column,value', [
    ('issuer','https://other.example.test'), ('resource','https://other.example.test/mcp'),
    ('client_id','another-client'), ('scope','studio:write'), ('scope','studio:read studio:write'),
    ('owner','local:fictional-owner'), ('expires_at',1), ('revoked_at',1),
])
def test_verifier_rejects_every_mismatched_token_claim(app, column, value):
    client = app.test_client()
    sign_in(client)
    code = authorize(client)
    token = exchange(client, code).get_json()['access_token']
    assert verify_access_token(app, token) is not None
    with app.extensions['studio_store'].connect() as db:
        db.execute(f'UPDATE chatgpt_oauth_tokens SET {column}=?', (value,))
    assert verify_access_token(app, token) is None


def test_token_revocation_is_durable_idempotent_and_client_bound(app):
    client = app.test_client()
    sign_in(client)
    token = exchange(client, authorize(client)).get_json()['access_token']
    data = dict(token=token, client_id='wrong-client', token_type_hint='access_token')
    assert client.post('/chatgpt/oauth/revoke', data=data, base_url=ISSUER).status_code == 400
    assert verify_access_token(app, token) is not None
    data['client_id'] = 'fictional-chatgpt'
    assert client.post('/chatgpt/oauth/revoke', data=data, base_url=ISSUER).status_code == 200
    assert verify_access_token(app, token) is None
    restarted = make_app(app.extensions['studio_store'].path)
    assert verify_access_token(restarted, token) is None
    assert client.post('/chatgpt/oauth/revoke', data=data, base_url=ISSUER).status_code == 200
    data['token'] = 'never-existed'
    assert client.post('/chatgpt/oauth/revoke', data=data, base_url=ISSUER).status_code == 200


def test_concurrent_code_redeemers_only_receive_one_token(app):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    client = app.test_client()
    sign_in(client)
    code = authorize(client)
    barrier = Barrier(2)

    def redeem():
        peer = app.test_client()
        barrier.wait(timeout=5)
        return exchange(peer, code)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(redeem) for _ in range(2)]
        responses = [future.result(timeout=10) for future in futures]
    assert sorted(response.status_code for response in responses) == [200,400]
    with app.extensions['studio_store'].connect() as db:
        assert db.execute('SELECT COUNT(*) FROM chatgpt_oauth_tokens').fetchone()[0] == 1
    token = next(response.get_json()['access_token'] for response in responses if response.status_code == 200)
    # A detected replay revokes the one issued credential, per RFC 6749 guidance.
    assert verify_access_token(app, token) is None


def test_token_persistence_failure_rolls_back_code_consumption(app):
    import sqlite3
    client = app.test_client()
    sign_in(client)
    code = authorize(client)
    with app.extensions['studio_store'].connect() as db:
        db.execute("CREATE TRIGGER fail_oauth_token BEFORE INSERT ON chatgpt_oauth_tokens BEGIN SELECT RAISE(ABORT, 'simulated disk failure'); END;")
    with pytest.raises(sqlite3.IntegrityError, match='simulated disk failure'):
        exchange(client, code)
    with app.extensions['studio_store'].connect() as db:
        assert db.execute('SELECT consumed_at FROM chatgpt_oauth_codes').fetchone()[0] is None
        assert db.execute('SELECT COUNT(*) FROM chatgpt_oauth_tokens').fetchone()[0] == 0
        db.execute('DROP TRIGGER fail_oauth_token')
    assert exchange(client, code).status_code == 200


def test_oauth_tokens_never_appear_in_debug_logs(app, caplog):
    import logging
    client = app.test_client()
    sign_in(client)
    with caplog.at_level(logging.DEBUG):
        code = authorize(client)
        token = exchange(client, code).get_json()['access_token']
    assert token not in caplog.text
    assert code not in caplog.text


def test_secure_headers_host_scheme_and_body_limits(app):
    client = app.test_client()
    sign_in(client)
    response = client.get('/chatgpt/oauth/authorize', query_string=parameters(), base_url=ISSUER)
    assert response.headers['Cache-Control'] == 'no-store'
    assert response.headers['Referrer-Policy'] == 'no-referrer'
    assert response.headers['X-Frame-Options'] == 'DENY'
    assert "frame-ancestors 'none'" in response.headers['Content-Security-Policy']
    assert client.get('/.well-known/oauth-authorization-server', base_url='http://dredge.example.test').status_code == 400
    assert client.get('/.well-known/oauth-authorization-server', base_url='https://evil.example.test').status_code == 400
    assert client.post('/chatgpt/oauth/token', data='x' * 17000, base_url=ISSUER).status_code == 413


def test_verifier_accepts_neither_cookies_nor_unissued_credentials(app):
    client = app.test_client()
    sign_in(client)
    for value in (None, 123, '', 'a' * 42, 'a' * 43, 'a' * 44, 'é' * 43):
        assert verify_access_token(app, value) is None
    token = exchange(client, authorize(client)).get_json()['access_token']
    app.config['DREDGE_CHATGPT_ENABLED'] = False
    assert verify_access_token(app, token) is None
    assert exchange(client, 'any').status_code == 503


@pytest.mark.parametrize('redirect_uri', ['é', '"', '\r\n', REDIRECT + '?x="', 'https://☃.example/callback'])
def test_malicious_redirect_values_fail_without_error_reflection(app, redirect_uri):
    client = app.test_client()
    response = client.get('/chatgpt/oauth/authorize', query_string=parameters(redirect_uri=redirect_uri), base_url=ISSUER)
    assert response.status_code == 400
    assert 'Location' not in response.headers


def test_cross_browser_consent_cannot_be_replayed_with_local_csrf(app):
    owner_browser = app.test_client()
    attacker_browser = app.test_client()
    sign_in(owner_browser)
    sign_in(attacker_browser, 'github:attacker')
    owner_fields = begin(owner_browser)
    attacker_fields = begin(attacker_browser)
    response = attacker_browser.post('/chatgpt/oauth/authorize', data={
        'transaction':owner_fields['transaction'], 'csrf_token':attacker_fields['csrf_token'],
        'decision':'allow',
    }, base_url=ISSUER)
    assert response.status_code == 403


def test_concurrent_consent_submissions_issue_one_code(app):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    client = app.test_client()
    sign_in(client)
    fields = begin(client)
    cookie = client.get_cookie('session', domain='dredge.example.test')
    barrier = Barrier(2)

    def submit():
        peer = app.test_client()
        peer.set_cookie('session', cookie.value, domain='dredge.example.test')
        barrier.wait(timeout=5)
        return peer.post('/chatgpt/oauth/authorize', data={**fields, 'decision':'allow'}, base_url=ISSUER)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(submit) for _ in range(2)]
        responses = [future.result(timeout=10) for future in futures]
    assert sorted(response.status_code for response in responses) == [302,403]
    with app.extensions['studio_store'].connect() as db:
        assert db.execute('SELECT COUNT(*) FROM chatgpt_oauth_codes').fetchone()[0] == 1


@pytest.mark.parametrize('data', [
    {'client_id':'fictional-chatgpt'},
    {'client_id':'fictional-chatgpt','token':'fictional','token_type_hint':'refresh_token'},
    {'client_id':'fictional-chatgpt','token':'fictional','scope':'studio:read'},
])
def test_invalid_revocation_payload_fails_harmlessly(app, data):
    client = app.test_client()
    assert client.post('/chatgpt/oauth/revoke', data=data, base_url=ISSUER).status_code in {400, 401}


def test_code_and_bearer_are_bound_to_current_registered_client(app):
    client = app.test_client()
    sign_in(client)
    first = authorize(client)
    token = exchange(client, first).get_json()['access_token']
    code = authorize(client)
    app.config['CHATGPT_CLIENT_ID'] = 'new-registration'
    assert verify_access_token(app, token) is None
    assert exchange(client, code, client_id='new-registration').status_code == 400


@pytest.mark.parametrize('config', [
    {'CHATGPT_ISSUER':'http://dredge.example.test'},
    {'CHATGPT_ISSUER':ISSUER + '/'}, {'CHATGPT_ISSUER':ISSUER + '/issuer'},
    {'CHATGPT_ISSUER':'https://user:secret@dredge.example.test'},
    {'CHATGPT_RESOURCE':'https://another.example.test/mcp'},
    {'CHATGPT_REDIRECT_URI':'https://chatgpt.example.test/callback#fragment'},
    {'CHATGPT_REDIRECT_URI':'https://chatgpt.example.test/callback?query=1'},
    {'CHATGPT_REDIRECT_URI':'http://chatgpt.example.test/callback'},
    {'CHATGPT_CLIENT_ID':''}, {'CHATGPT_CLIENT_ID':None},
])
def test_registration_rejects_unsafe_configuration(tmp_path, config):
    app = Flask(__name__)
    app.config.update(TESTING=True, SECRET_KEY='fictional-test-key',
                      DREDGE_CHATGPT_ENABLED=True, CHATGPT_ISSUER=ISSUER,
                      CHATGPT_RESOURCE=RESOURCE, CHATGPT_CLIENT_ID='fictional-chatgpt',
                      CHATGPT_REDIRECT_URI=REDIRECT)
    app.config.update(config)
    app.extensions['studio_store'] = StudioStore(tmp_path / 'studio.sqlite3')
    with pytest.raises(RuntimeError, match='CHATGPT_'):
        register_chatgpt_oauth(app)


@pytest.mark.parametrize('bad_verifier', ['a' * 64 + '\n', 'a' * 64 + '\r', 'a' * 42, 'a' * 129, 'a' * 63 + 'é'])
def test_verifier_must_have_exact_rfc7636_unreserved_shape(app, bad_verifier):
    client = app.test_client()
    sign_in(client)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(bad_verifier.encode()).digest()).rstrip(b'=').decode()
    code = authorize(client, code_challenge=challenge)
    response = exchange(client, code, code_verifier=bad_verifier)
    assert response.status_code == 400
    with app.extensions['studio_store'].connect() as db:
        assert db.execute('SELECT COUNT(*) FROM chatgpt_oauth_tokens').fetchone()[0] == 0


@pytest.mark.parametrize('field', ['response_type','client_id','scope','state','resource','code_challenge_method'])
@pytest.mark.parametrize('value', ['é','"','\r\n','\x00'])
def test_authorization_malformed_input_matrix_returns_safe_errors(app, field, value):
    client = app.test_client()
    sign_in(client)
    response = client.get('/chatgpt/oauth/authorize', query_string=parameters(**{field:value}), base_url=ISSUER)
    # State is deliberately opaque. It is preserved in the transaction, never rendered.
    if field == 'state':
        assert response.status_code == 200
    else:
        assert response.status_code in {400,401}
        assert 'Location' not in response.headers


@pytest.mark.parametrize('field', ['grant_type','client_id','code','resource','redirect_uri','code_verifier'])
@pytest.mark.parametrize('value', ['é','"','\r\n','\x00'])
def test_token_malformed_input_matrix_returns_safe_errors(app, field, value):
    client = app.test_client()
    sign_in(client)
    code = authorize(client)
    response = exchange(client, code, **{field:value})
    assert response.status_code in {400,401}
    assert exchange(client, code).status_code == 200


def test_opaque_state_is_preserved_without_html_or_header_injection(app):
    client = app.test_client()
    sign_in(client)
    state = '<script>alert("fictional")</script>\r\nX-Injected: true & é'
    fields = begin(client, state=state)
    response = client.post('/chatgpt/oauth/authorize', data={**fields, 'decision':'deny'}, base_url=ISSUER)
    assert response.status_code == 302
    params = parse_qs(urlsplit(response.location).query)
    assert params['state'] == [state]
    assert 'X-Injected' not in response.headers
