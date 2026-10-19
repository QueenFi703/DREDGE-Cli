"""Synthetic, in-process diagnostics checks. No provider or live OAuth requests."""
import hashlib
import json
import logging
import sqlite3
import subprocess
import sys
from contextlib import contextmanager
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest
from flask import g

from dredge import chatgpt_oauth as oauth
from test_chatgpt_oauth import ISSUER, VERIFIER, begin, exchange, make_app, sign_in


@pytest.fixture(autouse=True)
def capture_isolated_logger(caplog):
    oauth._CONSENT_LOG.addHandler(caplog.handler)
    yield
    oauth._CONSENT_LOG.removeHandler(caplog.handler)


@pytest.fixture
def app(tmp_path):
    return make_app(tmp_path / 'diagnostics.sqlite3')


def events(caplog):
    return [json.loads(record.getMessage()) for record in caplog.records
            if record.name == 'dredge.oauth.consent']


def submit(client, fields, **changes):
    return client.post(oauth.AUTHORIZE_PATH,
                       data={**fields, 'decision': 'allow', **changes}, base_url=ISSUER)


def database_state(app):
    with sqlite3.connect(app.extensions['studio_store'].path) as db:
        return (db.execute('SELECT consumed_at FROM chatgpt_oauth_consents').fetchone()[0],
                db.execute('SELECT COUNT(*) FROM chatgpt_oauth_codes').fetchone()[0])


def test_event_order_reports_commit_only_after_persistence(app, caplog, monkeypatch):
    caplog.set_level(logging.INFO, logger='dredge.oauth.consent')
    client = app.test_client()
    sign_in(client)
    fields = begin(client)
    get_id = events(caplog)[-1]['request_id']
    caplog.clear()
    persisted = []
    original_log = oauth._CONSENT_LOG.info

    def capture(message):
        persisted.append((json.loads(message)['event'], database_state(app)))
        original_log(message)

    monkeypatch.setattr(oauth._CONSENT_LOG, 'info', capture)
    response = submit(client, fields)
    assert response.status_code == 302
    recorded = events(caplog)
    assert [item['event'] for item in recorded] == [
        'consent_marked_used', 'response_created', 'transaction_committed',
        'http_response_prepared',
    ]
    assert [item['commit_phase'] for item in recorded] == [
        'pending', 'pending', 'committed', 'committed',
    ]
    assert persisted[0][1] == persisted[1][1] == (None, 0)
    assert persisted[2][1][0] is not None and persisted[2][1][1] == 1
    assert len({item['request_id'] for item in recorded}) == 1
    assert recorded[-1]['request_id'] != get_id
    assert recorded[-1]['outcome'] == 'allow_response'
    assert recorded[-1]['status'] == 302
    assert recorded[-1]['age_seconds'] in {0, 1}
    assert 'request_id' not in response.headers
    assert 'no-store' == response.headers['Cache-Control']


@pytest.mark.parametrize('reason', ['missing_consent', 'expired', 'already_consumed', 'binding_mismatch'])
def test_same_public_error_has_distinct_private_reason(app, caplog, reason):
    caplog.set_level(logging.INFO, logger='dredge.oauth.consent')
    client = app.test_client()
    sign_in(client)
    fields = begin(client)
    with app.extensions['studio_store'].connect() as db:
        if reason == 'missing_consent':
            db.execute('DELETE FROM chatgpt_oauth_consents')
        elif reason == 'expired':
            db.execute('UPDATE chatgpt_oauth_consents SET expires_at=1')
        elif reason == 'already_consumed':
            db.execute('UPDATE chatgpt_oauth_consents SET consumed_at=1')
        else:
            db.execute('UPDATE chatgpt_oauth_consents SET binding_hash=?', ('0' * 64,))
    caplog.clear()
    response = submit(client, fields)
    assert response.status_code == 403
    assert response.get_json() == {
        'error': 'access_denied',
        'error_description': 'This consent request expired or is no longer valid.',
    }
    assert events(caplog)[-1]['outcome'] == reason
    assert events(caplog)[-1]['status'] == 403


@pytest.mark.parametrize('reason,status', [
    ('missing_binding', 403), ('csrf_mismatch', 403), ('owner_changed', 403),
    ('login_required', 401), ('configuration_changed', 403), ('invalid_decision', 400),
])
def test_other_security_rejections_remain_unchanged(app, caplog, reason, status):
    caplog.set_level(logging.INFO, logger='dredge.oauth.consent')
    client = app.test_client()
    sign_in(client)
    fields = begin(client)
    changes = {}
    if reason in {'missing_binding', 'login_required'}:
        with client.session_transaction(base_url=ISSUER) as saved:
            saved.pop('chatgpt_oauth_binding' if reason == 'missing_binding' else '_user_id')
    elif reason == 'csrf_mismatch':
        changes['csrf_token'] = 'invalid'
    elif reason == 'owner_changed':
        sign_in(client, 'github:different-fictional-owner')
    elif reason == 'configuration_changed':
        app.config['CHATGPT_REDIRECT_URI'] += '/changed'
    else:
        changes['decision'] = 'invalid'
    caplog.clear()
    assert submit(client, fields, **changes).status_code == status
    assert events(caplog)[-1]['outcome'] == reason
    assert database_state(app) == (None, 0)


@pytest.mark.parametrize('failure', ['response_creation', 'commit'])
def test_failed_response_or_commit_rolls_back_without_false_success(app, caplog, monkeypatch, failure):
    caplog.set_level(logging.INFO, logger='dredge.oauth.consent')
    client = app.test_client()
    sign_in(client)
    fields = begin(client)
    caplog.clear()
    canary = 'CANARY-exception-secret-not-for-logs'
    store = app.extensions['studio_store']
    original_connect = store.connect

    if failure == 'response_creation':
        def fail(*args, **kwargs):
            raise RuntimeError(canary)
        monkeypatch.setattr(app.extensions['chatgpt_oauth_server'], 'create_authorization_response', fail)
    else:
        @contextmanager
        def fail_commit():
            with original_connect() as db:
                yield db
                raise RuntimeError(canary)
        monkeypatch.setattr(store, 'connect', fail_commit)

    with pytest.raises(RuntimeError, match=canary):
        submit(client, fields)
    recorded = events(caplog)
    assert recorded[-1]['event'] == 'transaction_failed'
    assert recorded[-1]['commit_phase'] == 'failed'
    assert not any(item['event'] == 'transaction_committed' for item in recorded)
    assert any(item['event'] == 'response_created' for item in recorded) == (failure == 'commit')
    assert database_state(app) == (None, 0)
    assert canary not in caplog.text


@pytest.mark.parametrize('failure', ['logger', 'request_id', 'serializer', 'configuration'])
@pytest.mark.parametrize('decision', ['allow', 'deny'])
def test_diagnostic_failures_cannot_break_authorization(app, monkeypatch, failure, decision):
    def fail(*args, **kwargs):
        raise RuntimeError('synthetic diagnostic failure')

    if failure == 'logger':
        monkeypatch.setattr(oauth._CONSENT_LOG, 'info', fail)
    elif failure == 'request_id':
        monkeypatch.setattr(oauth.uuid, 'uuid4', fail)
    elif failure == 'configuration':
        monkeypatch.setattr(oauth._CONSENT_LOG, 'setLevel', fail)
        oauth._configure_consent_logging()
    else:
        original_dumps = oauth.json.dumps
        def fail_diagnostic_only(value, *args, **kwargs):
            if isinstance(value, dict) and value.get('schema') == 'dredge.oauth.consent.v1':
                fail()
            return original_dumps(value, *args, **kwargs)
        monkeypatch.setattr(oauth.json, 'dumps', fail_diagnostic_only)
    client = app.test_client()
    sign_in(client)
    fields = begin(client)
    response = submit(client, fields, decision=decision)
    assert response.status_code == 302
    params = parse_qs(urlsplit(response.location).query)
    if decision == 'allow':
        assert exchange(client, params['code'][0]).status_code == 200
    else:
        assert params['error'] == ['access_denied'] and 'code' not in params
    assert submit(client, fields).status_code == 403


def test_no_credentials_private_data_or_hashes_in_diagnostics(app, caplog):
    caplog.set_level(logging.DEBUG)
    client = app.test_client()
    owner = 'github:CANARY-private-owner'
    state = 'CANARY-oauth-state-private'
    sign_in(client, owner)
    fields = begin(client, state=state)
    with client.session_transaction(base_url=ISSUER) as saved:
        binding = saved['chatgpt_oauth_binding']
    cookie = client.get_cookie('session', domain='dredge.example.test').value
    response = client.post(oauth.AUTHORIZE_PATH, data={**fields, 'decision': 'allow'},
                           headers={'X-Request-ID': 'CANARY-untrusted-request-id'}, base_url=ISSUER)
    assert response.status_code == 302
    code = parse_qs(urlsplit(response.location).query)['code'][0]
    token = exchange(client, code).get_json()
    secrets = [owner, state, fields['transaction'], fields['csrf_token'], binding, cookie,
               code, token['access_token'], token['refresh_token'], VERIFIER,
               'CANARY-untrusted-request-id', response.location]
    for value in secrets:
        assert value not in caplog.text
        assert hashlib.sha256(value.encode()).hexdigest() not in caplog.text
    allowed = {'schema', 'event', 'request_id', 'method', 'outcome', 'commit_phase',
               'transaction_stage', 'age_seconds', 'status', 'response_kind',
               'callback_origin_matches', 'callback_path_matches'}
    assert events(caplog)
    assert all(set(item) <= allowed for item in events(caplog))


def test_emitter_drops_unknown_fields_and_values(app, caplog):
    caplog.set_level(logging.INFO, logger='dredge.oauth.consent')
    with app.test_request_context(oauth.AUTHORIZE_PATH, base_url=ISSUER):
        oauth._consent_start()
        g.chatgpt_consent_diagnostic.update({
            'outcome': 'CANARY-unknown-outcome', 'method': 'CANARY-method',
            'private': 'CANARY-private-field', 'age_seconds': 'CANARY-not-numeric',
        })
        oauth._consent_emit('http_response_prepared', status='CANARY-not-status')
        oauth._consent_emit('CANARY-unknown-event')
    assert len(events(caplog)) == 1
    assert set(events(caplog)[0]) == {'schema', 'event', 'request_id', 'commit_phase', 'transaction_stage'}
    assert 'CANARY' not in caplog.text


@pytest.mark.parametrize('age,status', [(599, 302), (600, 403)])
def test_original_ten_minute_consent_boundary_is_unchanged(app, caplog, monkeypatch, age, status):
    caplog.set_level(logging.INFO, logger='dredge.oauth.consent')
    now = 1800000000
    monkeypatch.setattr(oauth, 'time', SimpleNamespace(time=lambda: now))
    client = app.test_client()
    sign_in(client)
    fields = begin(client)
    with app.extensions['studio_store'].connect() as db:
        expiry = db.execute('SELECT expires_at FROM chatgpt_oauth_consents').fetchone()[0]
    assert expiry == now + 600
    now += age
    caplog.clear()
    assert submit(client, fields).status_code == status
    assert events(caplog)[-1]['age_seconds'] == age


def test_default_uvicorn_tunnel_factory_emits_only_dedicated_json(tmp_path):
    script = '''
import json, logging, logging.config, os, socket, sys
from uvicorn.config import LOGGING_CONFIG
logging.config.dictConfig(LOGGING_CONFIG)
def forbidden(*args, **kwargs):
    raise AssertionError('No network calls in startup diagnostics test')
socket.socket.connect = forbidden
socket.socket.connect_ex = forbidden
socket.getaddrinfo = forbidden
os.environ.update({
    'DREDGE_CHATGPT_ENABLED': 'true', 'DREDGE_CHATGPT_TUNNEL_ENABLED': 'true',
    'CHATGPT_ISSUER': 'https://dredge.example.test',
    'CHATGPT_RESOURCE': 'https://dredge.example.test/chatgpt/mcp',
    'CHATGPT_CLIENT_ID': 'fictional-client',
    'CHATGPT_REDIRECT_URI': 'https://chatgpt.com/connector/oauth/fictional',
    'SECRET_KEY': 'fictional-test-session-key-at-least-32-bytes',
    'STUDIO_DB_PATH': sys.argv[1],
})
before = (logging.getLogger().level, len(logging.getLogger().handlers))
from dredge import server, chatgpt_tunnel
captured = []
original_create = server.create_app
def capture_app():
    app = original_create()
    captured.append(app)
    return app
server.create_app = capture_app
# Test startup wiring only, without emulating the real OS socket boundary.
chatgpt_tunnel._verify_socket = lambda: None
chatgpt_tunnel.create_tunnel_app()
sys.stderr.write('DIAGNOSTIC_REQUEST_START\\n')
response = captured[0].test_client().get('/chatgpt/oauth/authorize',
                                        base_url='https://dredge.example.test')
assert response.status_code == 400
after = (logging.getLogger().level, len(logging.getLogger().handlers))
assert before == after
print(json.dumps({'root_level': after[0], 'root_handlers': after[1]}))
'''
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path / 'child.sqlite3')],
                            capture_output=True, text=True, timeout=20, check=True)
    assert json.loads(result.stdout) == {'root_level': logging.WARNING, 'root_handlers': 0}
    request_output = result.stderr.split('DIAGNOSTIC_REQUEST_START\n', 1)[1]
    records = [json.loads(line) for line in request_output.splitlines()]
    assert records and all(item['schema'] == 'dredge.oauth.consent.v1' for item in records)
    assert records[-1]['event'] == 'http_response_prepared'
    assert records[-1]['status'] == 400


def test_deny_event_sequence_reports_no_code(app, caplog):
    client = app.test_client()
    sign_in(client)
    fields = begin(client)
    caplog.clear()
    response = submit(client, fields, decision='deny')
    assert response.status_code == 302
    recorded = events(caplog)
    assert [item['event'] for item in recorded] == [
        'consent_marked_used', 'response_created', 'transaction_committed',
        'http_response_prepared',
    ]
    assert recorded[-1]['outcome'] == 'deny_response'
    assert database_state(app)[1] == 0


def test_stream_failure_does_not_print_exception_or_break_auth(app, capsys, monkeypatch):
    class BrokenStream:
        def write(self, text):
            raise OSError('CANARY-private-logging-exception')

        def flush(self):
            pass

    for handler in oauth._CONSENT_LOG.handlers:
        if isinstance(handler, oauth._ConsentLogHandler):
            monkeypatch.setattr(handler, 'stream', BrokenStream())
    client = app.test_client()
    sign_in(client)
    fields = begin(client)
    assert submit(client, fields).status_code == 302
    assert 'CANARY' not in capsys.readouterr().err


@pytest.mark.parametrize('decision,kind', [('allow', 'code_present'), ('deny', 'error_present')])
def test_prepared_redirect_classifies_presence_without_claiming_delivery(app, caplog, decision, kind):
    client = app.test_client()
    sign_in(client)
    fields = begin(client, state='CANARY-redirect-state')
    caplog.clear()
    response = submit(client, fields, decision=decision)
    recorded = events(caplog)
    assert response.status_code == 302
    assert all('response_kind' not in item for item in recorded[:-1])
    assert recorded[-1]['response_kind'] == kind
    assert recorded[-1]['callback_origin_matches'] is True
    assert recorded[-1]['callback_path_matches'] is True
    assert 'CANARY' not in caplog.text and response.location not in caplog.text


def test_allow_decision_can_produce_error_redirect_and_is_not_mislabeled(app, caplog, monkeypatch):
    client = app.test_client()
    sign_in(client)
    fields = begin(client)
    callback = app.config['CHATGPT_REDIRECT_URI']
    prepared = app.response_class('CANARY-private-body', status=302,
        headers={'Location': callback + '?error=CANARY-error&state=CANARY-state',
                 'X-Private': 'CANARY-private-header'})
    monkeypatch.setattr(app.extensions['chatgpt_oauth_server'],
                        'create_authorization_response', lambda *a, **k: prepared)
    caplog.clear()
    response = submit(client, fields)
    assert response.status_code == 302
    assert response.data == b'CANARY-private-body'
    assert response.headers['X-Private'] == 'CANARY-private-header'
    assert response.location == prepared.location
    assert events(caplog)[-1]['outcome'] == 'allow_response'
    assert events(caplog)[-1]['response_kind'] == 'error_present'
    assert database_state(app)[0] is not None
    assert 'CANARY' not in caplog.text


@pytest.mark.parametrize('suffix,kind', [
    ('?code=CANARY-code&state=CANARY-state&iss=CANARY-issuer', 'code_present'),
    ('?error=CANARY-error&error_description=CANARY-detail', 'error_present'),
    ('?code=CANARY-code&error=CANARY-error', 'code_and_error'),
    ('?state=CANARY-state', 'neither'),
    ('', 'neither'),
    ('?code=', 'code_present'),
    ('?%63ode=CANARY-code', 'code_present'),
    ('?code=CANARY-one&code=CANARY-two', 'unclassified'),
    ('?%ZZ=CANARY-invalid', 'unclassified'),
    ('?%FF=CANARY-invalid', 'unclassified'),
    ('?=CANARY-invalid', 'unclassified'),
    ('?code=CANARY-code#CANARY-fragment', 'unclassified'),
    ('?code=CANARY-code ' , 'unclassified'),
    ('?code=' + 'CANARY-oversized' * 1000, 'unclassified'),
    ('?' + '&'.join('x%d=CANARY-value' % i for i in range(33)), 'unclassified'),
])
def test_bounded_redirect_shape_parser_never_emits_values(app, caplog, suffix, kind):
    callback = app.config['CHATGPT_REDIRECT_URI']
    response = app.response_class('CANARY-body', status=302,
                                  headers={'Location': callback + suffix})
    before = (response.status_code, response.data, list(response.headers))
    with app.test_request_context(oauth.AUTHORIZE_PATH, method='POST', base_url=ISSUER):
        oauth._consent_start()
        oauth._consent_classify_response(response)
        oauth._consent_emit('http_response_prepared', status=response.status_code)
    assert events(caplog)[-1]['response_kind'] == kind
    assert 'CANARY' not in caplog.text
    assert callback not in caplog.text
    assert before == (response.status_code, response.data, list(response.headers))
    if kind == 'unclassified':
        assert 'callback_origin_matches' not in events(caplog)[-1]
        assert 'callback_path_matches' not in events(caplog)[-1]


@pytest.mark.parametrize('location,origin,path', [
    ('https://other.example.test/connector/callback?code=CANARY-code', False, True),
    ('https://chatgpt.example.test/different?error=CANARY-error', True, False),
    ('https://other.example.test/different?code=CANARY-code', False, False),
    ('https://chatgpt.example.test:443/connector/callback?code=CANARY-code', False, True),
    ('https://chatgpt.example.test/connector/%63allback?code=CANARY-code', True, False),
])
def test_origin_and_path_comparisons_are_exact_and_independent(app, caplog, location, origin, path):
    response = app.response_class(status=302, headers={'Location': location})
    with app.test_request_context(oauth.AUTHORIZE_PATH, method='POST', base_url=ISSUER):
        oauth._consent_start()
        oauth._consent_classify_response(response)
        oauth._consent_emit('http_response_prepared', status=302)
    last = events(caplog)[-1]
    assert last['callback_origin_matches'] is origin
    assert last['callback_path_matches'] is path
    assert 'CANARY' not in caplog.text and location not in caplog.text


@pytest.mark.parametrize('locations,kind', [
    ([], 'no_location'),
    ([''], 'unclassified'),
    (['/CANARY-relative?code=CANARY-code'], 'unclassified'),
    (['http://chatgpt.example.test/connector/callback?code=CANARY-code'], 'unclassified'),
    (['https://CANARY-user:CANARY-password@chatgpt.example.test/connector/callback'], 'unclassified'),
    (['https://chatgpt.example.test:bad/connector/callback'], 'unclassified'),
    (['https://chatgpt.example.test:99999/connector/callback'], 'unclassified'),
    (['https://[CANARY-broken/connector/callback'], 'unclassified'),
    (['https://chatgpt.example.test/one', 'https://chatgpt.example.test/two'], 'unclassified'),
])
def test_missing_malformed_and_duplicate_locations_are_safe(app, caplog, locations, kind):
    response = app.response_class('CANARY-body', status=302)
    for location in locations:
        response.headers.add('Location', location)
    before = list(response.headers)
    with app.test_request_context(oauth.AUTHORIZE_PATH, method='POST', base_url=ISSUER):
        oauth._consent_start()
        oauth._consent_classify_response(response)
        oauth._consent_emit('http_response_prepared', status=302)
    assert events(caplog)[-1]['response_kind'] == kind
    assert 'callback_origin_matches' not in events(caplog)[-1]
    assert 'CANARY' not in caplog.text
    assert list(response.headers) == before


@pytest.mark.parametrize('failure', ['parser', 'header_access', 'configuration'])
def test_classifier_failure_is_silent_and_response_untouched(app, caplog, monkeypatch, failure):
    def fail(*args, **kwargs):
        raise RuntimeError('CANARY-parser-private-exception')
    response = app.response_class('CANARY-body', status=302,
        headers={'Location': app.config['CHATGPT_REDIRECT_URI'] + '?code=CANARY-code'})
    before = (response.status_code, response.data, list(response.headers))
    if failure == 'parser':
        monkeypatch.setattr(oauth, 'urlsplit', fail)
    elif failure == 'header_access':
        monkeypatch.setattr(response.headers, 'getlist', fail)
    else:
        monkeypatch.setitem(app.config, 'CHATGPT_REDIRECT_URI', object())
    with app.test_request_context(oauth.AUTHORIZE_PATH, method='POST', base_url=ISSUER):
        oauth._consent_start()
        oauth._consent_classify_response(response)
        oauth._consent_emit('http_response_prepared', status=302)
    assert events(caplog)[-1]['response_kind'] == 'unclassified'
    assert 'CANARY' not in caplog.text
    assert before == (response.status_code, response.data, list(response.headers))


def test_new_fields_are_allowlisted_and_type_checked(app, caplog):
    with app.test_request_context(oauth.AUTHORIZE_PATH, method='POST', base_url=ISSUER):
        oauth._consent_start()
        g.chatgpt_consent_diagnostic.update({
            'response_kind': 'CANARY-unknown-kind',
            'callback_origin_matches': 'CANARY-not-bool',
            'callback_path_matches': 1,
        })
        oauth._consent_emit('http_response_prepared', status=302)
    last = events(caplog)[-1]
    assert not {'response_kind', 'callback_origin_matches', 'callback_path_matches'} & set(last)
    assert 'CANARY' not in caplog.text


@pytest.mark.parametrize('decision', ['allow', 'deny'])
def test_redirect_parser_failure_cannot_break_live_handler_semantics(app, caplog, monkeypatch, decision):
    def fail(*args, **kwargs):
        raise RuntimeError('CANARY-classifier-exception')
    client = app.test_client()
    sign_in(client)
    fields = begin(client)
    monkeypatch.setattr(oauth, 'unquote_plus', fail)
    caplog.clear()
    response = submit(client, fields, decision=decision)
    assert response.status_code == 302
    params = parse_qs(urlsplit(response.location).query)
    assert events(caplog)[-1]['response_kind'] == 'unclassified'
    if decision == 'allow':
        assert exchange(client, params['code'][0]).status_code == 200
    else:
        assert params['error'] == ['access_denied'] and 'code' not in params
    assert 'CANARY' not in caplog.text
    assert submit(client, fields).status_code == 403


@pytest.mark.parametrize('count,kind', [(32, 'neither'), (33, 'unclassified')])
def test_query_field_limit_is_explicit(app, caplog, count, kind):
    location = app.config['CHATGPT_REDIRECT_URI'] + '?' + '&'.join(
        'x%d=CANARY-value' % i for i in range(count))
    response = app.response_class(status=302, headers={'Location': location})
    with app.test_request_context(oauth.AUTHORIZE_PATH, method='POST', base_url=ISSUER):
        oauth._consent_start()
        oauth._consent_classify_response(response)
        oauth._consent_emit('http_response_prepared', status=302)
    assert events(caplog)[-1]['response_kind'] == kind
    assert 'CANARY' not in caplog.text


@pytest.mark.parametrize('suffix,kind', [
    ('?code=CANARY-CACHE-code&state=CANARY-CACHE-state', 'code_present'),
    ('?error=CANARY-CACHE-error&error_description=CANARY-CACHE-detail', 'error_present'),
    ('?%ZZ=CANARY-CACHE-malformed', 'unclassified'),
])
def test_oauth_query_values_never_enter_stdlib_parser_cache(app, caplog, monkeypatch, suffix, kind):
    original = oauth.urlsplit
    parsed_inputs = []
    def capture(value, *args, **kwargs):
        parsed_inputs.append(value)
        return original(value, *args, **kwargs)
    response = app.response_class(status=302,
        headers={'Location': app.config['CHATGPT_REDIRECT_URI'] + suffix})
    with app.test_request_context(oauth.AUTHORIZE_PATH, method='POST', base_url=ISSUER):
        monkeypatch.setattr(oauth, 'urlsplit', capture)
        oauth._consent_start()
        oauth._consent_classify_response(response)
        oauth._consent_emit('http_response_prepared', status=302)
    assert parsed_inputs
    assert all('?' not in value and 'CANARY' not in value for value in parsed_inputs)
    assert events(caplog)[-1]['response_kind'] == kind
    assert 'CANARY' not in caplog.text


@pytest.mark.parametrize('location', [
    'https://CANARY-user:CANARY-password@chatgpt.example.test/callback?code=CANARY-code',
    'https://chatgpt.example.test/callback?code=CANARY-code#CANARY-fragment',
])
def test_userinfo_and_fragment_rejected_before_cached_parser(app, caplog, monkeypatch, location):
    def forbidden(*args, **kwargs):
        pytest.fail('Unsupported credential-bearing or fragment URI reached URL parser')
    response = app.response_class(status=302, headers={'Location': location})
    with app.test_request_context(oauth.AUTHORIZE_PATH, method='POST', base_url=ISSUER):
        monkeypatch.setattr(oauth, 'urlsplit', forbidden)
        oauth._consent_start()
        oauth._consent_classify_response(response)
        oauth._consent_emit('http_response_prepared', status=302)
    assert events(caplog)[-1]['response_kind'] == 'unclassified'
    assert 'CANARY' not in caplog.text
