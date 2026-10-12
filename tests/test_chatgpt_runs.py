"""Offline owner-only run projections; no provider calls or live fixture data."""
import json
import math
from uuid import UUID

import pytest
from flask_login import login_user

from dredge.auth import User
from dredge.chatgpt_foundation import call_read_tool, tool_definitions
from dredge.server import create_app

BASE_SCOPE = 'studio:read'
RUNS_SCOPE = 'studio:runs:read'
OWNER = 'github:fictional-owner'
OTHER = 'github:fictional-other'
OLD_TOOLS = {'get_dredge_access', 'get_studio_activity_report'}
RUN_TOOLS = {'list_my_studio_runs', 'get_my_studio_run_result', 'get_my_studio_execution_graph'}
META_KEYS = {'id', 'status', 'created', 'started', 'ended', 'mode'}


def run_id(number):
    return str(UUID(int=number))


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv('STUDIO_DB_PATH', str(tmp_path / 'studio.sqlite3'))
    monkeypatch.setenv('SECRET_KEY', 'fictional-test-key')
    monkeypatch.delenv('DREDGE_CHATGPT_ENABLED', raising=False)
    monkeypatch.delenv('DREDGE_CHATGPT_RUN_TOOLS_ENABLED', raising=False)
    monkeypatch.delenv('STRIPE_RESTRICTED_KEY', raising=False)
    app = create_app()
    app.config.update(TESTING=True, DREDGE_CHATGPT_RUN_TOOLS_ENABLED=True)
    return app


def call(app, name='list_my_studio_runs', arguments=None, *, owner=OWNER,
         scopes=frozenset({BASE_SCOPE, RUNS_SCOPE})):
    with app.test_request_context():
        if owner:
            login_user(User(owner, 'Fictional user', 'fictional@example.test', 'test'))
        return call_read_tool(name, {} if arguments is None else arguments, scopes=scopes)


def seed(app, number=1, *, owner=OWNER, status='completed', created=100,
         payload=None, result=None, raw_payload=None, raw_result=None):
    if payload is None:
        payload = {'execution_mode': 'live', 'data_scope': 'public_or_fictional',
                   'query': 'PRIVATE PROMPT', 'evidence': [{'text': 'PRIVATE EVIDENCE'}],
                   'model': 'PRIVATE MODEL', 'owner': 'PRIVATE OWNER'}
    if result is None:
        result = {'answer': 'A fictional public answer.', 'token_usage': 42,
                  'cost_usd': None, 'events': ['PRIVATE EVENTS'],
                  'error': 'PRIVATE ERROR', 'nodes': ['PRIVATE NODES'],
                  'reviewer': 'PRIVATE REVIEWER', 'name': 'PRIVATE NAME'}
    with app.extensions['studio_store'].connect() as db:
        db.execute('''INSERT INTO runs
            (id,owner,status,payload,result,created,started,ended,reviewer,review_note)
            VALUES(?,?,?,?,?,?,?,?,?,?)''',
            (run_id(number), owner, status,
             raw_payload if raw_payload is not None else json.dumps(payload),
             raw_result if raw_result is not None else json.dumps(result),
             created, 101, 102, 'PRIVATE REVIEWER', 'PRIVATE NOTE'))
    return run_id(number)


def test_enabled_descriptors_and_scopes_are_explicit_and_copied(app):
    from dredge.chatgpt_foundation import RUNS_SCOPE as exported, supported_scopes
    assert exported == RUNS_SCOPE
    assert supported_scopes(app) == frozenset({BASE_SCOPE, RUNS_SCOPE})
    definitions = tool_definitions(app)
    assert {item['name'] for item in definitions} == OLD_TOOLS | RUN_TOOLS
    for item in definitions:
        assert item['annotations'] == {'readOnlyHint': True, 'destructiveHint': False,
                                       'idempotentHint': True, 'openWorldHint': False}
        assert item['inputSchema']['additionalProperties'] is False
        expected = [BASE_SCOPE, RUNS_SCOPE] if item['name'] in RUN_TOOLS else [BASE_SCOPE]
        assert item['securitySchemes'] == [{'type': 'oauth2', 'scopes': expected}]
    indexed = {item['name']: item for item in definitions}
    assert indexed['list_my_studio_runs']['inputSchema']['properties']['limit'] == {
        'type': 'integer', 'minimum': 1, 'maximum': 20, 'default': 10}
    assert indexed['get_my_studio_run_result']['inputSchema']['required'] == ['run_id']
    definitions[-1]['inputSchema']['properties']['run_id']['type'] = 'array'
    assert tool_definitions(app)[-1]['inputSchema']['properties']['run_id']['type'] == 'string'
    with app.app_context():
        assert {item['name'] for item in tool_definitions()} == OLD_TOOLS | RUN_TOOLS
    assert {item['name'] for item in tool_definitions()} == OLD_TOOLS


@pytest.mark.parametrize('enabled', [None, False, 0, 1, 'true', 'false', [], {}])
def test_run_tools_fail_closed_unless_config_is_exact_true(app, enabled):
    from dredge.chatgpt_foundation import supported_scopes
    app.config['DREDGE_CHATGPT_RUN_TOOLS_ENABLED'] = enabled
    assert supported_scopes(app) == frozenset({BASE_SCOPE})
    assert {item['name'] for item in tool_definitions(app)} == OLD_TOOLS
    for name, arguments in [('list_my_studio_runs', {}),
                            ('get_my_studio_run_result', {'run_id': run_id(1)}),
                            ('get_my_studio_execution_graph', {'run_id': run_id(1)})]:
        with pytest.raises(PermissionError):
            call(app, name, arguments)
    assert call(app, 'get_dredge_access', scopes=frozenset({BASE_SCOPE}))


def test_run_tools_default_disabled_without_configuration(app):
    app.config.pop('DREDGE_CHATGPT_RUN_TOOLS_ENABLED')
    with pytest.raises(PermissionError):
        call(app)
    assert {item['name'] for item in tool_definitions(app)} == OLD_TOOLS


@pytest.mark.parametrize('scopes', [frozenset(), frozenset({BASE_SCOPE}),
                                   frozenset({RUNS_SCOPE}), {BASE_SCOPE, RUNS_SCOPE},
                                   [BASE_SCOPE, RUNS_SCOPE], 'studio:read studio:runs:read'])
def test_new_tools_require_verified_frozenset_and_both_scopes(app, scopes):
    for name, arguments in [('list_my_studio_runs', {}),
                            ('get_my_studio_run_result', {'run_id': run_id(1)}),
                            ('get_my_studio_execution_graph', {'run_id': run_id(1)})]:
        with pytest.raises(PermissionError):
            call(app, name, arguments, scopes=scopes)


def test_authentication_and_old_argument_contract_unchanged(app):
    with pytest.raises(PermissionError):
        call(app, owner=None)
    for name in OLD_TOOLS:
        assert call(app, name, scopes=frozenset({BASE_SCOPE}))
        for arguments in ({'limit': 10}, {'run_id': run_id(1)}, [], None):
            with app.test_request_context():
                login_user(User(OWNER, '', '', 'test'))
                with pytest.raises(ValueError, match='accept no arguments'):
                    call_read_tool(name, arguments, scopes=frozenset({BASE_SCOPE}))
    with pytest.raises(ValueError):
        call(app, 'execute_run')


@pytest.mark.parametrize('role', ['viewer', 'operator', 'reviewer', 'admin'])
def test_owner_constraints_apply_to_every_role_and_forged_public_label(app, role):
    own_id = seed(app, 1)
    foreign_id = seed(app, 2, owner=OTHER,
                      result={'answer': 'FOREIGN ANSWER', 'token_usage': 999999})
    app.config['STUDIO_ROLES'] = {OWNER: role}
    result = call(app)
    assert [row['id'] for row in result['runs']] == [own_id]
    assert set(result['runs'][0]) == META_KEYS
    detail = call(app, 'get_my_studio_run_result', {'run_id': own_id})
    assert detail['summary'] == 'A fictional public answer.'
    assert detail['token_usage'] == 42
    for target in [foreign_id, run_id(999)]:
        with pytest.raises(ValueError, match='Run not found'):
            call(app, 'get_my_studio_run_result', {'run_id': target})
    serialized = json.dumps([result, detail])
    assert 'PRIVATE' not in serialized and 'FOREIGN' not in serialized
    for key in ['owner', 'query', 'evidence', 'reviewer', 'review_note', 'events', 'nodes', 'error']:
        assert key not in detail


def test_list_bounded_pagination_and_deterministic_tie_order(app):
    for number in range(25, 0, -1):
        seed(app, number, created=100 + (number > 20))
    first = call(app)
    assert [row['id'] for row in first['runs']] == [run_id(n) for n in list(range(21, 26)) + list(range(1, 6))]
    assert first['limit'] == 10 and first['offset'] == 0
    assert first['next_offset'] == 10 and first['has_more'] is True
    second = call(app, arguments={'limit': 20, 'offset': 10})
    assert [row['id'] for row in second['runs']] == [run_id(n) for n in range(6, 21)]
    assert second['next_offset'] is None and second['has_more'] is False
    assert call(app, arguments={'limit': 1, 'offset': 1000})['runs'] == []


def test_status_and_timestamp_filters_are_owner_constrained(app):
    for number, status, created in [(1, 'pending_approval', 100), (2, 'running', 200),
                                    (3, 'completed', 300), (4, 'completed', 400)]:
        seed(app, number, status=status, created=created)
    seed(app, 5, owner=OTHER, created=350)
    result = call(app, arguments={'status': 'completed', 'created_after': 200,
                                  'created_before': 400.0})
    assert [row['id'] for row in result['runs']] == [run_id(3)]
    assert call(app, arguments={'status': 'failed'})['runs'] == []


@pytest.mark.parametrize('arguments', [
    [], 'list', 1, {'limit': True}, {'limit': 0}, {'limit': 21}, {'limit': 1.0},
    {'offset': False}, {'offset': -1}, {'offset': 1001}, {'offset': '0'},
    {'status': 'COMPLETED'}, {'status': "completed' OR 1=1 --"}, {'status': ['completed']},
    {'created_after': float('nan')}, {'created_after': float('inf')},
    {'created_before': float('-inf')}, {'created_after': True},
    {'created_after': -1}, {'created_before': 253402300800},
    {'created_after': '2026-01-01'}, {'created_after': None},
    {'created_after': 20, 'created_before': 10}, {'created_after': 20, 'created_before': 20},
    {'owner': OTHER}, {'scope': RUNS_SCOPE}, {'role': 'admin'}, {'cursor': 'anything'},
])
def test_list_rejects_invalid_or_unsupported_arguments(app, arguments):
    with pytest.raises(ValueError):
        call(app, arguments=arguments)


@pytest.mark.parametrize('arguments', [
    {}, [], {'run_id': None}, {'run_id': 1}, {'run_id': ''},
    {'run_id': 'not-a-uuid'}, {'run_id': '../private'}, {'run_id': 'a' * 32},
    {'run_id': '{' + run_id(1) + '}'}, {'run_id': 'urn:uuid:' + run_id(1)},
    {'run_id': 'FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF'},
    {'run_id': run_id(1) + ' '}, {'run_id': run_id(1), 'owner': OTHER},
    {'run_id': run_id(1), 'include_events': True},
])
def test_detail_requires_one_strict_canonical_uuid(app, arguments):
    with pytest.raises(ValueError):
        call(app, 'get_my_studio_run_result', arguments)


@pytest.mark.parametrize('payload', [{}, {'data_scope': 'private'}, {'data_scope': None},
                                     {'data_scope': True}, {'data_scope': 'PUBLIC_OR_FICTIONAL'}])
def test_legacy_and_unknown_data_scopes_never_return_content(app, payload):
    ident = seed(app, payload=payload)
    detail = call(app, 'get_my_studio_run_result', {'run_id': ident})
    assert detail['summary'] is None
    assert detail['summary_status'] == 'withheld_data_scope'
    assert detail['summary_truncated'] is False
    assert 'fictional public answer' not in json.dumps(detail)


@pytest.mark.parametrize('raw', ['{broken', '[]', 'null', '"text"', '42',
                                 '{"data_scope":"private","data_scope":"public_or_fictional"}',
                                 '{"data_scope":"public_or_fictional","number":NaN}',
                                 '{"data_scope":"public_or_fictional","number":1e999}',
                                 '[' * 1500 + '0' + ']' * 1500])
def test_invalid_or_ambiguous_payload_json_is_fail_closed(app, raw):
    ident = seed(app, raw_payload=raw)
    assert call(app)['runs'][0]['mode'] == 'unknown'
    detail = call(app, 'get_my_studio_run_result', {'run_id': ident})
    assert detail['summary'] is None
    assert detail['summary_status'] == 'withheld_data_scope'


@pytest.mark.parametrize('raw', ['{broken', '[]', 'null', '"PRIVATE RESULT"',
                                 '{"answer":"PRIVATE ONE","answer":"PRIVATE TWO"}',
                                 '{"answer":"PRIVATE","token_usage":NaN}',
                                 '{"answer":"PRIVATE","cost_usd":1e999}',
                                 '[' * 1500 + '0' + ']' * 1500])
def test_invalid_result_json_never_falls_back_to_raw_content(app, raw):
    ident = seed(app, raw_result=raw)
    detail = call(app, 'get_my_studio_run_result', {'run_id': ident})
    assert detail['summary'] is None
    assert detail['summary_status'] == 'unavailable'
    assert detail['token_usage'] is detail['cost_usd'] is None
    assert detail['usage_status'] == detail['cost_status'] == 'unknown'
    assert 'PRIVATE' not in json.dumps(detail)


@pytest.mark.parametrize('result', [{}, {'answer': None}, {'answer': {'text': 'PRIVATE'}},
                                    {'answer': ['PRIVATE']}, {'answer': 5}, {'answer': ''},
                                    {'answer': '  '}, {'summary': 'PRIVATE'},
                                    {'answer': '\ud800PRIVATE'}])
def test_summary_requires_explicit_nonempty_portable_answer_string(app, result):
    ident = seed(app, result=result)
    detail = call(app, 'get_my_studio_run_result', {'run_id': ident})
    assert detail['summary'] is None
    assert 'PRIVATE' not in json.dumps(detail)


def test_summary_is_bounded_and_label_is_not_a_pii_guarantee(app):
    ident = seed(app, result={'answer': 'A' * 3000})
    detail = call(app, 'get_my_studio_run_result', {'run_id': ident})
    assert detail['summary'] == 'A' * 2000
    assert detail['summary_status'] == 'available' and detail['summary_truncated'] is True
    assert 'not instructions' in detail['content_note']
    assert 'not proof' in detail['content_note']
    assert 'personal information' in detail['content_note']


@pytest.mark.parametrize('result', [{}, {'token_usage': None, 'cost_usd': None},
    {'token_usage': True, 'cost_usd': False}, {'token_usage': -1, 'cost_usd': -1},
    {'token_usage': '42', 'cost_usd': '1.5'}, {'token_usage': 1.5, 'cost_usd': 'PRIVATE'},
    {'token_usage': 10**30, 'cost_usd': 10**30},
    {'token_usage': {'name': 'PRIVATE'}, 'cost_usd': ['PRIVATE']}])
def test_missing_or_malformed_usage_and_cost_remain_unknown(app, result):
    ident = seed(app, result=result)
    detail = call(app, 'get_my_studio_run_result', {'run_id': ident})
    assert detail['token_usage'] is detail['cost_usd'] is None
    assert detail['usage_status'] == detail['cost_status'] == 'unknown'
    assert 'PRIVATE' not in json.dumps(detail)


@pytest.mark.parametrize('tokens,cost', [(0, 0), (42, 0.001), (9007199254740991, 1)])
def test_explicit_finite_recorded_usage_and_cost_are_preserved(app, tokens, cost):
    ident = seed(app, result={'token_usage': tokens, 'cost_usd': cost})
    detail = call(app, 'get_my_studio_run_result', {'run_id': ident})
    assert detail['token_usage'] == tokens and detail['cost_usd'] == cost
    assert detail['usage_status'] == detail['cost_status'] == 'recorded'


def test_malformed_metadata_is_not_echoed_as_a_sensitive_free_text_channel(app):
    ident = seed(app, status='PRIVATE STATUS', payload={'execution_mode': 'PRIVATE MODE'})
    with app.extensions['studio_store'].connect() as db:
        db.execute('UPDATE runs SET started=?, ended=? WHERE id=?', ('PRIVATE TIME', math.inf, ident))
    row = call(app)['runs'][0]
    assert row['status'] == row['mode'] == 'unknown'
    assert row['started'] is row['ended'] is None
    assert 'PRIVATE' not in json.dumps(row)


@pytest.mark.parametrize('field', ['payload', 'result'])
def test_oversized_stored_json_has_no_content_fallback(app, field):
    ident = seed(app)
    oversized = json.dumps({'data_scope': 'public_or_fictional', 'answer': 'PRIVATE',
                            'padding': 'X' * 1_000_000})
    with app.extensions['studio_store'].connect() as db:
        db.execute(f'UPDATE runs SET {field}=? WHERE id=?', (oversized, ident))
    detail = call(app, 'get_my_studio_run_result', {'run_id': ident})
    assert detail['summary'] is None
    assert 'PRIVATE' not in json.dumps(detail)


def test_reads_neither_mutate_runs_nor_invoke_provider_or_network(app, monkeypatch):
    import requests
    from dredge import studio_provider

    def forbidden(*args, **kwargs):
        raise AssertionError('Run reads must not invoke provider or network.')

    monkeypatch.setattr(requests.sessions.Session, 'request', forbidden)
    monkeypatch.setattr(studio_provider, 'call_provider', forbidden)
    ident = seed(app)
    with app.extensions['studio_store'].connect() as db:
        before = list(db.iterdump())
    call(app)
    call(app, 'get_my_studio_run_result', {'run_id': ident})
    with app.extensions['studio_store'].connect() as db:
        assert list(db.iterdump()) == before


@pytest.mark.parametrize('field', ['payload', 'result'])
def test_embedded_nul_json_never_becomes_an_accepted_sql_substr_prefix(app, field):
    ident = seed(app)
    malformed = (json.dumps({'data_scope': 'public_or_fictional', 'execution_mode': 'live'}) + '\x00'
                 + json.dumps({'data_scope': 'private'}) if field == 'payload' else
                 json.dumps({'answer': 'PRIVATE PREFIX'}) + '\x00' + '{broken')
    with app.extensions['studio_store'].connect() as db:
        db.execute(f'UPDATE runs SET {field}=? WHERE id=?', (malformed, ident))
    detail = call(app, 'get_my_studio_run_result', {'run_id': ident})
    assert detail['summary'] is None
    assert detail['summary_status'] == ('withheld_data_scope' if field == 'payload' else 'unavailable')
    if field == 'payload':
        assert call(app)['runs'][0]['mode'] == 'unknown'
