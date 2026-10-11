"""Offline foundation tests; these do not claim an OAuth/MCP connection works."""
import json
import time

import pytest
from flask_login import login_user

from dredge.auth import User
from dredge.server import create_app
from dredge.chatgpt_foundation import call_read_tool, tool_definitions


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv('STUDIO_DB_PATH', str(tmp_path / 'studio.sqlite3'))
    monkeypatch.setenv('SECRET_KEY', 'test-only-key')
    monkeypatch.delenv('DREDGE_CHATGPT_ENABLED', raising=False)
    monkeypatch.delenv('STRIPE_RESTRICTED_KEY', raising=False)
    app = create_app()
    app.config['TESTING'] = True
    return app


def call(app, name, arguments=None, owner='github:owner', scopes=('studio:read',)):
    with app.test_request_context():
        if owner:
            login_user(User(owner, 'Fictional user', 'fictional@example.test', 'test'))
        return call_read_tool(name, arguments or {}, scopes=frozenset(scopes))


def seed(app, owner, run_id, result=None):
    with app.extensions['studio_store'].connect() as db:
        db.execute('INSERT INTO runs(id,owner,status,payload,result,created,started,ended) VALUES(?,?,?,?,?,?,?,?)',
                   (run_id, owner, 'completed', json.dumps({'execution_mode':'live','query':'private', 'pipeline_type':'default'}),
                    json.dumps(result or {'provider_calls':1}), time.time(), 10, 11))


def test_disabled_registers_no_mcp_or_oauth_routes(app):
    for path in ('/mcp', '/chatgpt/mcp', '/.well-known/oauth-protected-resource'):
        assert app.test_client().post(path).status_code == 404


def test_enabling_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv('STUDIO_DB_PATH', str(tmp_path / 'studio.sqlite3'))
    monkeypatch.setenv('DREDGE_CHATGPT_ENABLED', 'true')
    with pytest.raises(RuntimeError, match='OAuth configuration is incomplete'):
        create_app()


def test_only_two_readonly_scoped_no_argument_tools():
    definitions = tool_definitions()
    assert {t['name'] for t in definitions} == {'get_dredge_access', 'get_studio_activity_report'}
    for tool in definitions:
        assert tool['annotations'] == {'readOnlyHint':True, 'destructiveHint':False, 'idempotentHint':True, 'openWorldHint':False}
        assert tool['inputSchema'] == {'type':'object','properties':{},'additionalProperties':False}
        assert tool['securitySchemes'] == [{'type':'oauth2','scopes':['studio:read']}]
    definitions[0]['name'] = 'execute'
    assert tool_definitions()[0]['name'] == 'get_dredge_access'


def test_auth_scope_and_identity_injection_rejected(app):
    with pytest.raises(PermissionError):
        call(app, 'get_dredge_access', owner=None)
    with pytest.raises(PermissionError):
        call(app, 'get_dredge_access', scopes=('studio:write',))
    for arguments in ({'owner':'github:other'}, {'role':'admin'}, {'scope':'studio:read'}, []):
        with app.test_request_context():
            login_user(User('github:owner', 'Fictional', '', 'test'))
            with pytest.raises(ValueError):
                call_read_tool('get_studio_activity_report', arguments, scopes=frozenset({'studio:read'}))
    with pytest.raises(ValueError):
        call(app, 'execute_run')


def test_owner_only_report_and_missing_usage_truth(app):
    seed(app, 'github:owner', 'own')
    seed(app, 'github:other', 'other', {'provider_calls':1,'token_usage':999})
    app.config['STUDIO_ROLES'] = {'github:owner':'admin'}
    report = call(app, 'get_studio_activity_report')
    assert report['proposed_runs'] == report['completed_runs'] == report['provider_calls'] == 1
    assert report['token_usage'] is None
    assert report['cost_usd'] is None
    assert report['calls_without_usage'] == 1
    assert report['usage_status'] == 'partial'
    assert 'Missing usage is not zero usage' in report['cost_note']
    assert 'private' not in json.dumps(report)
    assert 'owner' not in report


def test_existing_report_and_tool_have_identical_results(app):
    seed(app, 'github:owner', 'own', {'provider_calls':1,'token_usage':14})
    from dredge.studio import usage_report
    with app.test_request_context():
        login_user(User('github:owner', 'Fictional', '', 'test'))
        assert usage_report().get_json() == call_read_tool('get_studio_activity_report', {}, scopes=frozenset({'studio:read'}))


def test_access_is_safe_server_state_without_new_report_gate(app):
    app.config.update(STRIPE_BILLING_MODE='live', BILLING_REQUIRE_SUBSCRIPTION=True)
    access = call(app, 'get_dredge_access')
    assert access == {'workspace_role':'viewer','studio_report_available':True,
                      'subscription_status':'inactive','subscription_requirement_enabled':True,
                      'billing_configured':False,'billing_environment':'live'}
    assert call(app, 'get_studio_activity_report')['proposed_runs'] == 0
    with app.extensions['studio_store'].connect() as db:
        db.execute('INSERT INTO billing_subscriptions VALUES(?,?,?,?,?,?)',
                   ('sub_private','github:owner','live','active',time.time()+1000,0))
    assert call(app, 'get_dredge_access')['subscription_status'] == 'active'
    assert call(app, 'get_dredge_access', owner='github:other')['subscription_status'] == 'inactive'
    app.config['STRIPE_BILLING_MODE'] = 'sandbox'
    assert call(app, 'get_dredge_access')['subscription_status'] == 'sandbox_only'
