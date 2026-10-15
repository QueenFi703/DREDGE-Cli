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


def test_application_registers_public_subscription_information(app):
    response = app.test_client().get('/subscription-info')
    assert response.status_code == 200
    assert 'Subscription access' in response.get_data(as_text=True)
    assert '/billing' not in response.get_data(as_text=True)


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
    for name in ('get_dredge_access', 'get_studio_activity_report'):
        for arguments in ({'owner':'github:other'}, {'role':'admin'}, {'scope':'studio:read'}, []):
            with app.test_request_context():
                login_user(User('github:owner', 'Fictional', '', 'test'))
                with pytest.raises(ValueError):
                    call_read_tool(name, arguments, scopes=frozenset({'studio:read'}))
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
    app.config.update(STRIPE_BILLING_MODE='live', BILLING_REQUIRE_SUBSCRIPTION=True,
                      BILLING_ORIGIN='https://dredge.example.test')
    access = call(app, 'get_dredge_access')
    assert access == {'workspace_role':'viewer','studio_report_available':True,
                      'subscription_status':'inactive','subscription_requirement_enabled':True,
                      'billing_configured':False,'billing_environment':'live',
                      'billing_environments':{'stripe':'live','paypal':'sandbox'},
                      'subscription_status_source':'stored_verified_records',
                      'subscription_status_checked_live':False,
                      'in_chat_purchases_enabled':False,
                      'paid_access':False,'paid_ai_included':False,
                      'subscription_info_url':'https://dredge.example.test/subscription-info',
                      'entitlement_notes':[
                          'Paid access requires a verified, active, unexpired live subscription for this account. Sandbox records do not grant paid access.',
                          'Workspace roles and permissions are assigned separately; a subscription does not grant or change them.',
                          'Existing owner-only stored reads remain available with their required authorization scopes, regardless of subscription status.',
                          'Paid AI usage is not included. These tools only read stored information and do not execute runs or call AI providers.',
                      ]}
    assert call(app, 'get_studio_activity_report')['proposed_runs'] == 0
    with app.extensions['studio_store'].connect() as db:
        db.execute('INSERT INTO billing_subscriptions VALUES(?,?,?,?,?,?)',
                   ('sub_private','github:owner','live','active',time.time()+1000,0))
    assert call(app, 'get_dredge_access')['subscription_status'] == 'active'
    assert call(app, 'get_dredge_access')['paid_access'] is True
    assert call(app, 'get_dredge_access', owner='github:other')['subscription_status'] == 'inactive'
    app.config['STRIPE_BILLING_MODE'] = 'sandbox'
    assert call(app, 'get_dredge_access')['subscription_status'] == 'sandbox_only'
    assert call(app, 'get_dredge_access')['paid_access'] is False
    with app.extensions['studio_store'].connect() as db:
        db.execute('DELETE FROM billing_subscriptions')
        db.execute('INSERT INTO billing_subscriptions VALUES(?,?,?,?,?,?)',
                   ('sub_sandbox','github:owner','sandbox','active',time.time()+1000,0))
    assert call(app, 'get_dredge_access')['subscription_status'] == 'sandbox_only'
    assert call(app, 'get_dredge_access')['paid_access'] is False


@pytest.mark.parametrize('workspace_role', ['viewer', 'operator', 'reviewer', 'admin'])
@pytest.mark.parametrize('paid_access', [False, True])
def test_entitlements_use_only_bound_owner_and_never_assign_roles(app, monkeypatch,
                                                               workspace_role, paid_access):
    seen = []

    def access_for_owner(owner):
        seen.append(owner)
        return {'subscription_status': 'active' if paid_access else 'inactive',
                'paid_access': paid_access, 'paid_ai_included': False}

    monkeypatch.setattr('dredge.chatgpt_foundation.subscription_access', access_for_owner,
                        raising=False)
    app.config['STUDIO_ROLES'] = {'github:owner': workspace_role}
    access = call(app, 'get_dredge_access')
    assert seen == ['github:owner']
    assert access['workspace_role'] == workspace_role
    assert access['paid_access'] is paid_access
    assert access['paid_ai_included'] is False
    assert app.config['STUDIO_ROLES'] == {'github:owner': workspace_role}
    assert 'owner' not in access


def test_access_and_tool_descriptors_only_link_to_information(app):
    app.config.update(DREDGE_CHATGPT_RUN_TOOLS_ENABLED=True,
                      BILLING_ORIGIN='https://dredge.example.test')
    definitions = tool_definitions(app)
    assert {tool['name'] for tool in definitions} == {
        'get_dredge_access', 'get_studio_activity_report', 'list_my_studio_runs',
        'get_my_studio_run_result', 'get_my_studio_execution_graph', 'find_my_saved_research'}
    assert all(item['annotations']['readOnlyHint'] for item in definitions)
    output = json.dumps([definitions, call(app, 'get_dredge_access')])
    assert 'https://dredge.example.test/subscription-info' in output
    for forbidden in ('/billing', '/checkout', '/portal', 'stripe.com', 'paypal.com',
                      'upgrade_url', 'purchase_url', 'checkout_url', 'outputTemplate'):
        assert forbidden not in output


def test_access_reports_mixed_environments_without_provider_calls_or_writes(app, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Reading plugin access must not contact a payment provider.')

    monkeypatch.setattr('dredge.billing.client', forbidden)
    monkeypatch.setattr('dredge.chatgpt_foundation.configured', lambda: False)
    monkeypatch.setattr('dredge.paypal_billing.paypal_client', forbidden)
    app.config.update(STRIPE_BILLING_MODE='sandbox', PAYPAL_BILLING_MODE='live')
    with app.extensions['studio_store'].connect() as db:
        before = list(db.iterdump())
    access = call(app, 'get_dredge_access')
    assert access['billing_environments'] == {'stripe': 'sandbox', 'paypal': 'live'}
    assert access['subscription_status'] == 'inactive'
    assert access['subscription_status_source'] == 'stored_verified_records'
    assert access['subscription_status_checked_live'] is False
    assert access['in_chat_purchases_enabled'] is False
    with app.extensions['studio_store'].connect() as db:
        assert list(db.iterdump()) == before


@pytest.mark.parametrize('workspace_role', ['viewer', 'operator', 'reviewer', 'admin'])
def test_inactive_accounts_keep_all_owner_bound_stored_reads(app, workspace_role):
    own = '00000000-0000-0000-0000-000000000001'
    other = '00000000-0000-0000-0000-000000000002'
    seed(app, 'github:owner', own)
    seed(app, 'github:other', other)
    app.config.update(STRIPE_BILLING_MODE='live', BILLING_REQUIRE_SUBSCRIPTION=True,
                      DREDGE_CHATGPT_RUN_TOOLS_ENABLED=True,
                      STUDIO_ROLES={'github:owner': workspace_role})
    scopes = ('studio:read', 'studio:runs:read')
    assert call(app, 'get_dredge_access')['paid_access'] is False
    assert call(app, 'get_studio_activity_report')['proposed_runs'] == 1
    runs = call(app, 'list_my_studio_runs', scopes=scopes)
    assert [run['id'] for run in runs['runs']] == [own]
    for name in ('get_my_studio_run_result', 'get_my_studio_execution_graph'):
        assert call(app, name, {'run_id': own}, scopes=scopes)
        with pytest.raises(ValueError, match='Run not found'):
            call(app, name, {'run_id': other}, scopes=scopes)
        with pytest.raises(PermissionError, match='studio:runs:read'):
            call(app, name, {'run_id': own})
