"""Contract tests for the public preview, real traces, and governance boundaries."""
import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from dredge.auth import User, _users, remember_studio_session
from dredge.server import create_app
from dredge.studio import StudioStore
from dredge.architecture import Node, NodeType, PipelineContext, DAGExecutionEngine, dredge_run_pipeline


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv('STUDIO_DB_PATH', str(tmp_path / 'studio.sqlite3'))
    monkeypatch.setenv('SECRET_KEY', 'testing-only-stable-signing-key')
    monkeypatch.setenv('STUDIO_ROLES_JSON', json.dumps({'test:op':'operator','test:review':'reviewer','test:admin':'admin','test:other':'operator'}))
    application = create_app()
    application.config['TESTING'] = True
    yield application
    _users.clear()


def client_for(app, user_id='test:op'):
    user = User(user_id, user_id, 'fictional@example.test', 'test')
    _users[user_id] = user
    client = app.test_client()
    with client.session_transaction() as sess:
        sess['_user_id'] = user.id
        sess['_fresh'] = True
    token = client.get('/api/studio/session').get_json()['csrf_token']
    return client, {'X-CSRF-Token':token}


def proposal(client, headers, **extra):
    payload = {'query':'Evaluate a fictional library service.', 'evidence':[{'title':'Public reference','url':'https://docs.python.org/3/library/sqlite3.html','excerpt':'User-supplied example, not fetched.'}]}
    payload.update(extra)
    return client.post('/api/studio/runs', json=payload, headers=headers)


def approve(app, run_id):
    reviewer, headers = client_for(app, 'test:review')
    return reviewer.post(f'/api/studio/runs/{run_id}/review', json={'decision':'approve','note':'The local proposal is suitable for execution.'}, headers=headers)


def test_public_preview_does_not_touch_account_or_storage(app):
    anonymous = app.test_client()
    for path in ['/auth/login','/architecture','/preview']:
        assert anonymous.get(path).status_code == 200
    preview = anonymous.get('/api/studio/preview').get_json()
    assert preview['mode'] == 'simulated'
    assert all(node['duration_ms'] is None for node in preview['nodes'])
    assert 'owner' not in preview
    with app.extensions['studio_store'].connect() as db:
        assert db.execute('SELECT COUNT(*) FROM runs').fetchone()[0] == 0


def test_unauthenticated_private_api_never_returns_login_html(app):
    client = app.test_client()
    for path in ['/api/studio/runs','/api/studio/status','/api/studio/report','/api/advanced/models/list','/api/dependabot/alerts']:
        response = client.get(path)
        assert response.status_code == 401
        assert response.is_json
    assert client.get('/advanced').status_code == 302


def test_default_viewer_and_no_client_role_escalation(app):
    client, headers = client_for(app, 'test:viewer')
    assert client.get('/api/studio/session').get_json()['role'] == 'viewer'
    assert proposal(client, headers, role='admin').status_code == 403
    assert client.get('/api/studio/audit').status_code == 403


def test_mutations_require_session_csrf(app):
    client, _ = client_for(app)
    assert proposal(client, {}).status_code == 403
    assert proposal(client, {'X-CSRF-Token':'wrong'}).status_code == 403


def test_approval_trace_evidence_and_measured_reporting(app):
    client, headers = client_for(app)
    created = proposal(client, headers)
    assert created.status_code == 201
    run_id = created.get_json()['id']
    assert created.get_json()['nodes'] == []
    assert client.post(f'/api/studio/runs/{run_id}/execute', json={}, headers=headers).status_code == 409
    assert approve(app, run_id).status_code == 200
    result = client.post(f'/api/studio/runs/{run_id}/execute', json={}, headers=headers)
    assert result.status_code == 200
    run = result.get_json()
    assert run['status'] == 'completed'
    assert [node['id'] for node in run['nodes']] == ['cli_entry','dag_engine','mode_graph','translate','normalize','redis_cache','telemetry']
    assert len(run['events']) == 14
    assert all(node['status']=='completed' and node['duration_ms']>=0 for node in run['nodes'])
    assert run['nodes'][1]['dependencies'] == ['cli_entry']
    assert run['nodes'][3]['mode'] == 'simulated'
    assert run['evidence'][0]['provenance'] == 'user_supplied'
    assert run['evidence'][0]['verification'] == 'unverified'
    assert client.post(f'/api/studio/runs/{run_id}/execute', json={}, headers=headers).status_code == 409
    report = client.get('/api/studio/report').get_json()
    assert report['completed_runs'] == 1 and report['success_rate'] == 1
    assert report['average_duration_ms'] >= 0
    assert report['cost_usd'] is None and report['token_usage'] is None
    reviewer, _ = client_for(app, 'test:review')
    audit = reviewer.get('/api/studio/audit').get_json()
    assert audit['chain_verified'] is True
    assert [event['action'] for event in reversed(audit['events'])] == ['run_proposed','run_approved','run_started','run_completed']


def test_separate_reviewer_no_self_approval_and_no_repeat_decision(app):
    admin, headers = client_for(app, 'test:admin')
    run_id = proposal(admin, headers).get_json()['id']
    assert admin.post(f'/api/studio/runs/{run_id}/review',json={'decision':'approve','note':'Self approval'},headers=headers).status_code == 403
    assert approve(app, run_id).status_code == 200
    assert approve(app, run_id).status_code == 409


def test_owner_isolation_and_review_role(app):
    client, headers = client_for(app)
    run_id = proposal(client, headers).get_json()['id']
    other, other_headers = client_for(app, 'test:other')
    assert other.get('/api/studio/runs').get_json()['runs'] == []
    assert other.get('/api/studio/runs/'+run_id).status_code == 404
    assert other.post(f'/api/studio/runs/{run_id}/execute',json={},headers=other_headers).status_code == 404
    assert other.post(f'/api/studio/runs/{run_id}/review',json={'decision':'approve','note':'No'},headers=other_headers).status_code == 403


def test_rejection_cannot_execute(app):
    client, headers = client_for(app)
    run_id = proposal(client, headers).get_json()['id']
    reviewer, review_headers = client_for(app,'test:review')
    assert reviewer.post(f'/api/studio/runs/{run_id}/review',json={'decision':'reject','note':'Revise the objective.'},headers=review_headers).status_code == 200
    assert client.post(f'/api/studio/runs/{run_id}/execute',json={},headers=headers).status_code == 409


@pytest.mark.parametrize('url',['javascript:alert(1)','http://example.com','https://user:password@example.com','https://127.0.0.1','https://169.254.169.254','https://host.internal','https://localhost',None])
def test_unsafe_source_urls_rejected(app,url):
    client, headers = client_for(app)
    assert proposal(client,headers,evidence=[{'title':'Unsafe','url':url,'excerpt':''}]).status_code == 400


@pytest.mark.parametrize('payload',[None,[],{'query':''},{'query':'ok','pipeline_type':'unexpected'},{'query':'ok','evidence':'invalid'}])
def test_invalid_payloads(app,payload):
    client, headers = client_for(app)
    assert client.post('/api/studio/runs',json=payload,headers=headers).status_code == 400


def test_session_expiry_does_not_erase_saved_runs(app):
    client, headers = client_for(app)
    run_id = proposal(client,headers).get_json()['id']
    with client.session_transaction() as sess:
        sess['studio_last_activity'] = time.time()-app.config['STUDIO_IDLE_SECONDS']-1
    response = client.get('/api/studio/runs')
    assert response.status_code == 401
    assert response.get_json()['code']=='session_expired'
    with app.extensions['studio_store'].connect() as db:
        assert db.execute('SELECT id FROM runs WHERE id=?',(run_id,)).fetchone()


def test_signed_profile_survives_process_user_cache_loss(app):
    client, _ = client_for(app)
    user = _users['test:op']
    with app.test_request_context():
        remember_studio_session(user)
        from flask import session
        profile = dict(session['studio_profile'])
    with client.session_transaction() as sess:
        sess['studio_profile'] = profile
    _users.clear()
    assert client.get('/api/studio/session').get_json()['role']=='operator'


def test_store_reopen_and_append_only_audit(app):
    client, headers = client_for(app)
    run_id = proposal(client,headers).get_json()['id']
    reopened = StudioStore(app.extensions['studio_store'].path)
    with reopened.connect() as db:
        assert db.execute('SELECT id FROM runs WHERE id=?',(run_id,)).fetchone()
        with pytest.raises(Exception,match='append-only'):
            db.execute('DELETE FROM audit')


def test_legacy_pipeline_cannot_bypass_review(app):
    client, headers = client_for(app)
    assert client.post('/api/architecture/pipeline/execute',json={'input_data':{}},headers=headers).status_code==409
    assert client.get('/api/advanced/models/list').get_json()['operation_mode']=='simulated'


def test_empty_reports_and_provider_state_are_honest(app):
    client, _ = client_for(app)
    report=client.get('/api/studio/report').get_json()
    assert report['success_rate'] is None and report['average_duration_ms'] is None
    status=client.get('/api/studio/status').get_json()
    assert next(m for m in status['models'] if m['name'].startswith('Astra · '))['status']=='not_connected'
    assert next(t for t in status['tools'] if t['name']=='Local DAG engine')['status']=='not_yet_observed'


def test_login_errors_are_accessible_and_do_not_reflect_arbitrary_input(app):
    client=app.test_client()
    page=client.get('/auth/login?reason=session_expired').get_data(as_text=True)
    assert 'role="alert"' in page and 'Your session expired' in page
    page=client.get('/auth/login?error=%3Cscript%3Eevil%3C/script%3E').get_data(as_text=True)
    assert '<script>evil' not in page
    assert 'GitHub unavailable' in page


def test_ios_graph_and_failure_trace_callback():
    ios=asyncio.run(dredge_run_pipeline({'query':'example'},pipeline_type='ios_swift'))
    assert [node['id'] for node in ios['trace_nodes']]==['ingest','async_translation','redis_cache']
    engine=DAGExecutionEngine()
    def fail(context):
        raise RuntimeError('test failure')
    engine.add_node(Node('failure',NodeType.EXECUTE,fail))
    events=[]
    with pytest.raises(RuntimeError):
        asyncio.run(engine.execute(PipelineContext('failure-run',{},trace_callback=events.append)))
    assert [event['status'] for event in events]==['running','failed']
    assert events[-1]['duration_ms']>=0


def test_execution_failure_is_stored_and_reported(app,monkeypatch):
    client, headers=client_for(app)
    run_id=proposal(client,headers).get_json()['id']
    approve(app,run_id)
    async def fail(*args,**kwargs):
        raise RuntimeError('secret internal exception')
    monkeypatch.setattr('dredge.studio.dredge_run_pipeline',fail)
    response=client.post(f'/api/studio/runs/{run_id}/execute',json={},headers=headers)
    assert response.status_code==500 and response.get_json()['status']=='failed'
    assert 'secret internal' not in response.get_data(as_text=True)
    assert client.get('/api/studio/report').get_json()['failed_runs']==1


def test_gateway_mount_routes_to_studio_and_preserves_health(tmp_path,monkeypatch):
    monkeypatch.setenv('STUDIO_DB_PATH',str(tmp_path/'gateway.sqlite3'))
    from fastapi.testclient import TestClient
    from dredge.orion_gateway import app as gateway
    with TestClient(gateway) as client:
        assert client.get('/health').json()['service']=='dredge-orion-gateway'
        assert client.get('/auth/login').status_code==200
        assert client.get('/architecture').status_code==200
        assert client.get('/preview').status_code==200
        assert client.get('/static/studio.js').status_code==200
        assert client.get('/api/studio/runs').status_code==401


def test_concurrent_execution_claims_exactly_once(app):
    client, headers = client_for(app)
    run_id = proposal(client,headers,pipeline_type='ios_swift').get_json()['id']
    assert approve(app,run_id).status_code==200
    clients = [client_for(app) for _ in range(2)]
    def execute(pair):
        requester, request_headers = pair
        return requester.post(f'/api/studio/runs/{run_id}/execute',json={},headers=request_headers).status_code
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(execute,clients))==[200,409]
    with app.extensions['studio_store'].connect() as db:
        assert db.execute('SELECT COUNT(*) FROM trace_events WHERE run_id=?',(run_id,)).fetchone()[0]==6
        assert db.execute("SELECT COUNT(*) FROM audit WHERE run_id=? AND action='run_started'",(run_id,)).fetchone()[0]==1
