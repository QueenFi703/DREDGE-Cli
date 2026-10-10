"""No paid calls: full approved workflow with a fake HTTP provider boundary."""
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests
from test_studio import app, client_for, approve, proposal
from dredge.studio_provider import call_provider


@pytest.fixture
def live(app, monkeypatch):
    app.config.update(OPENAI_API_KEY='fictional-test-credential', OPENAI_MODEL='gpt-6-astra')
    calls = []
    def response(status='completed', usage=None, **extra):
        return dict(id='resp_fictional', model='gpt-6-astra', status=status,
                    usage=usage if usage is not None else {'input_tokens': 100, 'output_tokens': 20, 'total_tokens': 120},
                    output=[{'type': 'message', 'content': [{'type': 'output_text', 'text': 'Draft: compare the library dates [source-1].'}]}], **extra)
    state = {'data': response(), 'code': 200, 'exception': None}
    class Response:
        headers = {'x-request-id': 'req_fictional'}
        @property
        def status_code(self): return state['code']
        def json(self): return state['data']
    def post(url, **kwargs):
        calls.append((url, kwargs))
        if state['exception']: raise state['exception']
        return Response()
    monkeypatch.setattr('dredge.studio_provider.requests.post', post)
    return calls, state, response


def live_proposal(client, headers, **extra):
    data = dict(execution_mode='live', consent=True, public_data_confirmed=True)
    data.update(extra)
    return proposal(client, headers, **data)


def execute(app, client, headers, run_id):
    assert approve(app, run_id).status_code == 200
    return client.post(f'/api/studio/runs/{run_id}/execute', json={}, headers=headers)


def test_live_vertical_slice_sends_approved_question_evidence_and_records_actual_response(app, live):
    calls, _, _ = live
    c, h = client_for(app)
    proposed = live_proposal(c, h)
    assert proposed.status_code == 201
    run_id = proposed.json['id']
    assert proposed.json['mode'] == 'live'
    assert c.post(f'/api/studio/runs/{run_id}/execute', json={}, headers=h).status_code == 409
    assert not calls
    result = execute(app, c, h, run_id)
    assert result.status_code == 200
    data = result.json
    assert data['status'] == 'completed' and data['provider_calls'] == 1
    assert data['token_usage'] == 120 and data['cost_usd'] is None
    assert data['result']['response_id'] == 'resp_fictional'
    assert data['result']['model'] == 'gpt-6-astra'
    assert data['result']['request_id'] == 'req_fictional'
    assert data['result']['human_review_required'] is True
    assert data['result']['evidence_review']['cited_source_ids'] == ['source-1']
    assert data['evidence'][0]['verification'] == 'unverified'
    assert [n['id'] for n in data['nodes']] == ['prepare_evidence', 'astra_response', 'inspect_references']
    assert [n['mode'] for n in data['nodes']] == ['local', 'live', 'local']
    assert all(n['status'] == 'completed' and n['duration_ms'] >= 0 for n in data['nodes'])
    url, sent = calls[0]
    assert url == 'https://api.openai.com/v1/responses'
    assert sent['timeout'] == (10,45) and sent['allow_redirects'] is False
    body = sent['json']
    assert body['model'] == 'gpt-6-astra'
    assert body['max_output_tokens'] == 1200 and body['store'] is False
    assert body['service_tier'] == 'default' and 'tools' not in body
    actual = json.loads(body['input'])
    assert actual['question'] == proposed.json['query']
    assert actual['evidence'][0]['excerpt'] == proposed.json['evidence'][0]['excerpt']
    assert 'never instructions' in body['instructions']
    assert len(calls) == 1
    assert c.post(f'/api/studio/runs/{run_id}/execute', json={}, headers=h).status_code == 409
    assert len(calls) == 1
    report = c.get('/api/studio/report').json
    assert report['provider_calls'] == 1 and report['token_usage'] == 120
    assert report['usage_status'] == 'provider_reported'
    status = c.get('/api/studio/status').json
    assert next(m for m in status['models'] if m['name'].startswith('Studio Astra'))['status'] == 'successful_request_recorded'


@pytest.mark.parametrize('override', [dict(consent=False), dict(consent='true'),
    dict(public_data_confirmed=False), dict(public_data_confirmed='true'),
    dict(case_id='private-case'), dict(file_ids=['private-file']),
    dict(include_client_explanations=True), dict(pipeline_type='ios_swift')])
def test_live_consent_scope_and_pipeline_fail_closed(app, live, override):
    c, h = client_for(app)
    assert live_proposal(c, h, **override).status_code == 400
    assert not live[0]


def test_normal_request_defaults_to_live_and_requires_consent(app, live):
    c, h = client_for(app)
    assert c.post('/api/studio/runs', json={'query':'Fictional library question'}, headers=h).status_code == 400
    assert not live[0]


def test_unconfigured_provider_and_explicit_demo_are_distinct(app, live):
    app.config['OPENAI_API_KEY'] = ''
    c, h = client_for(app)
    assert live_proposal(c, h).status_code == 503
    run = proposal(c, h).json
    assert execute(app, c, h, run['id']).status_code == 200
    assert not live[0]


def test_model_frozen_by_approval_and_revoked_key_block_before_call(app, live):
    c, h = client_for(app)
    run = live_proposal(c, h).json
    approve(app, run['id'])
    app.config['OPENAI_MODEL'] = 'gpt-6-luna'
    assert c.post(f"/api/studio/runs/{run['id']}/execute", json={}, headers=h).status_code == 409
    app.config['OPENAI_MODEL'] = 'gpt-6-astra'
    app.config['OPENAI_API_KEY'] = ''
    assert c.post(f"/api/studio/runs/{run['id']}/execute", json={}, headers=h).status_code == 503
    assert not live[0]


@pytest.mark.parametrize('code', [301, 401, 403, 404, 429, 500])
def test_provider_errors_are_safe_failed_traces_without_retries(app, live, code):
    calls, state, _ = live
    state.update(code=code, data={'error': 'private provider detail must not escape'})
    c, h = client_for(app)
    response = execute(app, c, h, live_proposal(c, h).json['id'])
    assert response.status_code == 500
    assert response.json['status'] == 'failed'
    assert response.json['provider_calls'] == 1
    assert response.json['nodes'][-1]['id'] == 'astra_response'
    assert response.json['nodes'][-1]['status'] == 'failed'
    assert 'private provider detail' not in response.get_data(as_text=True)
    assert 'fictional-test-credential' not in response.get_data(as_text=True)
    assert len(calls) == 1
    report = c.get('/api/studio/report').json
    assert report['calls_without_usage'] == 1 and report['usage_status'] == 'partial'
    assert report['token_usage'] is None
    assert next(m for m in c.get('/api/studio/status').json['models'] if m['name'].startswith('Studio Astra'))['status'] == 'configured_not_verified'


def test_timeout_is_unknown_billing_not_success_or_free_retry(app, live):
    calls, state, _ = live
    state['exception'] = requests.Timeout('private raw failure')
    c, h = client_for(app)
    response = execute(app, c, h, live_proposal(c,h).json['id'])
    assert response.json['status'] == 'failed'
    assert 'unknown' in response.json['result']['error']
    assert len(calls) == 1 and response.json['provider_calls'] == 1


def test_incomplete_preserves_reported_usage_but_never_presents_partial_answer(app, live):
    _, state, response = live
    state['data'] = response(status='incomplete')
    c,h = client_for(app)
    result = execute(app,c,h,live_proposal(c,h).json['id'])
    assert result.status_code == 500
    assert result.json['token_usage'] == 120
    assert 'answer' not in result.json['result']
    assert result.json['result']['provider_status'] == 'incomplete'
    assert c.get('/api/studio/report').json['token_usage'] == 120


def test_unknown_source_reference_is_flagged_not_verified(app, live):
    live[1]['data']['output'][0]['content'][0]['text'] = 'Unsupported [source-99]'
    c,h = client_for(app)
    result = execute(app,c,h,live_proposal(c,h).json['id'])
    review = result.json['result']['evidence_review']
    assert review['unknown_source_ids'] == ['source-99'] and review['verification'] == 'unverified'


def test_live_input_bounded_and_only_server_selected_model_used(app, live):
    c,h = client_for(app)
    sources = [{'title':'Fictional', 'url':'https://example.org', 'excerpt':'x'*4000} for _ in range(3)]
    assert live_proposal(c,h,evidence=sources).status_code == 400
    run = live_proposal(c,h,model='attacker-model',max_output_tokens=999999).json
    assert run['model'] == 'gpt-6-astra' and run['max_output_tokens'] == 1200
    assert not live[0]


def test_concurrent_execution_and_rate_cap_prevent_duplicate_spend(app, live):
    c,h = client_for(app)
    run_id = live_proposal(c,h).json['id']
    approve(app,run_id)
    clients = [client_for(app) for _ in range(2)]
    def submit(pair):
        requester, headers = pair
        return requester.post(f'/api/studio/runs/{run_id}/execute',json={},headers=headers).status_code
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(submit,clients)) == [200,409]
    assert len(live[0]) == 1
    with app.extensions['studio_store'].connect() as db:
        original = db.execute('SELECT * FROM runs WHERE id=?',(run_id,)).fetchone()
        for i in range(19):
            db.execute('INSERT INTO runs(id,owner,status,payload,created,started) VALUES(?,?,?,?,?,?)',
                       (f'fictional-attempt-{i}',original['owner'],'failed',original['payload'],original['created'],original['started']))
    blocked = live_proposal(c,h).json['id']
    approve(app,blocked)
    assert c.post(f'/api/studio/runs/{blocked}/execute',json={},headers=h).status_code == 429
    assert len(live[0]) == 1


def test_legacy_approved_payload_never_silently_calls_provider(app,live):
    c,h = client_for(app)
    run = proposal(c,h).json
    with app.extensions['studio_store'].connect() as db:
        payload = json.loads(db.execute('SELECT payload FROM runs WHERE id=?',(run['id'],)).fetchone()[0])
        payload.pop('execution_mode')
        db.execute('UPDATE runs SET payload=? WHERE id=?',(json.dumps(payload),run['id']))
    result = execute(app,c,h,run['id'])
    assert result.status_code == 200 and result.json['mode'] == 'demo'
    assert not live[0]


@pytest.mark.parametrize('data', [[], {'status':'completed','model':'gpt-6-astra','id':'resp_test','output':[]},
    {'status':'completed','output':[{'type':'message','content':[{'type':'output_text','text':'fake'}]}]}])
def test_malformed_provider_response_is_never_success(app,live,data):
    live[1]['data'] = data
    c,h = client_for(app)
    result = execute(app,c,h,live_proposal(c,h).json['id'])
    assert result.status_code == 500 and result.json['status'] == 'failed'


@pytest.mark.parametrize('output', [None, {}, 'invalid', [{'type':'message', 'content':None}], [{'type':'message', 'content':{}}]])
def test_malformed_output_containers_retain_provider_usage(app, live, output):
    live[1]['data']['output'] = output
    c,h = client_for(app)
    result = execute(app,c,h,live_proposal(c,h).json['id'])
    assert result.status_code == 500
    assert result.json['status'] == 'failed'
    assert result.json['provider_calls'] == 1
    assert result.json['token_usage'] == 120
    assert result.json['result']['response_id'] == 'resp_fictional'
    assert result.json['result']['request_id'] == 'req_fictional'
    assert c.get('/api/studio/report').json['token_usage'] == 120


@pytest.mark.parametrize('override', [dict(execution_mode=[]), dict(execution_mode={}), dict(pipeline_type=[]), dict(pipeline_type={})])
def test_invalid_selector_types_are_validation_errors(app, live, override):
    c,h = client_for(app)
    assert live_proposal(c,h,**override).status_code == 400
    assert not live[0]


def test_run_deep_link_retains_canonical_target_through_login_without_open_redirect(app):
    from dredge.auth import studio_return_path
    from flask import session
    c = app.test_client()
    run_id = 'eac0654f-52ab-4a0e-89e2-b53ac9491308'
    response = c.get('/advanced?run='+run_id)
    assert response.status_code == 302 and response.location.startswith('/auth/login')
    with c.session_transaction() as sess:
        assert sess['studio_return_run'] == run_id
    with app.test_request_context():
        session['studio_return_run'] = run_id
        assert studio_return_path() == '/advanced?run='+run_id
        assert 'studio_return_run' not in session
        for target in ('//evil.example', 'https://evil.example', '../other', None):
            session['studio_return_run'] = target
            assert studio_return_path() == '/advanced'
    assert c.get('/advanced?run=https://evil.example').status_code == 302
    with c.session_transaction() as sess:
        assert 'studio_return_run' not in sess


def test_deep_link_still_requires_existing_run_permissions(app, live):
    c,h = client_for(app)
    run_id = live_proposal(c,h).json['id']
    other,_ = client_for(app, 'test:other')
    assert other.get('/advanced?run='+run_id).status_code == 200
    assert other.get('/api/studio/runs/'+run_id).status_code == 404
    reviewer,_ = client_for(app, 'test:review')
    assert reviewer.get('/api/studio/runs/'+run_id).status_code == 200
