import json
import pytest
import requests
from test_studio import app, client_for
from test_casework import enable
from dredge.research_provider import provider, SMOKE_PROFILE, SMOKE_QUESTION
from dredge.studio_provider import ProviderError


def response(status='completed'):
    return dict(id='resp_real', model='gpt-6-astra-actual', status=status,
        usage={'input_tokens':40, 'output_tokens':12, 'total_tokens':52}, output=[
            {'type':'reasoning','summary':'NEVER SAVE REASONING'},
            {'type':'web_search_call','id':'ws_real','status':'completed','action':{
                'type':'search','query':'NEVER SAVE QUERY','sources':[
                    {'url':'https://dss.mo.gov/child-support/'}, {'url':'javascript:alert(1)'},
                    {'url':'https://secret:password@example.org/'}, {'url':'https://127.0.0.1/'}]}},
            {'type':'message','content':[{'type':'output_text','text':'Official page.', 'annotations':[
                {'type':'url_citation','url':'https://dss.mo.gov/child-support/','title':'Missouri','start_index':0,'end_index':8}]}]}])


def mock_response(monkeypatch, data, status=200):
    seen=[]
    class R:
        status_code=status
        headers={'x-request-id':'req_real'}
        def json(self): return data
    monkeypatch.setattr('dredge.research_provider.requests.post',lambda *a,**kw:seen.append(kw) or R())
    return seen


def test_execution_is_independent_from_citations_and_sanitized(app, monkeypatch):
    enable(app);seen=mock_response(monkeypatch,response())
    with app.app_context(): r=provider({'model':'gpt-6-astra'})
    assert r['model']=='gpt-6-astra-actual' and r['response_id']=='resp_real'
    assert r['request_id']=='req_real' and r['usage']['total_tokens']==52
    assert r['web_search_verified'] is True
    assert r['search_calls']==[{'id':'ws_real','status':'completed','action':{'type':'search','sources':['https://dss.mo.gov/child-support/']}}]
    assert 'NEVER SAVE' not in json.dumps(r) and 'password' not in json.dumps(r)
    assert r['cost_usd'] is None and seen[0]['allow_redirects'] is False
    data=response();data['output']=data['output'][2:];mock_response(monkeypatch,data)
    with app.app_context(): r=provider({})
    assert r['blocks'][0]['citations'] and not r['web_search_verified']


@pytest.mark.parametrize('failure', ['incomplete','failed','invalid_output','invalid_content','missing_id'])
def test_failure_retains_metadata_but_no_answer(app, monkeypatch,failure):
    enable(app);data=response(failure if failure in ('incomplete','failed') else 'completed')
    if failure=='invalid_output':data['output']={}
    if failure=='invalid_content':data['output']=[{'type':'message','content':{}}]
    if failure=='missing_id':data.pop('id')
    mock_response(monkeypatch,data)
    with app.app_context(),pytest.raises(ProviderError) as exc:provider({})
    assert exc.value.metadata['usage']['total_tokens']==52
    assert 'blocks' not in exc.value.metadata


def body(**kw):
    return dict(kind='research', question='Public policy', consent=True,public_question_confirmed=True, **kw)


def test_smoke_fixed_bounded_once_and_history(app,monkeypatch):
    enable(app);c,h=client_for(app);seen=mock_response(monkeypatch,response())
    r=c.post('/api/casework/ai',json=body(profile=SMOKE_PROFILE),headers=h)
    assert r.status_code==200
    payload=seen[0]['json']
    assert payload['input']==SMOKE_QUESTION and payload['service_tier']=='default'
    assert payload['max_output_tokens']==600 and payload['max_tool_calls']==1
    assert payload['tools']==[{'type':'web_search','search_context_size':'low','return_token_budget':'default','external_web_access':True}]
    assert payload['tool_choice']=='required' and 'previous_response_id' not in payload
    assert c.post('/api/casework/ai',json=body(profile=SMOKE_PROFILE),headers=h).status_code==409
    other,oh=client_for(app,'test:other')
    assert other.post('/api/casework/ai',json=body(profile=SMOKE_PROFILE),headers=oh).status_code==409
    assert len(seen)==1
    history=c.get('/api/casework/research-history').get_json()['outputs']
    assert len(history)==1 and history[0]['response_id']=='resp_real'
    assert history[0]['reservation_usd']==3.5 and history[0]['status']=='completed'
    assert other.get('/api/casework/research-history').get_json()['outputs']==[]


def test_incomplete_is_persisted_not_accepted_or_retried(app,monkeypatch):
    enable(app);c,h=client_for(app);seen=mock_response(monkeypatch,response('incomplete'))
    r=c.post('/api/casework/ai',json=body(profile=SMOKE_PROFILE),headers=h)
    assert r.status_code==502 and r.get_json()['usage']['total_tokens']==52
    with app.extensions['studio_store'].connect() as db:
        assert db.execute('SELECT data FROM casework_outputs').fetchone()['data'] is None
        assert db.execute('SELECT status FROM casework_execution').fetchone()['status']=='failed'
    saved=c.get('/api/casework/research-history').get_json()['outputs'][0]
    assert saved['blocks']==[] and saved['provider_status']=='incomplete'
    assert c.post('/api/casework/ai',json=body(profile=SMOKE_PROFILE),headers=h).status_code==409
    assert len(seen)==1


def test_timeout_and_gates(app,monkeypatch):
    enable(app);c,h=client_for(app);calls=[]
    def timeout(*a,**kw):calls.append(1);raise requests.Timeout('SECRET')
    monkeypatch.setattr('dredge.research_provider.requests.post',timeout)
    assert c.post('/api/casework/ai',json=body(profile=SMOKE_PROFILE,case_id='private'),headers=h).status_code==400
    assert not calls
    r=c.post('/api/casework/ai',json=body(profile=SMOKE_PROFILE),headers=h)
    assert r.status_code==502 and b'SECRET' not in r.data
    assert r.get_json()['usage'] is None and r.get_json()['cost_usd'] is None
    assert c.post('/api/casework/ai',json=body(profile=SMOKE_PROFILE),headers=h).status_code==409
    assert len(calls)==1


def test_status_requires_completed_search_and_case_law_tracks_actual_calls(app,monkeypatch):
    enable(app);c,h=client_for(app)
    data=response();data['output']=data['output'][2:];mock_response(monkeypatch,data)
    r=c.post('/api/casework/ai',json=body(),headers=h)
    assert r.status_code==200
    web=lambda: next(x for x in c.get('/api/studio/status').get_json()['tools'] if x['name']=='OpenAI web research')
    assert web()['status']=='configured_not_verified'
    seen=mock_response(monkeypatch,response())
    r=c.post('/api/casework/ai',json={**body(),'kind':'case_law','jurisdiction':'Missouri'},headers=h)
    assert r.status_code==200 and r.get_json()['web_search_verified']
    assert json.loads(seen[0]['json']['input'])['jurisdiction']=='Missouri'
    assert web()['status']=='successful_request_recorded'


@pytest.mark.parametrize('bad_type', [[], {}, None, 7])
def test_malformed_search_action_does_not_lose_usage(app,monkeypatch,bad_type):
    enable(app);data=response('incomplete');data['output'][1]['action']['type']=bad_type
    data['output'][1]['action']['sources'] += [{'url':'https://example.org/?access_token=secret'}, {'url':'https://example.org/public#secret'}]
    mock_response(monkeypatch,data);c,h=client_for(app)
    r=c.post('/api/casework/ai',json=body(),headers=h)
    assert r.status_code==502 and r.get_json()['usage']['total_tokens']==52
    assert r.get_json()['response_id']=='resp_real'
    assert 'secret' not in json.dumps(r.get_json())
    assert 'https://example.org/public' in r.get_json()['search_calls'][0]['action']['sources']


def test_status_survives_missing_or_changed_encryption_key(app,monkeypatch):
    from cryptography.fernet import Fernet
    enable(app);c,h=client_for(app);mock_response(monkeypatch,response())
    assert c.post('/api/casework/ai',json=body(),headers=h).status_code==200
    for key in (None, Fernet(Fernet.generate_key())):
        app.extensions['casework_cipher']=key
        result=c.get('/api/studio/status')
        assert result.status_code==200
        web=next(x for x in result.get_json()['tools'] if x['name']=='OpenAI web research')
        assert web['last_observed'] is None


@pytest.mark.parametrize('key', ['sig','auth','key','code','X-Amz-Signature','session_id','api_key'])
def test_search_urls_never_retain_credential_query_keys(key):
    from dredge.research_provider import safe_url
    assert safe_url('https://example.org/doc?'+key+'=secret') is None
    assert safe_url('https://example.org/statute?section=12')=='https://example.org/statute?section=12'
