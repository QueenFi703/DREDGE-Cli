import io
import json
import pytest
from cryptography.fernet import Fernet
from test_studio import app, client_for


def enable(app):
    app.extensions['casework_cipher'] = Fernet(Fernet.generate_key())
    app.config.update(CASEWORK_REAL_DATA_ENABLED=True, CASEWORK_AI_DATA_ENABLED=True,
      OPENAI_API_KEY='test-only-not-a-live-key', CASEWORK_MEMBERS={
        'test:op': {'agency':'a', 'role':'caseworker'},
        'test:other': {'agency':'a', 'role':'caseworker'},
        'test:review': {'agency':'b', 'role':'supervisor'},
        'test:admin': {'agency':'a', 'role':'supervisor'}})


def create(client, headers):
    result = client.post('/api/casework/cases', json={'title':'PRIVATE CLIENT REFERENCE'}, headers=headers)
    assert result.status_code == 201
    return result.get_json()['id']


def test_fail_closed_without_affecting_studio(app):
    c,h = client_for(app)
    assert c.get('/api/casework/cases').status_code == 403
    assert c.get('/api/studio/session').status_code == 200
    enable(app)
    app.config['CASEWORK_REAL_DATA_ENABLED'] = False
    assert c.post('/api/casework/cases',json={'title':'Test'},headers=h).status_code == 409
    assert c.post('/api/casework/cases',json={'title':'Test'}).status_code == 403


def test_tenant_assignment_and_encryption(app):
    enable(app)
    c,h=client_for(app); case_id=create(c,h)
    uploaded=c.post(f'/api/casework/cases/{case_id}/files',data={'file':(io.BytesIO(b'CLIENT SECRET EVIDENCE'),'private.txt')},headers=h)
    assert uploaded.status_code == 201
    file_id=uploaded.get_json()['id']
    for user in ('test:other','test:review'):
        other,_=client_for(app,user)
        assert other.get('/api/casework/cases').get_json()['cases'] == []
        assert other.get(f'/api/casework/cases/{case_id}').status_code == 404
        assert other.get(f'/api/casework/cases/{case_id}/files/{file_id}').status_code == 404
    supervisor,_=client_for(app,'test:admin')
    assert supervisor.get(f'/api/casework/cases/{case_id}').status_code == 200
    response=c.get(f'/api/casework/cases/{case_id}/files/{file_id}')
    assert response.data == b'CLIENT SECRET EVIDENCE'
    assert response.headers['Cache-Control'] == 'no-store'
    with app.extensions['studio_store'].connect() as db:
        rows=db.execute('SELECT data FROM casework_cases').fetchall()+db.execute('SELECT data FROM casework_files').fetchall()
        assert all(b'PRIVATE CLIENT' not in r['data'] and b'CLIENT SECRET' not in r['data'] for r in rows)
        assert 'CLIENT SECRET' not in json.dumps([dict(r) for r in db.execute('SELECT * FROM audit')])
        assert db.execute('SELECT COUNT(*) FROM runs').fetchone()[0] == 0


def test_upload_validation_archive_and_size(app):
    enable(app);c,h=client_for(app);case_id=create(c,h)
    path=f'/api/casework/cases/{case_id}/files'
    for name,content in [('bad.html',b'<script>'),('fake.pdf',b'not pdf'),('null.txt',b'hi\x00there')]:
        assert c.post(path,data={'file':(io.BytesIO(content),name)},headers=h).status_code == 400
    # Larger than global 64 KB works at the upload endpoint.
    assert c.post(path,data={'file':(io.BytesIO(b'x'*70000),'ok.txt')},headers=h).status_code == 201
    assert c.post(path,data={'file':(io.BytesIO(b'x'*(5*1024*1024+1)),'big.txt')},headers=h).status_code == 400
    assert c.post(f'/api/casework/cases/{case_id}/archive',json={},headers=h).status_code == 200
    assert c.post(path,data={'file':(io.BytesIO(b'x'),'ok.txt')},headers=h).status_code == 409


def test_web_research_never_receives_case_data(app, monkeypatch):
    enable(app);c,h=client_for(app);case_id=create(c,h)
    captured=[]
    def fake(payload):
        captured.append(payload)
        return {'blocks':[{'text':'A policy', 'citations':[]}], 'usage':{}, 'model':'gpt-6-astra','mode':'live','human_review_required':True}
    monkeypatch.setattr('dredge.casework.provider',fake)
    body=dict(kind='research',question='General public policy',consent=True,public_question_confirmed=True)
    assert c.post('/api/casework/ai',json={**body,'case_id':case_id},headers=h).status_code == 400
    assert not captured
    assert c.post('/api/casework/ai',json=body,headers=h).status_code == 200
    assert captured[0]['input'] == 'General public policy'
    assert captured[0]['tools'] == [{'type':'web_search'}]
    assert captured[0]['store'] is False
    assert 'PRIVATE CLIENT' not in json.dumps(captured)
    with app.extensions['studio_store'].connect() as db:
        assert db.execute('SELECT COUNT(*) FROM casework_outputs WHERE data IS NOT NULL').fetchone()[0] == 1


def test_analysis_requires_approval_selected_files_and_no_tools(app, monkeypatch):
    enable(app);c,h=client_for(app);case_id=create(c,h)
    file_id=c.post(f'/api/casework/cases/{case_id}/files',data={'file':(io.BytesIO(b'Evidence text'),'a.txt')},headers=h).get_json()['id']
    captured=[]
    monkeypatch.setattr('dredge.casework.provider',lambda p: captured.append(p) or {'blocks':[],'usage':{}})
    body=dict(kind='analysis',case_id=case_id,file_ids=[file_id],question='Summarize',consent=True)
    app.config['CASEWORK_AI_DATA_ENABLED']=False
    assert c.post('/api/casework/ai',json=body,headers=h).status_code == 409
    assert not captured
    app.config['CASEWORK_AI_DATA_ENABLED']=True
    assert c.post('/api/casework/ai',json={**body,'consent':False},headers=h).status_code == 400
    assert c.post('/api/casework/ai',json=body,headers=h).status_code == 200
    assert 'tools' not in captured[0]
    assert json.loads(captured[0]['input'])['evidence'] == [{'id':file_id,'text':'Evidence text'}]


def test_enterprise_quote_no_charge_or_access_grant(app):
    enable(app);c,h=client_for(app,'test:viewer')
    result=c.post('/api/casework/enterprise-quote',json={'agency':'Pilot Agency','seats':15},headers=h)
    assert result.status_code == 201
    assert result.get_json()['interval'] == 'year'
    assert app.config['CASEWORK_MEMBERS'].get('test:viewer') is None
    assert c.post('/api/casework/enterprise-quote',json={'agency':'Pilot Agency','seats':15}).status_code == 403


def test_daily_limit_prevents_provider_spend(app, monkeypatch):
    enable(app);c,h=client_for(app)
    monkeypatch.setattr('dredge.casework.provider',lambda p:{'blocks':[],'usage':{}})
    body=dict(kind='research',question='Public policy',consent=True,public_question_confirmed=True)
    for _ in range(20):
        assert c.post('/api/casework/ai',json=body,headers=h).status_code == 200
    assert c.post('/api/casework/ai',json=body,headers=h).status_code == 429


def test_provider_contract_citations_and_failure(app, monkeypatch):
    enable(app)
    from dredge.casework import provider
    seen=[]
    class Response:
        def raise_for_status(self): pass
        def json(self):
            return {'status':'completed','usage':{'input_tokens':30,'output_tokens':10},'output':[{'type':'message','content':[{'type':'output_text','text':'See source','annotations':[{'type':'url_citation','url':'https://www.usa.gov/benefits','title':'Benefits','start_index':4,'end_index':10},{'type':'url_citation','url':'javascript:alert(1)'}]}]}]}
    def post(url, **kwargs):
        seen.append((url,kwargs));return Response()
    monkeypatch.setattr('dredge.casework.requests.post',post)
    with app.app_context():
        result=provider({'model':'gpt-6-astra','store':False})
    assert seen[0][0] == 'https://api.openai.com/v1/responses'
    assert seen[0][1]['timeout'] == (10,45)
    assert result['blocks'][0]['citations'] == [{'url':'https://www.usa.gov/benefits','title':'Benefits','start':4,'end':10}]
    def fail(p): raise ValueError('Provider private error with SECRET')
    monkeypatch.setattr('dredge.casework.provider',fail)
    c,h=client_for(app)
    response=c.post('/api/casework/ai',json=dict(kind='research',question='Policy',consent=True,public_question_confirmed=True),headers=h)
    assert response.status_code == 502
    assert b'SECRET' not in response.data
    with app.extensions['studio_store'].connect() as db:
        assert db.execute('SELECT COUNT(*) FROM casework_outputs WHERE data IS NULL').fetchone()[0] == 1
