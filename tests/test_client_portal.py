import io
import json
from test_studio import app,client_for
from test_casework import enable,create


def assigned_case(app):
    enable(app);staff,h=client_for(app);case_id=create(staff,h)
    result=staff.post('/api/client/cases/'+case_id+'/assign',json={'client_id':'github:client','identity_verified':True},headers=h)
    assert result.status_code==200
    client,ch=client_for(app,'github:client')
    return staff,h,client,ch,case_id


def test_client_access_is_assigned_not_studio_role(app):
    staff,h,c,ch,case_id=assigned_case(app)
    assert c.get('/api/client/cases').get_json()['cases'][0]['id']==case_id
    assert c.get('/api/casework/cases').status_code==403
    other,oh=client_for(app,'github:stranger')
    assert other.get('/api/client/cases').get_json()['cases']==[]
    path=f'/api/client/cases/{case_id}/documents'
    assert other.get(path).status_code==404
    assert c.post(path,data={'file':(io.BytesIO(b'Client evidence'),'evidence.txt'),'consent':'true'}).status_code==403
    assert c.post(path,data={'file':(io.BytesIO(b'Client evidence'),'evidence.txt')},headers=ch).status_code==400


def test_receipt_encryption_redaction_and_transparency(app):
    staff,h,c,ch,case_id=assigned_case(app);path=f'/api/client/cases/{case_id}/documents'
    receipt=c.post(path,data={'file':(io.BytesIO(b'Name Alice. Identifier 12345.'),'identity.txt'),'consent':'true'},headers=ch)
    assert receipt.status_code==201
    d=receipt.get_json();assert d['ai_reviewed'] is False and len(d['sha256'])==64
    assert staff.get(f'/api/casework/cases/{case_id}/files/{d["id"]}').status_code==200
    copy=c.post(path+'/'+d['id']+'/redact',json={'terms':['Alice','12345']},headers=ch)
    assert copy.status_code==201
    assert 'Alice' not in copy.get_json()['preview']
    assert c.get(path+'/'+copy.get_json()['id']).data==b'Name [REDACTED]. Identifier [REDACTED].'
    assert c.get(path+'/'+d['id']).data==b'Name Alice. Identifier 12345.'
    events=c.get(f'/api/client/cases/{case_id}/transparency').get_json()['events']
    assert any(e['action']=='casework.file_downloaded' for e in events)
    with app.extensions['studio_store'].connect() as db:
        assert all(b'Alice' not in r['data'] for r in db.execute('SELECT data FROM client_documents'))
        assert 'Alice' not in json.dumps([dict(r) for r in db.execute('SELECT * FROM audit')])


def test_ai_discernment_includes_client_explanation_without_web(app,monkeypatch):
    staff,h,c,ch,case_id=assigned_case(app)
    file_id=c.post(f'/api/client/cases/{case_id}/documents',data={'file':(io.BytesIO(b'Reported amount 100'),'a.txt'),'consent':'true'},headers=ch).get_json()['id']
    assert c.post(f'/api/client/cases/{case_id}/explanations',json={'text':'The amount was corrected later.'},headers=ch).status_code==201
    payload=[]
    monkeypatch.setattr('dredge.casework.provider',lambda p:payload.append(p) or {'blocks':[],'usage':{}})
    result=staff.post('/api/casework/ai',json={'kind':'discernment','question':'Compare records','case_id':case_id,'file_ids':[file_id],'consent':True,'include_client_explanations':True},headers=h)
    assert result.status_code==200
    assert 'tools' not in payload[0]
    assert json.loads(payload[0]['input'])['client_explanations']==['The amount was corrected later.']
    assert 'Never infer fraud' in payload[0]['instructions']
    assert c.get(f'/api/client/cases/{case_id}/drafts').get_json()['drafts']==[]
    output_id=result.get_json()['id']
    assert staff.post(f'/api/client/staff/cases/{case_id}/release/{output_id}',json={'evidence_verified':True},headers=h).status_code==403
    reviewer,rh=client_for(app,'test:admin')
    assert reviewer.post(f'/api/client/staff/cases/{case_id}/release/{output_id}',json={'evidence_verified':True},headers=rh).status_code==200
    assert len(c.get(f'/api/client/cases/{case_id}/drafts').get_json()['drafts'])==1


def test_per_case_estimate_no_payment_or_attorney_claim(app):
    staff,h,c,ch,case_id=assigned_case(app)
    request=c.post(f'/api/client/cases/{case_id}/legal-preparation',json={'jurisdiction':'Missouri','scope':'Evidence timeline'},headers=ch)
    assert request.status_code==201
    quote_id=request.get_json()['id']
    provider,ph=client_for(app,'test:admin');app.config['LEGAL_PREPARATION_PROVIDER_IDS']={'test:admin'}
    assert provider.post('/api/client/staff/preparation/'+quote_id,json={'fee_usd':150,'scope':'Timeline and draft letter'},headers=ph).status_code==200
    quote=c.get(f'/api/client/cases/{case_id}/legal-preparation').get_json()['requests'][0]
    assert quote['fee_usd']==150 and quote['payment_status']=='not_collected'
    assert quote['attorney_review_included'] is False


def test_photos_store_without_fake_ocr_or_identity_claim(app):
    from PIL import Image
    staff,h,c,ch,case_id=assigned_case(app);buffer=io.BytesIO();Image.new('RGB',(10,10)).save(buffer,format='PNG');buffer.seek(0)
    result=c.post(f'/api/client/cases/{case_id}/documents',data={'file':(buffer,'id.png'),'consent':'true'},headers=ch)
    assert result.status_code==201
    assert staff.post('/api/casework/ai',json={'kind':'analysis','question':'Review','case_id':case_id,'file_ids':[result.get_json()['id']],'consent':True},headers=h).status_code==409
