import io
import json
from test_studio import app,client_for
from test_casework import enable,create


def setup(app):
    enable(app);c,h=client_for(app);case=create(c,h)
    file=c.post(f'/api/casework/cases/{case}/files',data={'file':(io.BytesIO(b'FICTIONAL payment record 900'),'record.txt')},headers=h).get_json()['id']
    return c,h,case,file


def test_review_is_encrypted_source_linked_and_concurrency_guarded(app):
    c,h,case,file=setup(app);url=f'/api/casework/cases/{case}/review'
    body=dict(action='event',version=0,date='2026-01-01',kind='payment',amount='900.00',description='PRIVATE observed amount',file_id=file,locator='Line 1')
    assert c.post(url,json=body).status_code==403
    assert c.post(url,json={**body,'file_id':'another-case'},headers=h).status_code==400
    assert c.post(url,json={**body,'amount':'NaN'},headers=h).status_code==400
    assert c.post(url,json=body,headers=h).status_code==200
    assert c.post(url,json=body,headers=h).status_code==409
    data=c.get(url).get_json();assert data['integration']=='not_connected'
    event=data['events'][0]['id']
    assert c.post(url,json=dict(action='issue',version=1,event_ids=[event],question='Which period does this record cover?'),headers=h).status_code==200
    issue=c.get(url).get_json()['issues'][0]['id']
    assert c.post(url,json=dict(action='resolve',version=2,issue_id=issue,status='resolved',resolution=''),headers=h).status_code==400
    other,oh=client_for(app,'test:other');assert other.get(url).status_code==404
    cross,_=client_for(app,'test:review');assert cross.get(url).status_code==404
    with app.extensions['studio_store'].connect() as db:
        assert b'PRIVATE' not in db.execute('SELECT data FROM child_support_review').fetchone()['data']
        assert 'PRIVATE' not in json.dumps([dict(r) for r in db.execute('SELECT * FROM audit')])
    app.config['CASEWORK_REAL_DATA_ENABLED']=False
    assert c.post(url,json=dict(action='profile',version=2,origin='manual'),headers=h).status_code==409


def test_packet_requires_independent_review_and_fresh_snapshot(app):
    c,h,case,file=setup(app);url=f'/api/casework/cases/{case}'
    c.post(url+'/review',json=dict(action='event',version=0,date='2026-01-01',kind='notice',amount='',description='Notice date',file_id=file,locator='Page 1'),headers=h)
    packet=c.post(url+'/packets',json={},headers=h);assert packet.status_code==201
    pid=packet.get_json()['id'];purl=url+'/packets/'+pid
    attest=dict(evidence_verified=True,client_explanations_reviewed=True)
    assert c.post(purl,json=attest,headers=h).status_code==403
    admin,ah=client_for(app,'test:admin')
    assert admin.post(purl,json={},headers=ah).status_code==400
    assert admin.post(purl,json=attest,headers=ah).status_code==200
    assert c.get(purl).get_json()['status']=='reviewed'
    old=c.get(purl).get_json()['snapshot']
    c.post(url+'/review',json=dict(action='profile',version=1,origin='manual',reference='REF',purpose='Clarify date'),headers=h)
    assert c.get(purl).get_json()['matches_current_case'] is False
    assert c.get(purl).get_json()['snapshot']==old
    pid=c.post(url+'/packets',json={},headers=h).get_json()['id']
    # New client explanation invalidates the packet even when review version stays the same.
    from dredge.casework import encrypt
    with app.extensions['studio_store'].connect() as db:
        db.execute('INSERT INTO client_messages VALUES(?,?,?,?,?)',('message',case,'test:client',app.extensions['casework_cipher'].encrypt(json.dumps({'text':'Fictional explanation'}).encode()),1))
    assert admin.post(url+'/packets/'+pid,json=attest,headers=ah).status_code==409


def test_export_attestation_and_packet_authorization(app):
    c,h,case,file=setup(app);url=f'/api/casework/cases/{case}'
    assert c.post(url+'/review',json=dict(action='profile',version=0,origin='approved_export'),headers=h).status_code==400
    assert c.post(url+'/review',json=dict(action='profile',version=0,origin='approved_export',authorized=True),headers=h).status_code==200
    assert c.post(url+'/packets',json={},headers=h).status_code==400
    c.post(url+'/review',json=dict(action='event',version=1,date='2026-02-01',kind='document',amount='',description='Fictional evidence',file_id=file,locator='Line 1'),headers=h)
    pid=c.post(url+'/packets',json={},headers=h).get_json()['id']
    other,oh=client_for(app,'test:other');assert other.get(url+'/packets/'+pid).status_code==404
    assert other.post(url+'/packets/'+pid,json={},headers=oh).status_code==404
    c.post(url+'/archive',json={},headers=h)
    assert c.post(url+'/packets',json={},headers=h).status_code==409
