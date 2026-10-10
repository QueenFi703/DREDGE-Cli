"""Client-owned submissions, receipts and preparation requests; no role grants."""
import base64
import hashlib
import hmac
import io
import json
import secrets
import time
import uuid
from functools import wraps
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request, session, send_file
from flask_login import current_user, login_required
from werkzeug.utils import secure_filename
from .casework import cipher, encrypt, decrypt, get_case, protected, json_body, MAX_FILE
from .studio import store

bp = Blueprint('clients', __name__)


def register_clients(app):
    with app.extensions['studio_store'].connect() as db:
        db.executescript('''
        CREATE TABLE IF NOT EXISTS client_releases(output_id TEXT PRIMARY KEY, case_id TEXT NOT NULL, reviewer TEXT NOT NULL, created REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS case_clients(case_id TEXT NOT NULL, client TEXT NOT NULL,
          assigned_by TEXT NOT NULL, created REAL NOT NULL, PRIMARY KEY(case_id,client));
        CREATE TABLE IF NOT EXISTS client_documents(id TEXT PRIMARY KEY, case_id TEXT NOT NULL,
          owner TEXT NOT NULL, data BLOB NOT NULL, created REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS client_messages(id TEXT PRIMARY KEY, case_id TEXT NOT NULL,
          owner TEXT NOT NULL, data BLOB NOT NULL, created REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS legal_preparation(id TEXT PRIMARY KEY, case_id TEXT NOT NULL,
          owner TEXT NOT NULL, state TEXT NOT NULL, data BLOB NOT NULL, created REAL NOT NULL);
        ''')
    app.register_blueprint(bp)


def client_guard(fn):
    @wraps(fn)
    @login_required
    def wrapped(*args, **kwargs):
        if not cipher():
            return jsonify(error='Encrypted storage is not configured.'),503
        return fn(*args, **kwargs)
    return wrapped


def assigned(db,case_id):
    return db.execute('SELECT 1 FROM case_clients cc JOIN casework_cases c ON c.id=cc.case_id WHERE cc.case_id=? AND cc.client=? AND c.archived=0',(case_id,current_user.get_id())).fetchone()


def event(db,action,object_id):
    store().audit(db,current_user.get_id(),'client.'+action,object_id)


@bp.before_request
def guard():
    if request.path.startswith('/api/client/') and request.method=='POST':
        if not current_user.is_authenticated:
            return jsonify(error='Sign in again.',code='session_expired'),401
        token=session.get('studio_csrf','')
        if not token or not hmac.compare_digest(token,request.headers.get('X-CSRF-Token','')):
            return jsonify(error='Refresh before submitting.'),403
    if request.endpoint=='clients.upload':
        request.max_content_length=MAX_FILE+65536


@bp.get('/client')
@login_required
def portal():
    return send_file(Path(__file__).parent/'static'/'studio_client.html')


@bp.get('/api/client/session')
@login_required
def status():
    return jsonify(account_id=current_user.get_id(),csrf_token=session.setdefault('studio_csrf',secrets.token_urlsafe(32)),encrypted_storage=bool(cipher()),uploads_enabled=current_app.config['CASEWORK_REAL_DATA_ENABLED'])


@bp.get('/api/client/cases')
@client_guard
def cases():
    with store().connect() as db:
        rows=db.execute('SELECT c.* FROM casework_cases c JOIN case_clients cc ON cc.case_id=c.id WHERE cc.client=? AND c.archived=0',(current_user.get_id(),)).fetchall()
        return jsonify(cases=[dict(id=r['id'],title=decrypt(r['data'])['title']) for r in rows])


@bp.post('/api/client/cases/<case_id>/assign')
@protected
def assign(case_id):
    data=json_body();client_id=data.get('client_id')
    if not isinstance(client_id,str) or not client_id.startswith(('github:','google:')) or len(client_id)>160 or data.get('identity_verified') is not True:
        return jsonify(error='Enter the client’s verified signed-in account ID and confirm the identity check.'),400
    with store().connect() as db:
        if not get_case(db,case_id):
            return jsonify(error='Case not found.'),404
        db.execute('INSERT OR IGNORE INTO case_clients VALUES(?,?,?,?)',(case_id,client_id,current_user.get_id(),time.time()))
        event(db,'assigned',case_id)
    return jsonify(assigned=True)


def read_document(file):
    name=secure_filename(file.filename or '')
    content=file.read(MAX_FILE+1)
    if not content or len(content)>MAX_FILE:
        raise ValueError('Use a file containing data, up to 5 MB.')
    text=''
    if name.lower().endswith('.txt'):
        text=content.decode('utf-8');mime='text/plain'
        if '\x00' in text or len(text)>100000:raise ValueError('Invalid text document.')
    elif name.lower().endswith('.pdf') and content.startswith(b'%PDF-'):
        from pypdf import PdfReader
        reader=PdfReader(io.BytesIO(content))
        if reader.is_encrypted or len(reader.pages)>100:raise ValueError('Use an unencrypted PDF of up to 100 pages.')
        chunks=[]
        for page in reader.pages:
            chunk=page.extract_text() or ''
            chunks.append(chunk)
            if sum(len(c) for c in chunks)>100000:raise ValueError('Document text is too long.')
        text='\n'.join(chunks);mime='application/pdf'
    elif name.lower().endswith(('.jpg','.jpeg','.png')):
        from PIL import Image
        image=Image.open(io.BytesIO(content))
        if image.format not in {'JPEG','PNG'} or image.width*image.height>20000000:raise ValueError('Use a JPEG or PNG image of at most 20 megapixels.')
        image.verify();mime='image/jpeg' if image.format=='JPEG' else 'image/png'
    else:
        raise ValueError('Use TXT, PDF, JPEG or PNG.')
    return dict(name=name,mime=mime,text=text,content=base64.b64encode(content).decode(),sha256=hashlib.sha256(content).hexdigest(),kind='original')


@bp.route('/api/client/cases/<case_id>/documents',methods=['GET','POST'])
@client_guard
def upload(case_id):
    with store().connect() as db:
        if not assigned(db,case_id):return jsonify(error='Case not found.'),404
        if request.method=='GET':
            rows=db.execute('SELECT * FROM client_documents WHERE case_id=? AND owner=? ORDER BY created',(case_id,current_user.get_id())).fetchall()
            return jsonify(documents=[dict(id=r['id'],created=r['created'],**{k:v for k,v in decrypt(r['data']).items() if k in {'name','mime','sha256','kind','source_id'}}) for r in rows])
        if not current_app.config['CASEWORK_REAL_DATA_ENABLED']:return jsonify(error='Uploads await agency data handling approval.'),409
        if request.form.get('consent')!='true':return jsonify(error='Confirm that this document may be stored and reviewed by assigned agency staff.'),400
        if db.execute('SELECT COUNT(*) FROM client_documents WHERE case_id=? AND owner=?',(case_id,current_user.get_id())).fetchone()[0]>=50:return jsonify(error='Document limit reached.'),409
        file=request.files.get('file')
        if not file:return jsonify(error='Choose a document.'),400
        try:data=read_document(file)
        except Exception:return jsonify(error='Use a valid TXT, unencrypted PDF, JPEG or PNG, up to 5 MB. Scanned documents are stored without OCR.'),400
        document_id=str(uuid.uuid4())
        db.execute('INSERT INTO client_documents VALUES(?,?,?,?,?)',(document_id,case_id,current_user.get_id(),encrypt(data),time.time()))
        # Original evidence is also available through existing assigned-staff workflow.
        db.execute('INSERT INTO casework_files VALUES(?,?,?,?)',(document_id,case_id,encrypt(data),time.time()))
        event(db,'document_received',document_id)
        return jsonify(id=document_id,name=data['name'],sha256=data['sha256'],status='received',ai_reviewed=False),201


@bp.get('/api/client/cases/<case_id>/documents/<document_id>')
@client_guard
def download(case_id,document_id):
    with store().connect() as db:
        if not assigned(db,case_id):return jsonify(error='Document not found.'),404
        row=db.execute('SELECT data FROM client_documents WHERE id=? AND case_id=? AND owner=?',(document_id,case_id,current_user.get_id())).fetchone()
        if not row:return jsonify(error='Document not found.'),404
        data=decrypt(row['data']);event(db,'document_downloaded',document_id)
    return send_file(io.BytesIO(base64.b64decode(data['content'])),mimetype=data['mime'],as_attachment=True,download_name=data['name'])


@bp.post('/api/client/cases/<case_id>/documents/<document_id>/redact')
@client_guard
def redact(case_id,document_id):
    body=json_body();terms=body.get('terms')
    if not isinstance(terms,list) or not 1<=len(terms)<=30 or any(not isinstance(t,str) or not 1<=len(t)<=200 for t in terms):return jsonify(error='Enter 1–30 exact phrases to remove.'),400
    with store().connect() as db:
        if not assigned(db,case_id):return jsonify(error='Document not found.'),404
        row=db.execute('SELECT data FROM client_documents WHERE id=? AND case_id=? AND owner=?',(document_id,case_id,current_user.get_id())).fetchone()
        if not row:return jsonify(error='Document not found.'),404
        original=decrypt(row['data'])
        if not original['text'].strip():return jsonify(error='Redaction supports extracted text only. Scans and photos need specialist redaction.'),409
        text=original['text']
        for term in sorted(terms,key=len,reverse=True):text=text.replace(term,'[REDACTED]')
        content=text.encode();new_id=str(uuid.uuid4())
        data=dict(name='redacted-text.txt',mime='text/plain',text=text,content=base64.b64encode(content).decode(),sha256=hashlib.sha256(content).hexdigest(),kind='redacted_text_copy',source_id=document_id)
        db.execute('INSERT INTO client_documents VALUES(?,?,?,?,?)',(new_id,case_id,current_user.get_id(),encrypt(data),time.time()))
        event(db,'redacted_text_created',new_id)
    return jsonify(id=new_id,preview=text,warning='Text-only copy. Exact matches removed; review for spelling variants and indirect identifiers. Original PDF or image is unchanged.'),201


@bp.route('/api/client/cases/<case_id>/explanations',methods=['GET','POST'])
@client_guard
def explanations(case_id):
    with store().connect() as db:
        if not assigned(db,case_id):return jsonify(error='Case not found.'),404
        if request.method=='GET':
            rows=db.execute('SELECT * FROM client_messages WHERE case_id=? AND owner=? ORDER BY created',(case_id,current_user.get_id())).fetchall()
            return jsonify(explanations=[dict(id=r['id'],created=r['created'],**decrypt(r['data'])) for r in rows])
        text=json_body().get('text')
        if not isinstance(text,str) or not 1<=len(text.strip())<=4000:return jsonify(error='Enter an explanation up to 4,000 characters.'),400
        if not current_app.config['CASEWORK_REAL_DATA_ENABLED']:return jsonify(error='Case submissions await agency approval.'),409
        message_id=str(uuid.uuid4());db.execute('INSERT INTO client_messages VALUES(?,?,?,?,?)',(message_id,case_id,current_user.get_id(),encrypt(dict(text=text)),time.time()));event(db,'explanation_received',message_id)
        return jsonify(id=message_id,status='received'),201


@bp.route('/api/client/cases/<case_id>/legal-preparation',methods=['GET','POST'])
@client_guard
def preparation(case_id):
    with store().connect() as db:
        if not assigned(db,case_id):return jsonify(error='Case not found.'),404
        if request.method=='GET':
            rows=db.execute('SELECT * FROM legal_preparation WHERE case_id=? AND owner=? ORDER BY created',(case_id,current_user.get_id())).fetchall()
            return jsonify(requests=[dict(id=r['id'],state=r['state'],**decrypt(r['data'])) for r in rows])
        data=json_body();jurisdiction=data.get('jurisdiction');scope=data.get('scope')
        if not isinstance(jurisdiction,str) or not 1<=len(jurisdiction)<=120 or not isinstance(scope,str) or not 1<=len(scope)<=2000:return jsonify(error='Enter jurisdiction and the preparation scope.'),400
        if not current_app.config['CASEWORK_REAL_DATA_ENABLED']:return jsonify(error='Preparation intake awaits agency approval.'),409
        request_id=str(uuid.uuid4());db.execute('INSERT INTO legal_preparation VALUES(?,?,?,?,?,?)',(request_id,case_id,current_user.get_id(),'quote_requested',encrypt(dict(jurisdiction=jurisdiction,scope=scope,provider='ColeWorld inc. / Cultivating Faith',fee_usd=None,attorney_review_included=False)),time.time()))
        event(db,'legal_quote_requested',request_id)
    return jsonify(id=request_id,state='quote_requested',message='Per-case quote requested. No payment taken; no attorney review included.'),201


@bp.get('/api/client/cases/<case_id>/transparency')
@client_guard
def transparency(case_id):
    with store().connect() as db:
        if not assigned(db,case_id):return jsonify(error='Case not found.'),404
        ids=[r['id'] for r in db.execute('SELECT id FROM client_documents WHERE case_id=? AND owner=?',(case_id,current_user.get_id()))]
        events=[]
        for object_id in ids:
            rows=db.execute('SELECT recorded,action,actor FROM audit WHERE run_id=? ORDER BY seq',(object_id,)).fetchall()
            events.extend(dict(recorded=r['recorded'],action=r['action'],actor=r['actor']) for r in rows)
        return jsonify(events=sorted(events,key=lambda r:r['recorded']),scope='Recorded actions on your submitted documents. This is an activity log, not a complete external disclosure ledger.')


@bp.get('/api/client/staff/cases/<case_id>')
@protected
def staff_intake(case_id):
    with store().connect() as db:
        if not get_case(db,case_id):return jsonify(error='Case not found.'),404
        explanations=db.execute('SELECT * FROM client_messages WHERE case_id=? ORDER BY created',(case_id,)).fetchall()
        quotes=db.execute('SELECT * FROM legal_preparation WHERE case_id=? ORDER BY created',(case_id,)).fetchall()
        return jsonify(explanations=[dict(id=r['id'],**decrypt(r['data'])) for r in explanations],preparation_requests=[dict(id=r['id'],state=r['state'],**decrypt(r['data'])) for r in quotes])


@bp.route('/api/client/staff/preparation/<request_id>',methods=['POST'])
@protected
def preparation_quote(request_id):
    from .casework import member
    if current_user.get_id() not in current_app.config.get('LEGAL_PREPARATION_PROVIDER_IDS',{'github:174653334'}):
        return jsonify(error='Only the configured ColeWorld preparation provider can issue estimates.'),403
    if member()['role']!='supervisor':return jsonify(error='Agency supervisor required.'),403
    data=json_body();fee=data.get('fee_usd');scope=data.get('scope')
    if type(fee) is not int or not 1<=fee<=10000 or not isinstance(scope,str) or not 1<=len(scope)<=2000:return jsonify(error='Enter a whole-dollar per-case estimate and a preparation scope.'),400
    with store().connect() as db:
        row=db.execute('SELECT * FROM legal_preparation WHERE id=?',(request_id,)).fetchone()
        if not row or not get_case(db,row['case_id']):return jsonify(error='Request not found.'),404
        payload=decrypt(row['data']);payload.update(fee_usd=fee,scope=scope,attorney_review_included=False,payment_status='not_collected')
        db.execute('UPDATE legal_preparation SET data=?,state=? WHERE id=?',(encrypt(payload),'estimate_prepared',request_id));event(db,'legal_estimate_prepared',request_id)
    return jsonify(state='estimate_prepared',message='Estimate saved. Payment collection and attorney review are not included.')


@bp.route('/api/client/cases/<case_id>/drafts',methods=['GET'])
@client_guard
def drafts(case_id):
    with store().connect() as db:
        if not assigned(db,case_id):return jsonify(error='Case not found.'),404
        rows=db.execute('SELECT o.id,o.kind,o.data FROM casework_outputs o JOIN client_releases r ON r.output_id=o.id WHERE r.case_id=?',(case_id,)).fetchall()
        return jsonify(drafts=[dict(id=r['id'],kind=r['kind'],blocks=decrypt(r['data']).get('blocks',[]),status='human_reviewed_draft') for r in rows])


@bp.post('/api/client/staff/cases/<case_id>/release/<output_id>')
@protected
def release(case_id,output_id):
    data=json_body()
    if data.get('evidence_verified') is not True:return jsonify(error='Verify the evidence and client explanation before release.'),400
    with store().connect() as db:
        if not get_case(db,case_id):return jsonify(error='Case not found.'),404
        row=db.execute('SELECT * FROM casework_outputs WHERE id=? AND case_id=? AND data IS NOT NULL',(output_id,case_id)).fetchone()
        if not row:return jsonify(error='Draft not found.'),404
        if row['owner']==current_user.get_id():return jsonify(error='A different authorized staff member must review this draft.'),403
        if row['kind']=='legal_draft' and data.get('qualified_legal_review') is not True:return jsonify(error='Qualified legal review is required before releasing this draft.'),409
        db.execute('INSERT OR IGNORE INTO client_releases VALUES(?,?,?,?)',(output_id,case_id,current_user.get_id(),time.time()));event(db,'draft_released',output_id)
    return jsonify(released=True)
