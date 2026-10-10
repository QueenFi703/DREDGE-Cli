"""Evidence review companion. No MACSS connection, balance engine or enforcement actions."""
import hashlib
import json
import re
import time
import uuid
from datetime import date
from flask import current_app, jsonify, request
from flask_login import current_user
from .casework import bp, protected, store, get_case, encrypt, decrypt, audit, json_body
from .document_text import needs_review, revision


def register_review(app):
    with app.extensions['studio_store'].connect() as db:
        db.executescript('''CREATE TABLE IF NOT EXISTS child_support_review(
          case_id TEXT PRIMARY KEY, version INTEGER NOT NULL, data BLOB NOT NULL);
          CREATE TABLE IF NOT EXISTS child_support_packets(
          id TEXT PRIMARY KEY, case_id TEXT NOT NULL, creator TEXT NOT NULL,
          data BLOB NOT NULL, reviewer TEXT, reviewed REAL);''')


def state(db, case_id):
    row=db.execute('SELECT * FROM child_support_review WHERE case_id=?',(case_id,)).fetchone()
    return (row['version'],decrypt(row['data'])) if row else (0,dict(profile=dict(origin='fictional',reference='',purpose=''),events=[],issues=[]))


def text(value, maximum, required=True):
    if not isinstance(value,str) or len(value)>maximum or '\x00' in value or (required and not value.strip()):raise ValueError('Enter valid text within the stated field limits.')
    return value.strip()


def snapshot(db, case_id):
    version, review=state(db,case_id)
    files=[]
    for row in db.execute('SELECT * FROM casework_files WHERE case_id=? ORDER BY id',(case_id,)):
        doc=decrypt(row['data'])
        files.append(dict(id=row['id'],name=doc['name'],sha256=doc['sha256'],text_revision=revision(doc['text']),verified=not needs_review(doc) or doc.get('ocr_status')=='verified'))
    explanations=[dict(id=r['id'],**decrypt(r['data'])) for r in db.execute('SELECT * FROM client_messages WHERE case_id=? ORDER BY created',(case_id,))]
    drafts=[dict(id=r['id'],kind=r['kind'],draft=decrypt(r['data'])) for r in db.execute('SELECT * FROM casework_outputs WHERE case_id=? AND data IS NOT NULL ORDER BY created',(case_id,))]
    return dict(version=version,review=review,files=files,client_explanations=explanations,ai_drafts=drafts)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


@bp.route('/api/casework/cases/<case_id>/review',methods=['GET','POST'])
@protected
def review_case(case_id):
    with store().connect() as db:
        if request.method=='POST':db.execute('BEGIN IMMEDIATE')
        case=get_case(db,case_id)
        if not case:return jsonify(error='Case not found.'),404
        version, review=state(db,case_id)
        if request.method=='GET':
            packets=[dict(id=r['id'],creator=r['creator'],status='reviewed' if r['reviewer'] else 'awaiting_review',reviewer=r['reviewer']) for r in db.execute('SELECT * FROM child_support_packets WHERE case_id=?',(case_id,))]
            return jsonify(version=version,**review,packets=packets,integration='not_connected')
        if case['archived'] or not current_app.config['CASEWORK_REAL_DATA_ENABLED']:return jsonify(error='Case editing requires agency approval and an active case.'),409
        body=json_body()
        if type(body.get('version')) is not int or body['version']!=version:return jsonify(error='Case review changed. Reload before saving.'),409
        action=body.get('action')
        try:
            if action=='profile':
                if body.get('origin') not in {'fictional','manual','approved_export'}:raise ValueError('Choose fictional, manual entry or an agency-approved export.')
                if body.get('origin')=='approved_export' and body.get('authorized') is not True:raise ValueError('Confirm authorization to use this exported record.')
                review['profile']=dict(origin=body['origin'],reference=text(body.get('reference',''),120,False),purpose=text(body.get('purpose',''),1000,False))
            elif action=='event':
                if len(review['events'])>=200:raise ValueError('Maximum 200 timeline entries.')
                when=text(body.get('date'),10);date.fromisoformat(when)
                if not re.fullmatch(r'\d{4}-\d{2}-\d{2}',when):raise ValueError('Use YYYY-MM-DD.')
                kind=body.get('kind')
                if kind not in {'order','payment','notice','document','other'}:raise ValueError('Choose an entry type.')
                source=text(body.get('file_id'),100);row=db.execute('SELECT data FROM casework_files WHERE id=? AND case_id=?',(source,case_id)).fetchone()
                if not row:raise ValueError('Attach a source from this case.')
                doc=decrypt(row['data'])
                if needs_review(doc) and doc.get('ocr_status')!='verified':raise ValueError('Verify source OCR before adding a timeline entry.')
                amount=text(body.get('amount',''),16,False)
                if amount and not re.fullmatch(r'\d{1,10}(\.\d{1,2})?',amount):raise ValueError('Use a nonnegative USD amount with up to two decimal places.')
                review['events'].append(dict(id=str(uuid.uuid4()),date=when,kind=kind,description=text(body.get('description'),1000),amount=amount,file_id=source,locator=text(body.get('locator'),200),actor=current_user.get_id()))
            elif action=='issue':
                if len(review['issues'])>=100:raise ValueError('Maximum 100 review questions.')
                refs=body.get('event_ids')
                if not isinstance(refs,list) or not 1<=len(refs)<=20 or any(r not in {e['id'] for e in review['events']} for r in refs):raise ValueError('Select timeline entries supporting this question.')
                review['issues'].append(dict(id=str(uuid.uuid4()),question=text(body.get('question'),1000),event_ids=list(dict.fromkeys(refs)),status='open',resolution='',actor=current_user.get_id()))
            elif action=='resolve':
                issue=next((i for i in review['issues'] if i['id']==body.get('issue_id')),None)
                if not issue:raise ValueError('Review question not found.')
                if body.get('status') not in {'open','clarification_requested','resolved'}:raise ValueError('Choose a review status.')
                issue.update(status=body['status'],resolution=text(body.get('resolution',''),2000,body['status']=='resolved'),updated_by=current_user.get_id())
            else:raise ValueError('Choose a valid review action.')
        except (ValueError,TypeError):return jsonify(error='Check the fields, source references, dates, amounts and required verification.'),400
        version+=1
        db.execute('INSERT INTO child_support_review VALUES(?,?,?) ON CONFLICT(case_id) DO UPDATE SET version=excluded.version,data=excluded.data',(case_id,version,encrypt(review)))
        audit(db,'review_'+action,case_id)
        return jsonify(version=version)


@bp.post('/api/casework/cases/<case_id>/packets')
@protected
def prepare_packet(case_id):
    with store().connect() as db:
        db.execute('BEGIN IMMEDIATE');case=get_case(db,case_id)
        if not case:return jsonify(error='Case not found.'),404
        if case['archived'] or not current_app.config['CASEWORK_REAL_DATA_ENABLED']:return jsonify(error='Packet preparation is not enabled.'),409
        snap=snapshot(db,case_id)
        if not snap['review']['events']:return jsonify(error='Add source-linked timeline entries first.'),400
        if any(not f['verified'] for f in snap['files']):return jsonify(error='Verify document OCR before preparing the packet.'),409
        if db.execute('SELECT COUNT(*) FROM child_support_packets WHERE case_id=?',(case_id,)).fetchone()[0]>=50:return jsonify(error='Maximum 50 packets per case.'),409
        packet_id=str(uuid.uuid4());payload=dict(title=decrypt(case['data'])['title'],prepared=time.time(),snapshot=snap,fingerprint=digest(snap),integration='not_connected',notice='Working review packet. Not an official MACSS record, balance calculation, court order or legal opinion. AI drafts remain unverified until separately reviewed.')
        db.execute('INSERT INTO child_support_packets(id,case_id,creator,data) VALUES(?,?,?,?)',(packet_id,case_id,current_user.get_id(),encrypt(payload)));audit(db,'packet_prepared',packet_id)
        return jsonify(id=packet_id,status='awaiting_review'),201


@bp.route('/api/casework/cases/<case_id>/packets/<packet_id>',methods=['GET','POST'])
@protected
def packet_detail(case_id,packet_id):
    with store().connect() as db:
        if request.method=='POST':db.execute('BEGIN IMMEDIATE')
        case=get_case(db,case_id)
        if not case:return jsonify(error='Packet not found.'),404
        packet=db.execute('SELECT * FROM child_support_packets WHERE id=? AND case_id=?',(packet_id,case_id)).fetchone()
        if not packet:return jsonify(error='Packet not found.'),404
        payload=decrypt(packet['data']);current=digest(snapshot(db,case_id))==payload['fingerprint']
        if request.method=='GET':
            audit(db,'packet_viewed',packet_id)
            return jsonify(id=packet_id,creator=packet['creator'],reviewer=packet['reviewer'],reviewed=packet['reviewed'],status='reviewed' if packet['reviewer'] else 'awaiting_review',matches_current_case=current,**payload)
        if case['archived'] or not current_app.config['CASEWORK_REAL_DATA_ENABLED']:return jsonify(error='Packet review is not enabled.'),409
        if packet['creator']==current_user.get_id():return jsonify(error='A different authorized staff member must review this packet.'),403
        if json_body().get('evidence_verified') is not True or json_body().get('client_explanations_reviewed') is not True:return jsonify(error='Check the evidence and client explanations before approval.'),400
        if not current:return jsonify(error='The case changed. Prepare a new packet for review.'),409
        if packet['reviewer']:return jsonify(error='This packet has already been reviewed.'),409
        db.execute('UPDATE child_support_packets SET reviewer=?,reviewed=? WHERE id=?',(current_user.get_id(),time.time(),packet_id));audit(db,'packet_reviewed',packet_id)
        return jsonify(status='reviewed')
