"""Studio observability and review: real local traces, explicit simulated steps.

SQLite lives on STUDIO_DB_PATH; mount persistent storage for deployment durability.
Roles are assigned by the operator using STUDIO_ROLES_JSON (OAuth IDs, not names).
No endpoint grants roles. Live proposals call OpenAI only after explicit consent and independent approval.
"""
import asyncio
import hashlib
import hmac
import ipaddress
import json
import os
import secrets
import sqlite3
import time
import uuid
from functools import wraps
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

from flask import Blueprint, current_app, jsonify, request, session, send_file
from flask_login import current_user, login_required
from .architecture import dredge_run_pipeline
from .studio_provider import configured_model, run_live_pipeline, MAX_INPUT_CHARS, MAX_OUTPUT_TOKENS

studio_bp = Blueprint('studio', __name__)
ROLES = {'viewer', 'operator', 'reviewer', 'admin'}
STATIC = Path(__file__).parent / 'static'


class StudioStore:
    def __init__(self, path):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, status TEXT NOT NULL,
                    payload TEXT NOT NULL, result TEXT, created REAL NOT NULL,
                    started REAL, ended REAL, reviewer TEXT, review_note TEXT);
                CREATE TABLE IF NOT EXISTS trace_events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
                    recorded REAL NOT NULL, event TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS trace_run_idx ON trace_events(run_id,seq);
                CREATE TABLE IF NOT EXISTS audit (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, recorded REAL NOT NULL,
                    actor TEXT NOT NULL, action TEXT NOT NULL, run_id TEXT,
                    detail TEXT NOT NULL, previous_hash TEXT NOT NULL, hash TEXT NOT NULL);
                CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit
                    BEGIN SELECT RAISE(ABORT,'Audit entries are append-only'); END;
                CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit
                    BEGIN SELECT RAISE(ABORT,'Audit entries are append-only'); END;
            ''')
        os.chmod(self.path, 0o600)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA journal_mode=WAL')
        try:
            with db:
                yield db
        finally:
            db.close()

    def audit(self, db, actor, action, run_id=None, detail=None):
        recorded = time.time()
        detail_json = json.dumps(detail or {}, sort_keys=True, separators=(',', ':'))
        last = db.execute('SELECT hash FROM audit ORDER BY seq DESC LIMIT 1').fetchone()
        previous = last['hash'] if last else ''
        material = json.dumps([recorded, actor, action, run_id, detail_json, previous], separators=(',', ':'))
        digest = hashlib.sha256(material.encode()).hexdigest()
        db.execute('INSERT INTO audit(recorded,actor,action,run_id,detail,previous_hash,hash) VALUES(?,?,?,?,?,?,?)',
                   (recorded, actor, action, run_id, detail_json, previous, digest))

    def trace(self, run_id, event):
        with self.connect() as db:
            db.execute('INSERT INTO trace_events(run_id,recorded,event) VALUES(?,?,?)',
                       (run_id, time.time(), json.dumps(event)))


def store():
    return current_app.extensions['studio_store']


def role():
    mapping = current_app.config.get('STUDIO_ROLES', {})
    assigned = mapping.get(current_user.get_id(), 'viewer')
    return assigned if assigned in ROLES else 'viewer'


def allowed(*roles):
    def decorate(func):
        @wraps(func)
        @login_required
        def wrapped(*args, **kwargs):
            if role() not in roles:
                return jsonify(error='Your workspace role does not allow this action.'), 403
            return func(*args, **kwargs)
        return wrapped
    return decorate


def readable(run):
    return run and (run['owner'] == current_user.get_id() or role() in {'reviewer', 'admin'})


def serialize_run(run, detail=False):
    item = {k: run[k] for k in ('id', 'status', 'created', 'started', 'ended', 'reviewer', 'review_note')}
    payload = json.loads(run['payload'])
    result = json.loads(run['result']) if run['result'] else {}
    # Pre-upgrade proposals retain the local/demo behavior that was approved.
    mode = payload.get('execution_mode', 'demo')
    calls = result.get('provider_calls', 0)
    if mode == 'live' and not calls:
        with store().connect() as db:
            calls = int(bool(db.execute("SELECT 1 FROM trace_events WHERE run_id=? AND json_extract(event,'$.id')='astra_response' LIMIT 1", (run['id'],)).fetchone()))
    item.update(query=payload['query'], pipeline_type=payload['pipeline_type'], mode=mode,
                model=payload.get('model'), provider_calls=calls,
                token_usage=result.get('token_usage'), usage=result.get('usage'),
                cost_usd=result.get('cost_usd'), cost_status=result.get('cost_status', 'not_metered'),
                data_scope=payload.get('data_scope'), consent=payload.get('consent', False),
                max_output_tokens=payload.get('max_output_tokens'),
                can_execute=run['owner'] == current_user.get_id() and role() in {'operator','admin'})
    if detail:
        item['evidence'] = payload['evidence']
        item['result'] = json.loads(run['result']) if run['result'] else None
        with store().connect() as db:
            events = db.execute('SELECT seq,recorded,event FROM trace_events WHERE run_id=? ORDER BY seq', (run['id'],)).fetchall()
        item['events'] = [dict(seq=e['seq'], recorded=e['recorded'], **json.loads(e['event'])) for e in events]
        latest = {}
        for event in item['events']:
            latest[event['id']] = event
        item['nodes'] = list(latest.values())
    return item


def valid_url(value):
    """Allow source links, never fetch submitted URLs or accept credentials."""
    if not isinstance(value, str):
        return False
    try:
        parsed = urlsplit(value)
        if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password:
            return False
        host = parsed.hostname.lower()
        if parsed.port is not None and parsed.port != 443:
            return False
        if host == 'localhost' or '.' not in host or host.endswith(('.local', '.internal')):
            return False
        try:
            if not ipaddress.ip_address(host).is_global:
                return False
        except ValueError:
            pass
        return len(value) <= 2048
    except ValueError:
        return False


def validate_payload(data):
    if not isinstance(data, dict):
        raise ValueError('Provide a JSON object.')
    query = data.get('query', '')
    pipeline_type = data.get('pipeline_type', 'standard')
    evidence = data.get('evidence', [])
    if not isinstance(query, str) or not query.strip() or len(query) > 4000:
        raise ValueError('Enter a question between 1 and 4,000 characters.')
    if not isinstance(pipeline_type, str) or pipeline_type not in {'standard', 'ios_swift'}:
        raise ValueError('Choose a supported pipeline.')
    if not isinstance(evidence, list) or len(evidence) > 10:
        raise ValueError('Provide no more than 10 source records.')
    checked = []
    for index, source in enumerate(evidence):
        if not isinstance(source, dict) or not valid_url(source.get('url', '')):
            raise ValueError('Each source needs a public HTTPS URL without embedded credentials.')
        title, excerpt = source.get('title', ''), source.get('excerpt', '')
        if not isinstance(title, str) or not title.strip() or len(title) > 300 or not isinstance(excerpt, str) or len(excerpt) > 4000:
            raise ValueError('Sources need a title (up to 300 characters) and excerpt (up to 4,000).')
        checked.append(dict(id=f'source-{index+1}', title=title.strip(), url=source['url'], excerpt=excerpt,
                            provenance='user_supplied', verification='unverified', attached_at=time.time()))
    mode = data.get('execution_mode', 'live')
    if not isinstance(mode, str) or mode not in {'live', 'demo'}:
        raise ValueError('Choose live Astra or demonstration mode.')
    payload = dict(query=query.strip(), pipeline_type=pipeline_type, evidence=checked, execution_mode=mode)
    if mode == 'live':
        if pipeline_type != 'standard':
            raise ValueError('Live inference uses the Standard pipeline. iOS / Swift is demonstration-only.')
        if data.get('consent') is not True or data.get('public_data_confirmed') is not True:
            raise ValueError('Confirm sending the question and all source records to OpenAI, and that they contain only public or fictional information.')
        if data.get('case_id') or data.get('file_ids') or data.get('include_client_explanations'):
            raise ValueError('Private case records must use the agency-gated Casework workflow.')
        if len(query) + sum(len(source['title']) + len(source['url']) + len(source['excerpt']) for source in checked) > MAX_INPUT_CHARS:
            raise ValueError('Live questions and evidence must total at most 12,000 characters.')
        payload.update(model=configured_model(), max_output_tokens=MAX_OUTPUT_TOKENS,
                       consent=True, data_scope='public_or_fictional')
    return payload


@studio_bp.before_request
def protect_mutations():
    if request.path.startswith('/api/studio/') and request.method == 'POST':
        if not current_user.is_authenticated:
            return jsonify(error='Your session has expired. Sign in again.', code='session_expired'), 401
        supplied = request.headers.get('X-CSRF-Token', '')
        expected = session.get('studio_csrf', '')
        if not expected or not hmac.compare_digest(expected, supplied):
            return jsonify(error='Refresh the workspace before submitting this action.', code='csrf_failed'), 403


@studio_bp.route('/architecture')
def public_architecture():
    from flask import render_template_string
    return render_template_string((STATIC / 'studio_public.html').read_text(), page='architecture', error=None,
                                  github_enabled=False, google_enabled=False)


@studio_bp.route('/preview')
def public_preview():
    return send_file(STATIC / 'studio_workspace.html')


@studio_bp.route('/api/studio/session')
@login_required
def session_info():
    token = session.setdefault('studio_csrf', secrets.token_urlsafe(32))
    return jsonify(name=current_user.name, role=role(), csrf_token=token,
                   can_run=role() in {'operator','admin'}, can_review=role() in {'reviewer','admin'},
                   live_model=configured_model(), live_configured=bool(current_app.config.get('OPENAI_API_KEY')),
                   max_output_tokens=MAX_OUTPUT_TOKENS,
                   idle_timeout_seconds=current_app.config['STUDIO_IDLE_SECONDS'],
                   expires_at=session.get('studio_started', time.time()) + current_app.config['STUDIO_SESSION_SECONDS'])


@studio_bp.route('/api/studio/preview')
def preview_data():
    # Public fixture contains no account information and creates no persisted run.
    return jsonify(id='public-demo', query='How should staff clarify a difference between two fictional child-support records?',
                   status='completed', mode='simulated',
                   nodes=[dict(id='frame', dependencies=[], status='completed', mode='simulated', duration_ms=None),
                          dict(id='sources', dependencies=['frame'], status='completed', mode='simulated', duration_ms=None),
                          dict(id='alternatives', dependencies=['frame'], status='completed', mode='simulated', duration_ms=None),
                          dict(id='review', dependencies=['sources','alternatives'], status='completed', mode='simulated', duration_ms=None)],
                   evidence=[dict(id='demo-source', title='Fictional child-support training records',
                                  url='/static/child_support_training.txt',
                                  excerpt='FICTIONAL TRAINING DATA: January notice lists $900; February record lists $950. The client says the amount changed the following month. Check the periods before treating this as a discrepancy.',
                                  provenance='demonstration', verification='unverified')],
                   result={'summary':'Fictional demonstration: compare dates and amounts, preserve the client explanation, ask which period each record covers and prepare the evidence for staff review. No official balance, enforcement decision or MACSS change is made.'},
                   provider_calls=0, token_usage=None, cost_usd=None, cost_status='not_metered')


@studio_bp.route('/api/studio/runs', methods=['GET', 'POST'])
@login_required
def runs():
    if request.method == 'POST':
        if role() not in {'operator', 'admin'}:
            return jsonify(error='An operator role is required to propose a run.'), 403
        from .billing import require_subscription
        payment_guard = require_subscription()
        if payment_guard is not None:
            return payment_guard
        try:
            payload = validate_payload(request.get_json(silent=True))
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        if payload['execution_mode'] == 'live' and not current_app.config.get('OPENAI_API_KEY'):
            return jsonify(error='OpenAI is not configured. No provider call was made. Choose demonstration mode or ask the operator to configure the existing credential.'), 503
        run_id = str(uuid.uuid4())
        with store().connect() as db:
            db.execute('INSERT INTO runs(id,owner,status,payload,created) VALUES(?,?,?,?,?)',
                       (run_id, current_user.get_id(), 'pending_approval', json.dumps(payload), time.time()))
            store().audit(db, current_user.get_id(), 'run_proposed', run_id, {'pipeline_type': payload['pipeline_type'], 'execution_mode': payload['execution_mode'], 'model': payload.get('model'), 'data_scope': payload.get('data_scope'), 'consent': payload.get('consent', False)})
            run = db.execute('SELECT * FROM runs WHERE id=?', (run_id,)).fetchone()
        return jsonify(serialize_run(run, True)), 201
    with store().connect() as db:
        if role() in {'reviewer','admin'}:
            rows = db.execute('SELECT * FROM runs ORDER BY created DESC LIMIT 100').fetchall()
        else:
            rows = db.execute('SELECT * FROM runs WHERE owner=? ORDER BY created DESC LIMIT 100', (current_user.get_id(),)).fetchall()
    return jsonify(runs=[serialize_run(row) for row in rows])


@studio_bp.route('/api/studio/runs/<run_id>')
@login_required
def run_detail(run_id):
    with store().connect() as db:
        run = db.execute('SELECT * FROM runs WHERE id=?', (run_id,)).fetchone()
    if not readable(run):
        return jsonify(error='Run not found.'), 404
    return jsonify(serialize_run(run, True))


@studio_bp.route('/api/studio/runs/<run_id>/review', methods=['POST'])
@allowed('reviewer', 'admin')
def review_run(run_id):
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict) or data.get('decision') not in {'approve','reject'}:
        return jsonify(error='Choose approve or reject.'), 400
    note = data.get('note', '')
    if not isinstance(note, str) or not note.strip() or len(note) > 2000:
        return jsonify(error='Provide a review reason between 1 and 2,000 characters.'), 400
    with store().connect() as db:
        db.execute('BEGIN IMMEDIATE')
        run = db.execute('SELECT * FROM runs WHERE id=?', (run_id,)).fetchone()
        if not readable(run):
            return jsonify(error='Run not found.'), 404
        if run['owner'] == current_user.get_id():
            return jsonify(error='A separate reviewer must decide this request.'), 403
        if run['status'] != 'pending_approval':
            return jsonify(error='This request has already been decided.'), 409
        state = 'approved' if data['decision'] == 'approve' else 'rejected'
        db.execute('UPDATE runs SET status=?,reviewer=?,review_note=? WHERE id=?', (state, current_user.get_id(), note.strip(), run_id))
        store().audit(db, current_user.get_id(), 'run_'+state, run_id, {'reason': note.strip()})
    return jsonify(status=state)


@studio_bp.route('/api/studio/runs/<run_id>/execute', methods=['POST'])
@allowed('operator','admin')
def execute_run(run_id):
    from .billing import require_subscription
    payment_guard = require_subscription()
    if payment_guard is not None:
        return payment_guard
    with store().connect() as db:
        db.execute('BEGIN IMMEDIATE')
        run = db.execute('SELECT * FROM runs WHERE id=?', (run_id,)).fetchone()
        if not run or run['owner'] != current_user.get_id():
            return jsonify(error='Run not found.'), 404
        if run['status'] != 'approved':
            return jsonify(error='This run needs approval or has already executed.'), 409
        payload = json.loads(run['payload'])
        if payload.get('execution_mode') == 'live':
            if not current_app.config.get('OPENAI_API_KEY'):
                return jsonify(error='OpenAI is not configured. No provider call was made.'), 503
            if payload.get('model') != configured_model():
                return jsonify(error='The configured model changed after approval. Create a new proposal.'), 409
            if payload.get('consent') is not True or payload.get('data_scope') != 'public_or_fictional':
                return jsonify(error='This proposal lacks consent for public or fictional data. Create a new proposal.'), 409
            attempts = db.execute("SELECT COUNT(*) FROM runs WHERE owner=? AND started>? AND json_extract(payload,'$.execution_mode')='live'", (current_user.get_id(), time.time()-86400)).fetchone()[0]
            if attempts >= 20:
                return jsonify(error='Studio limit of 20 live attempts per account per rolling 24 hours reached.'), 429
        db.execute('UPDATE runs SET status=?,started=? WHERE id=?', ('running', time.time(), run_id))
        store().audit(db, current_user.get_id(), 'run_started', run_id)
    trace_store = store()
    try:
        callback = lambda event: trace_store.trace(run_id, event)
        if payload.get('execution_mode') == 'live':
            result = asyncio.run(run_live_pipeline(payload, run_id, callback))
        else:
            result = asyncio.run(dredge_run_pipeline({'query': payload['query'], 'mode':'standard'},
                         pipeline_type=payload['pipeline_type'], pipeline_id=run_id,
                         trace_callback=callback))
        state = 'failed' if result.get('status') == 'failed' else 'completed'
    except Exception:
        # Never log provider bodies, credentials, questions, or supplied evidence.
        current_app.logger.error('Studio pipeline failed: %s', run_id)
        result = {'error': 'The pipeline failed. Check this run’s recorded trace. No automatic retry was made.'}
        state = 'failed'
    with trace_store.connect() as db:
        db.execute('UPDATE runs SET status=?,result=?,ended=? WHERE id=?', (state,json.dumps(result),time.time(),run_id))
        trace_store.audit(db, current_user.get_id(), 'run_'+state, run_id)
        run = db.execute('SELECT * FROM runs WHERE id=?', (run_id,)).fetchone()
    return jsonify(serialize_run(run, True)), (200 if state == 'completed' else 500)


@studio_bp.route('/api/studio/status')
@login_required
def status_panels():
    with store().connect() as db:
        latest = db.execute('SELECT ended FROM runs WHERE owner=? AND status=? ORDER BY ended DESC LIMIT 1', (current_user.get_id(),'completed')).fetchone()
    from .casework import member, cipher, MODEL
    with store().connect() as db:
        live_success = db.execute("SELECT ended FROM runs WHERE owner=? AND status='completed' AND json_extract(payload,'$.execution_mode')='live' AND json_extract(payload,'$.model')=? ORDER BY ended DESC LIMIT 1", (current_user.get_id(), configured_model())).fetchone()
    membership = member()
    configured = bool(current_app.config.get('OPENAI_API_KEY'))
    encrypted = bool(cipher())
    provider_ready = configured and encrypted and bool(membership)
    analysis_ready = provider_ready and current_app.config.get('CASEWORK_AI_DATA_ENABLED', False)
    observations = {}
    if membership:
        with store().connect() as db:
            for kind in ('analysis','research'):
                result = db.execute('SELECT MAX(created) AS observed FROM casework_outputs WHERE agency=? AND kind=? AND data IS NOT NULL', (membership['agency'],kind)).fetchone()
                observations[kind] = result['observed']
    def provider_item(name, kind, ready, description):
        observed = observations.get(kind)
        status = 'not_connected' if not configured else 'agency_access_required' if not membership else 'encryption_required' if not encrypted else 'agency_approval_required' if not ready else 'successful_request_recorded' if observed else 'configured_not_verified'
        return dict(name=name, mode='live', status=status, label='Disconnected' if not configured else 'Recorded success' if ready and observed else 'Configured · Unverified' if ready else 'Setup required', description=description, last_observed=observed, href='/casework#casework' if membership else None)
    live_observed = live_success['ended'] if live_success else None
    return jsonify(observed_at=time.time(), models=[
        provider_item('Astra · '+MODEL, 'analysis', analysis_ready, 'Draft analysis of selected case evidence. No web tools. Requires agency data-transfer approval and per-request consent.'),
        dict(name='Studio Astra · '+configured_model(), mode='live', status='not_connected' if not configured else 'successful_request_recorded' if live_observed else 'configured_not_verified', label='Disconnected' if not configured else 'Recorded success' if live_observed else 'Configured · Unverified', last_observed=live_observed, description='Live Responses API for approved public or fictional questions and supplied evidence. Consent and a separate reviewer are required. No web retrieval. Up to 1,200 output tokens; no automatic retries.'),
        dict(name='Demonstration model catalog', mode='simulated', status='demonstration_catalog', description='Quasimoto, String Theory and scripted Deep / Google adapters are demonstrations, not connected provider models.', href='/advanced/toolkit')], tools=[
        provider_item('OpenAI web research', 'research', provider_ready, 'Public questions only, with source citations. Case attachments are excluded. Up to two web tool calls per request.'),
        dict(name='Local DAG engine', mode='local', status='last_execution_completed' if latest else 'not_yet_observed', last_observed=latest['ended'] if latest else None, description='Actual local pipeline execution after independent human approval.'),
        dict(name='Trace and audit storage', mode='local', status='read_write_available', description='Recorded node events and append-only audit entries. Configured storage path does not verify backup recovery.'),
        dict(name='Encrypted case files', mode='local', status='ready' if encrypted and membership and current_app.config.get('CASEWORK_REAL_DATA_ENABLED') else 'agency_approval_required' if encrypted and membership else 'setup_required', description='Agency-scoped TXT, PDF and photo storage with local OCR and staff text verification. Real client uploads remain gated until agency approval.', href='/casework#casework' if membership else None),
        dict(name='Agency Pages', mode='local', status='ready' if encrypted and membership else 'setup_required', description='Shared guidance with encrypted version history and conflict protection.', href='/casework#pages' if membership else None),
        dict(name='Source inspector', mode='local', status='user_supplied_sources_only', description='Live Studio answers reference supplied evidence IDs; reference checks do not verify claims. Web research citations are displayed in Casework.')],
        persistence='configured_path' if current_app.config.get('STUDIO_EXPLICIT_DB') else 'instance_disk_requires_persistent_volume',
        scope='Configuration and stored observations only. Refreshing this panel makes no external probes or paid requests. Recorded success is historical, not a current availability guarantee. Studio limit: 20 live attempts per account per rolling 24 hours and 1,200 output tokens. Casework agency limit: 20 attempts per rolling 24 hours and 2,400 output tokens.')



@studio_bp.route('/api/studio/report')
@login_required
def usage_report():
    with store().connect() as db:
        rows = db.execute('SELECT * FROM runs WHERE owner=?', (current_user.get_id(),)).fetchall()
    completed = [row for row in rows if row['status']=='completed']
    failed = [row for row in rows if row['status']=='failed']
    measured = [row['ended']-row['started'] for row in completed if row['started'] and row['ended']]
    terminal = len(completed)+len(failed)
    summaries = [serialize_run(row) for row in rows]
    calls = sum(row['provider_calls'] for row in summaries)
    metered = [row for row in summaries if row['provider_calls'] and row['token_usage'] is not None]
    return jsonify(scope='Your stored Studio runs, all time; Casework has separate usage limits', proposed_runs=len(rows),
                   completed_runs=len(completed), failed_runs=len(failed),
                   success_rate=(len(completed)/terminal if terminal else None),
                   average_duration_ms=(round(sum(measured)/len(measured)*1000,3) if measured else None),
                   provider_calls=calls, token_usage=sum(row['token_usage'] for row in metered) if metered else None,
                   usage_status='partial' if calls > len(metered) else 'provider_reported' if metered else 'unavailable',
                   calls_without_usage=calls-len(metered), cost_usd=None, cost_status='billing_unavailable',
                   cost_note='Tokens come from provider responses, including incomplete responses when available. Missing usage is not zero usage. Provider billing and infrastructure cost are not integrated. Timeouts may still incur charges.',
                   interrupted_runs=sum(row['status']=='running' for row in rows))


@studio_bp.route('/api/studio/audit')
@allowed('reviewer','admin')
def audit_events():
    with store().connect() as db:
        rows = db.execute('SELECT * FROM audit ORDER BY seq DESC LIMIT 200').fetchall()
        chain = db.execute('SELECT * FROM audit ORDER BY seq').fetchall()
    previous = ''
    verified = True
    for row in chain:
        material = json.dumps([row['recorded'],row['actor'],row['action'],row['run_id'],row['detail'],previous],separators=(',', ':'))
        if row['previous_hash'] != previous or row['hash'] != hashlib.sha256(material.encode()).hexdigest():
            verified = False
            break
        previous = row['hash']
    return jsonify(events=[dict(row) for row in rows], scope='Latest 200 instance events',
                   chain_verified=verified,
                   integrity='Hash-linked, append-only through this application; not a signed external audit service.')


def register_studio(app):
    try:
        roles = json.loads(os.environ.get('STUDIO_ROLES_JSON', '{}'))
        if not isinstance(roles, dict) or any(not isinstance(k,str) or v not in ROLES for k,v in roles.items()):
            raise ValueError('Invalid role mapping')
    except (ValueError, TypeError):
        raise RuntimeError('STUDIO_ROLES_JSON must map OAuth user IDs to viewer/operator/reviewer/admin.')
    app.config.setdefault('STUDIO_ROLES', roles)
    app.config.setdefault('OPENAI_MODEL', os.environ.get('OPENAI_MODEL', 'gpt-6-astra'))
    app.config.setdefault('STUDIO_IDLE_SECONDS', 1800)
    app.config.setdefault('STUDIO_SESSION_SECONDS', 43200)
    path = os.environ.get('STUDIO_DB_PATH', str(Path(app.instance_path) / 'studio.sqlite3'))
    app.config.setdefault('STUDIO_EXPLICIT_DB', bool(os.environ.get('STUDIO_DB_PATH')))
    app.extensions['studio_store'] = StudioStore(path)
    app.register_blueprint(studio_bp)
