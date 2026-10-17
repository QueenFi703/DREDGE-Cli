"""Optional Thresh account linking and entitlement authority. © 2026 Sophia Cole.

Disabled unless THRESH_BRIDGE_KEY is configured. Never match accounts by email,
create charges, assign Studio roles, or expose investigation data.
"""
import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import time
from urllib.parse import urlencode, urlsplit
from flask import Blueprint, current_app, jsonify, redirect, render_template_string, request, session, url_for
from flask_login import current_user

SITE_ID = 'appgprj_6ad3aae39c5481919a40c791221631ac'
SITE_ORIGIN = 'https://thresh-agent.queenfi24.chatgpt.site'
PAGE = '''<!doctype html><meta charset="utf-8"><title>Link Thresh · DREDGE</title>
<h1>{{ title }}</h1><p>{{ message }}</p>
{% if challenge %}<p>Signed in as {{ account }}. Link subscription status only. This grants no access to Studio runs, casework, Siri history, or GitHub repositories.</p>
<form method="post"><input type="hidden" name="csrf" value="{{ csrf }}"><input type="hidden" name="challenge" value="{{ challenge }}"><button>Link this DREDGE account to Thresh</button></form>{% endif %}
{% if linked %}<form method="post" action="/thresh/unlink"><input type="hidden" name="csrf" value="{{ csrf }}"><button>Revoke Thresh account link</button></form>{% endif %}
<a href="{{ site }}">Return to Thresh</a>'''


def register_thresh_bridge(app, *, store=None, subscription_check=None):
    key = os.environ.get('THRESH_BRIDGE_KEY', '')
    if not key:
        return False
    if len(key) < 32 or not key.isascii():
        raise RuntimeError('THRESH_BRIDGE_KEY must contain at least 32 characters.')
    if store is None:
        store = app.extensions['studio_store']
    if subscription_check is None:
        from .billing import configured, has_subscription, mode
        subscription_check = lambda owner: mode() == 'live' and configured() and has_subscription(owner)
    with store.connect() as db:
        db.executescript('''
        CREATE TABLE IF NOT EXISTS thresh_links(site_id TEXT,subject TEXT,owner TEXT,revoked INTEGER NOT NULL DEFAULT 0,
          PRIMARY KEY(site_id,subject),UNIQUE(site_id,owner));
        CREATE TABLE IF NOT EXISTS thresh_link_challenges(hash TEXT PRIMARY KEY,site_id TEXT,subject TEXT,expires INTEGER,
          UNIQUE(site_id,subject));
        CREATE TABLE IF NOT EXISTS thresh_repair_requests(
          request_id TEXT PRIMARY KEY,site_id TEXT NOT NULL,subject TEXT NOT NULL,owner TEXT NOT NULL,
          repository TEXT NOT NULL,base_sha TEXT NOT NULL,summary TEXT NOT NULL,status TEXT NOT NULL,
          created INTEGER NOT NULL,request_hash TEXT NOT NULL UNIQUE);
        CREATE TABLE IF NOT EXISTS thresh_oauth_clients(
          client_id TEXT PRIMARY KEY,redirect_uris TEXT NOT NULL,created INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS thresh_oauth_codes(
          hash TEXT PRIMARY KEY,client_id TEXT NOT NULL,redirect_uri TEXT NOT NULL,owner TEXT NOT NULL,
          challenge TEXT NOT NULL,scope TEXT NOT NULL,expires INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS thresh_oauth_tokens(
          hash TEXT PRIMARY KEY,owner TEXT NOT NULL,scope TEXT NOT NULL,expires INTEGER NOT NULL,revoked INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS thresh_reviewer_grants(
          owner TEXT PRIMARY KEY,expires INTEGER NOT NULL);
        ''')
    bp = Blueprint('thresh_bridge', __name__)

    @bp.after_request
    def private_response(response):
        response.headers.update({'Cache-Control': 'no-store', 'Referrer-Policy': 'no-referrer',
                                 'X-Content-Type-Options': 'nosniff', 'Content-Security-Policy': "default-src 'none'; form-action 'self'; frame-ancestors 'none'; base-uri 'none'"})
        return response

    def identity():
        if request.content_length is None or request.content_length > 2048:
            return None
        authorization = request.headers.get('Authorization', '')
        if not authorization.isascii() or not hmac.compare_digest(authorization, 'Bearer '+key):
            return None
        body = request.get_json(silent=True)
        if (not isinstance(body, dict) or set(body) != {'site_id', 'subject', 'entitlement'}
                or body['site_id'] != SITE_ID or body['entitlement'] != 'thresh:use'
                or not isinstance(body['subject'], str) or not 1 <= len(body['subject']) <= 256
                or '\x00' in body['subject']):
            return None
        return body['subject']

    def repair_request():
        if request.content_length is None or request.content_length > 16384:
            return None
        authorization = request.headers.get('Authorization', '')
        if not authorization.isascii() or not hmac.compare_digest(authorization, 'Bearer '+key):
            return None
        body = request.get_json(silent=True)
        expected = {'site_id', 'subject', 'entitlement', 'repository', 'base_sha', 'approval', 'findings'}
        if (not isinstance(body, dict) or set(body) != expected or body.get('site_id') != SITE_ID
                or body.get('entitlement') != 'thresh:use' or body.get('approval') != 'SUBMIT_REPAIR_REQUEST'
                or not isinstance(body.get('subject'), str) or not 1 <= len(body['subject']) <= 256
                or '\x00' in body['subject'] or not isinstance(body.get('repository'), str)
                or not re.fullmatch(r'[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?/[A-Za-z0-9][A-Za-z0-9_.-]{0,99}', body['repository'])
                or '..' in body['repository']
                or not isinstance(body.get('base_sha'), str) or not re.fullmatch(r'[0-9a-fA-F]{40}', body['base_sha'])
                or not isinstance(body.get('findings'), list) or not 1 <= len(body['findings']) <= 40):
            return None
        allowed_checks = {'windows-runner-compatibility', 'explicit-permissions'}
        for finding in body['findings']:
            if (not isinstance(finding, dict) or set(finding) != {'path', 'check'}
                    or finding.get('check') not in allowed_checks
                    or not isinstance(finding.get('path'), str)
                    or not re.fullmatch(r'\.github/workflows/[A-Za-z0-9][A-Za-z0-9_.-]*\.ya?ml', finding['path'])):
                return None
        body['base_sha'] = body['base_sha'].lower()
        return body

    def csrf_ok():
        origin = request.headers.get('Origin')
        expected = current_app.config.get('BILLING_ORIGIN', '').rstrip('/')
        if origin != expected or not expected.startswith('https://'):
            return False
        saved = session.get('thresh_csrf', '')
        supplied = request.form.get('csrf', '')
        return bool(saved and supplied.isascii() and hmac.compare_digest(saved, supplied))

    def challenge_row(token, db):
        if not re.fullmatch(r'[A-Za-z0-9_-]{43}', token or ''):
            return None
        return db.execute('SELECT * FROM thresh_link_challenges WHERE hash=? AND expires>?',
                          (hashlib.sha256(token.encode()).hexdigest(), int(time.time()))).fetchone()

    def page(title, message, **values):
        return render_template_string(PAGE, title=title, message=message, site=SITE_ORIGIN, **values)

    def origin():
        value = current_app.config.get('BILLING_ORIGIN', '').rstrip('/')
        return value if value.startswith('https://') else ''

    def owner_entitled(owner):
        with store.connect() as db:
            grant = db.execute('SELECT expires FROM thresh_reviewer_grants WHERE owner=? AND expires>?',
                               (owner, int(time.time()))).fetchone()
        return bool(grant or subscription_check(owner))

    def mcp_output(data, is_error=False):
        return {'content': [{'type': 'text', 'text': json.dumps(data, separators=(',', ':'))}], 'isError': is_error}

    @bp.get('/.well-known/openai-apps-challenge')
    def openai_apps_challenge():
        token = os.environ.get('THRESH_OPENAI_DOMAIN_TOKEN', '')
        if not re.fullmatch(r'[A-Za-z0-9_-]{32,128}', token):
            return 'Domain verification is not configured.', 404
        return token, 200, {'Content-Type': 'text/plain; charset=utf-8'}

    @bp.get('/.well-known/oauth-protected-resource/mcp')
    def oauth_resource_metadata():
        root = origin()
        if not root:
            return jsonify(error='OAuth origin is unavailable.'), 503
        return jsonify(resource=root+'/mcp', authorization_servers=[root],
                       scopes_supported=['thresh:use'], bearer_methods_supported=['header'])

    def authorization_server_metadata():
        root = origin()
        if not root:
            return None
        return dict(issuer=root, authorization_endpoint=root+'/oauth/authorize',
                    token_endpoint=root+'/oauth/token', registration_endpoint=root+'/oauth/register',
                    response_types_supported=['code'], response_modes_supported=['query'],
                    grant_types_supported=['authorization_code'],
                    code_challenge_methods_supported=['S256'], token_endpoint_auth_methods_supported=['none'],
                    scopes_supported=['thresh:use'])

    @bp.get('/.well-known/oauth-authorization-server')
    def oauth_server_metadata():
        metadata = authorization_server_metadata()
        if metadata is None:
            return jsonify(error='OAuth origin is unavailable.'), 503
        return jsonify(metadata)

    @bp.post('/oauth/register')
    def oauth_register():
        if request.content_length is None or request.content_length > 4096:
            return jsonify(error='invalid_client_metadata'), 400
        body = request.get_json(silent=True)
        uris = body.get('redirect_uris') if isinstance(body, dict) else None
        if not isinstance(uris, list) or not 1 <= len(uris) <= 5:
            return jsonify(error='invalid_redirect_uri'), 400
        clean = []
        for value in uris:
            try:
                parsed = urlsplit(value)
            except (TypeError, ValueError):
                return jsonify(error='invalid_redirect_uri'), 400
            if parsed.scheme != 'https' or not parsed.netloc or parsed.username or parsed.password or parsed.fragment or len(value) > 1000:
                return jsonify(error='invalid_redirect_uri'), 400
            clean.append(value)
        client_id = 'thresh_'+secrets.token_urlsafe(24)
        with store.connect() as db:
            db.execute('INSERT INTO thresh_oauth_clients VALUES(?,?,?)',
                       (client_id, json.dumps(clean, separators=(',', ':')), int(time.time())))
        return jsonify(client_id=client_id, client_id_issued_at=int(time.time()), redirect_uris=clean,
                       token_endpoint_auth_method='none'), 201

    def pending_authorization():
        values = session.get('thresh_oauth_pending') if request.method == 'POST' or request.path.endswith('/resume') else request.args.to_dict(flat=True)
        if not isinstance(values, dict):
            return None
        required = {'client_id', 'redirect_uri', 'response_type', 'state', 'code_challenge', 'code_challenge_method'}
        if not required.issubset(values) or values.get('response_type') != 'code' or values.get('code_challenge_method') != 'S256':
            return None
        if values.get('scope', 'thresh:use') != 'thresh:use' or not re.fullmatch(r'[A-Za-z0-9_-]{43}', values.get('code_challenge', '')):
            return None
        with store.connect() as db:
            client = db.execute('SELECT redirect_uris FROM thresh_oauth_clients WHERE client_id=?', (values['client_id'],)).fetchone()
        if not client or values['redirect_uri'] not in json.loads(client['redirect_uris']) or len(values['state']) > 1000:
            return None
        return {key: values[key] for key in required} | {'scope': 'thresh:use'}

    @bp.route('/oauth/authorize', methods=['GET', 'POST'])
    @bp.get('/oauth/authorize/resume')
    def oauth_authorize():
        values = pending_authorization()
        if values is None:
            return jsonify(error='invalid_request'), 400
        if not current_user.is_authenticated:
            session['thresh_oauth_pending'] = values
            return redirect(url_for('auth.login'))
        session['thresh_oauth_pending'] = values
        csrf = session.setdefault('thresh_oauth_csrf', secrets.token_urlsafe(32))
        if request.method == 'GET':
            entitled = owner_entitled(current_user.get_id())
            return render_template_string('''<!doctype html><meta charset="utf-8"><title>Connect Thresh</title>
<h1>Connect Thresh</h1><p>Allow ChatGPT to review supplied workflows and save approved repair requests for maintainer review.</p>
<p>Workflow source is not retained. Thresh does not edit GitHub or initiate charges.</p>
{% if not entitled %}<p>This account has no active DREDGE subscription. Reviewers may enter the temporary access code supplied in the review instructions.</p>{% endif %}
<form method="post" action="/oauth/authorize"><input type="hidden" name="csrf" value="{{ csrf }}">
{% if not entitled %}<label>Reviewer access code <input name="reviewer_code" type="password" autocomplete="one-time-code"></label>{% endif %}
<button name="decision" value="allow">Allow Thresh</button><button name="decision" value="deny">Cancel</button></form>''', csrf=csrf, entitled=entitled)
        if request.form.get('csrf') != csrf or request.form.get('decision') not in {'allow', 'deny'}:
            return jsonify(error='invalid_request'), 400
        target = values['redirect_uri']
        if request.form['decision'] == 'deny':
            return redirect(target+'?'+urlencode({'error': 'access_denied', 'state': values['state']}))
        owner = current_user.get_id()
        if not owner_entitled(owner):
            expected = os.environ.get('THRESH_REVIEWER_CODE_HASH', '')
            supplied = request.form.get('reviewer_code', '')
            expiry = int(os.environ.get('THRESH_REVIEWER_EXPIRES', '0') or 0)
            actual = hashlib.sha256(supplied.encode()).hexdigest() if supplied else ''
            if not expected or expiry <= int(time.time()) or not hmac.compare_digest(actual, expected):
                return jsonify(error='An active subscription or valid reviewer code is required.'), 403
            with store.connect() as db:
                db.execute('INSERT INTO thresh_reviewer_grants VALUES(?,?) ON CONFLICT(owner) DO UPDATE SET expires=excluded.expires', (owner, expiry))
        code = secrets.token_urlsafe(32)
        with store.connect() as db:
            db.execute('DELETE FROM thresh_oauth_codes WHERE expires<=?', (int(time.time()),))
            db.execute('INSERT INTO thresh_oauth_codes VALUES(?,?,?,?,?,?,?)',
                       (hashlib.sha256(code.encode()).hexdigest(), values['client_id'], values['redirect_uri'], owner,
                        values['code_challenge'], values['scope'], int(time.time())+300))
        session.pop('thresh_oauth_pending', None)
        return redirect(target+'?'+urlencode({'code': code, 'state': values['state']}))

    @bp.post('/oauth/token')
    def oauth_token():
        if request.content_length is None or request.content_length > 4096:
            return jsonify(error='invalid_request'), 400
        code = request.form.get('code', '')
        verifier = request.form.get('code_verifier', '')
        client_id = request.form.get('client_id', '')
        redirect_uri = request.form.get('redirect_uri', '')
        if request.form.get('grant_type') != 'authorization_code' or not re.fullmatch(r'[A-Za-z0-9_-]{43}', code) or not 43 <= len(verifier) <= 128:
            return jsonify(error='invalid_grant'), 400
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b'=').decode()
        with store.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM thresh_oauth_codes WHERE hash=? AND expires>?',
                             (hashlib.sha256(code.encode()).hexdigest(), int(time.time()))).fetchone()
            if not row or row['client_id'] != client_id or row['redirect_uri'] != redirect_uri or not hmac.compare_digest(row['challenge'], challenge):
                return jsonify(error='invalid_grant'), 400
            db.execute('DELETE FROM thresh_oauth_codes WHERE hash=?', (row['hash'],))
            token = secrets.token_urlsafe(32)
            expires = int(time.time()) + 604800
            db.execute('INSERT INTO thresh_oauth_tokens VALUES(?,?,?,?,0)',
                       (hashlib.sha256(token.encode()).hexdigest(), row['owner'], row['scope'], expires))
        return jsonify(access_token=token, token_type='Bearer', expires_in=604800, scope=row['scope'])

    def mcp_owner():
        authorization = request.headers.get('Authorization', '')
        if not authorization.startswith('Bearer ') or not authorization.isascii():
            return None
        digest = hashlib.sha256(authorization[7:].encode()).hexdigest()
        with store.connect() as db:
            row = db.execute('SELECT owner FROM thresh_oauth_tokens WHERE hash=? AND revoked=0 AND expires>?',
                             (digest, int(time.time()))).fetchone()
        return row['owner'] if row and owner_entitled(row['owner']) else None

    @bp.post('/mcp')
    def mcp():
        body = request.get_json(silent=True)
        ident = body.get('id') if isinstance(body, dict) else None
        def reply(result): return jsonify(jsonrpc='2.0', id=ident, result=result)
        def failure(code, message): return jsonify(jsonrpc='2.0', id=ident, error={'code': code, 'message': message})
        if not isinstance(body, dict) or body.get('jsonrpc') != '2.0' or not isinstance(body.get('method'), str):
            return failure(-32600, 'Invalid request')
        if body['method'] == 'notifications/initialized' and 'id' not in body:
            return '', 202
        if body['method'] == 'initialize':
            return reply({'protocolVersion': body.get('params', {}).get('protocolVersion', '2025-06-18'),
                          'capabilities': {'tools': {'listChanged': False}}, 'serverInfo': {'name': 'thresh-coleworld', 'version': '1.2.0'}})
        schema = {'type': 'object', 'properties': {'workflows': {'type': 'array', 'minItems': 1, 'maxItems': 20,
                  'items': {'type': 'object', 'properties': {'path': {'type': 'string'}, 'content': {'type': 'string'}},
                  'required': ['path', 'content'], 'additionalProperties': False}}}, 'required': ['workflows'], 'additionalProperties': False}
        submit_schema = json.loads(json.dumps(schema)); submit_schema['properties'].update({
            'repository': {'type': 'string'}, 'base_sha': {'type': 'string', 'minLength': 40, 'maxLength': 40},
            'approval': {'type': 'string', 'enum': ['SUBMIT_REPAIR_REQUEST']}}); submit_schema['required'] += ['repository', 'base_sha', 'approval']
        definitions = [
          {'name':'thresh_review_workflows','title':'Review GitHub workflows','description':'Review supplied workflow text for Windows runner compatibility and missing explicit permissions. Heuristic review only; never executes or modifies workflows.','inputSchema':schema,'annotations':{'readOnlyHint':True,'destructiveHint':False,'idempotentHint':True,'openWorldHint':False}},
          {'name':'thresh_plan_repairs','title':'Plan workflow repairs','description':'Create a bounded human-reviewed repair and verification plan from supplied workflows. Does not commit, deploy, or claim CI success.','inputSchema':schema,'annotations':{'readOnlyHint':True,'destructiveHint':False,'idempotentHint':True,'openWorldHint':False}},
          {'name':'thresh_submit_repair_request','title':'Submit a workflow repair request','description':'Write an account-bound, idempotent repair request for maintainer review after explicit approval. Never stores workflow source or changes GitHub.','inputSchema':submit_schema,'annotations':{'readOnlyHint':False,'destructiveHint':False,'idempotentHint':True,'openWorldHint':True}}]
        if body['method'] == 'tools/list': return reply({'tools': definitions})
        if body['method'] == 'ping': return reply({})
        if body['method'] != 'tools/call': return failure(-32601, 'Method not found')
        owner = mcp_owner()
        if not owner:
            response = failure(-32001, 'Connect an active DREDGE subscription or reviewer grant.')
            response.status_code = 401
            response.headers['WWW-Authenticate'] = 'Bearer resource_metadata="'+origin()+'/.well-known/oauth-protected-resource/mcp"'
            return response
        params = body.get('params') or {}; name = params.get('name'); args = params.get('arguments') or {}
        if name not in {item['name'] for item in definitions}: return failure(-32602, 'Unknown tool')
        workflows = args.get('workflows')
        if not isinstance(workflows, list) or not 1 <= len(workflows) <= 20:
            return failure(-32602, 'Supply 1 to 20 workflows.')
        findings=[]
        for item in workflows:
            if not isinstance(item, dict) or set(item) != {'path','content'} or not re.fullmatch(r'\.github/workflows/[A-Za-z0-9][A-Za-z0-9_.-]*\.ya?ml', item.get('path','')) or not isinstance(item.get('content'),str) or len(item['content'])>32000:
                return failure(-32602, 'Use bounded canonical workflow paths and text.')
            if re.search(r'\bwindows-(latest|2019|2022)\b', item['content']): findings.append({'path':item['path'],'check':'windows-runner-compatibility','action':'manual-review'})
            if not re.search(r'^permissions\s*:', item['content'], re.M): findings.append({'path':item['path'],'check':'explicit-permissions','action':'approval-required'})
        base={'version':'1.2.0','reviewed_files':len(workflows),'method':'text-pattern-heuristics','findings':findings,'writes_performed':0,'ci_tests_run':False,'limits':'Not a YAML parser or a full security audit. Workflow source is not retained.'}
        if name == 'thresh_review_workflows': return reply(mcp_output(base))
        if name == 'thresh_plan_repairs':
            base['plan']=[{'path':f['path'],'check':f['check'],'approval_required':True,'steps':['Inspect the complete workflow.','Prepare a minimal diff in an isolated branch.','Run affected checks.','Obtain maintainer approval.']} for f in findings]
            base.update(status='proposal-only',changes_committed=False); return reply(mcp_output(base))
        if set(args) != {'repository','base_sha','approval','workflows'} or args.get('approval') != 'SUBMIT_REPAIR_REQUEST' or not re.fullmatch(r'[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?/[A-Za-z0-9][A-Za-z0-9_.-]{0,99}', args.get('repository','')) or not re.fullmatch(r'[0-9a-fA-F]{40}',args.get('base_sha','')) or not findings:
            return failure(-32602,'Supply an approved bounded repair request with supported findings.')
        summary=json.dumps([{'path':f['path'],'check':f['check']} for f in findings],sort_keys=True,separators=(',',':'))
        digest=hashlib.sha256(json.dumps({'owner':owner,'repository':args['repository'],'base_sha':args['base_sha'].lower(),'findings':summary},sort_keys=True).encode()).hexdigest()
        with store.connect() as db:
            db.execute('BEGIN IMMEDIATE'); existing=db.execute('SELECT request_id,status FROM thresh_repair_requests WHERE request_hash=?',(digest,)).fetchone()
            if existing: request_id,status,deduplicated=existing['request_id'],existing['status'],True
            else:
                request_id='trr_'+secrets.token_hex(12); status='pending-maintainer-review'; deduplicated=False
                db.execute('INSERT INTO thresh_repair_requests VALUES(?,?,?,?,?,?,?,?,?,?)',(request_id,'direct-mcp','oauth',owner,args['repository'],args['base_sha'].lower(),summary,status,int(time.time()),digest))
        return reply(mcp_output({'version':'1.2.0','request_id':request_id,'status':status,'repository':args['repository'],'base_sha':args['base_sha'].lower(),'deduplicated':deduplicated,'github_changes_performed':0,'maintainer_approval_required':True}))

    @bp.post('/api/thresh/link-challenges')
    def start_link():
        subject = identity()
        if subject is None:
            return jsonify(error='Authorized Thresh Site identity required.'), 401
        origin = current_app.config.get('BILLING_ORIGIN', '').rstrip('/')
        if not origin.startswith('https://'):
            return jsonify(error='Website origin is not configured.'), 503
        token = secrets.token_urlsafe(32)
        with store.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('DELETE FROM thresh_link_challenges WHERE expires<=? OR (site_id=? AND subject=?)',
                       (int(time.time()), SITE_ID, subject))
            db.execute('INSERT INTO thresh_link_challenges VALUES(?,?,?,?)',
                       (hashlib.sha256(token.encode()).hexdigest(), SITE_ID, subject, int(time.time())+300))
        return jsonify(link_url=origin+'/thresh/link?challenge='+token, expires_in=300)

    @bp.route('/thresh/link', methods=['GET', 'POST'])
    def link():
        token = (request.form.get('challenge') if request.method == 'POST' else request.args.get('challenge')) or session.get('thresh_pending_link', '')
        with store.connect() as db:
            row = challenge_row(token, db)
        if row is None:
            session.pop('thresh_pending_link', None)
            return page('Link expired', 'Start a new link from Thresh.'), 400
        if not current_user.is_authenticated:
            if request.method != 'GET':
                return jsonify(error='Sign in to DREDGE.'), 401
            session['thresh_pending_link'] = token
            return redirect(url_for('auth.login'))
        if request.method == 'GET':
            session['thresh_pending_link'] = token
            csrf = session.setdefault('thresh_csrf', secrets.token_urlsafe(32))
            return page('Link your DREDGE account', 'Review this account before continuing.',
                        account=current_user.get_id(), challenge=token, csrf=csrf)
        if token != session.get('thresh_pending_link') or not csrf_ok():
            return jsonify(error='Refresh the link confirmation before continuing.'), 403
        with store.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = challenge_row(token, db)
            if row is None:
                return jsonify(error='Link expired or already used.'), 400
            pinned = db.execute('SELECT owner FROM thresh_links WHERE site_id=? AND subject=?', (SITE_ID, row['subject'])).fetchone()
            other = db.execute('SELECT subject FROM thresh_links WHERE site_id=? AND owner=?', (SITE_ID, current_user.get_id())).fetchone()
            if (pinned and pinned['owner'] != current_user.get_id()) or (other and other['subject'] != row['subject']):
                return jsonify(error='An existing account pin cannot be replaced through this flow.'), 409
            db.execute('INSERT INTO thresh_links VALUES(?,?,?,0) ON CONFLICT(site_id,subject) DO UPDATE SET revoked=0',
                       (SITE_ID, row['subject'], current_user.get_id()))
            db.execute('DELETE FROM thresh_link_challenges WHERE hash=?', (row['hash'],))
        session.pop('thresh_pending_link', None)
        return page('Account linked', 'Tool access is checked against the live subscription status on every call.', linked=True, csrf=session['thresh_csrf'])

    @bp.post('/thresh/unlink')
    def unlink():
        if not current_user.is_authenticated:
            return jsonify(error='Sign in to DREDGE.'), 401
        if not csrf_ok():
            return jsonify(error='Refresh the confirmation before continuing.'), 403
        with store.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('DELETE FROM thresh_link_challenges WHERE site_id=? AND subject IN (SELECT subject FROM thresh_links WHERE site_id=? AND owner=?)',
                       (SITE_ID, SITE_ID, current_user.get_id()))
            db.execute('UPDATE thresh_links SET revoked=1 WHERE site_id=? AND owner=?', (SITE_ID, current_user.get_id()))
        session.pop('thresh_pending_link', None)
        return page('Link revoked', 'The link no longer grants Thresh access. This does not cancel or change your subscription.')

    @bp.post('/api/thresh/entitlement')
    def entitlement():
        subject = identity()
        if subject is None:
            return jsonify(error='Authorized Thresh Site identity required.'), 401
        with store.connect() as db:
            link = db.execute('SELECT owner,revoked FROM thresh_links WHERE site_id=? AND subject=?', (SITE_ID, subject)).fetchone()
        active = bool(link and not link['revoked'] and subscription_check(link['owner']))
        return jsonify(site_id=SITE_ID, subject=subject, entitlement='thresh:use', active=active,
                       billing_mode='live' if active else 'unverified', expires_at=int(time.time())+60)

    @bp.post('/api/thresh/repair-requests')
    def create_repair_request():
        body = repair_request()
        if body is None:
            return jsonify(error='Authorized, bounded Thresh repair request required.'), 401
        with store.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            link = db.execute('SELECT owner,revoked FROM thresh_links WHERE site_id=? AND subject=?',
                              (SITE_ID, body['subject'])).fetchone()
            if not link or link['revoked'] or not subscription_check(link['owner']):
                return jsonify(error='A linked, active DREDGE subscription is required.'), 403
            summary = json.dumps(body['findings'], sort_keys=True, separators=(',', ':'))
            digest = hashlib.sha256(json.dumps({
                'site_id': SITE_ID, 'subject': body['subject'], 'repository': body['repository'],
                'base_sha': body['base_sha'], 'findings': body['findings']
            }, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            existing = db.execute('SELECT request_id,status,created FROM thresh_repair_requests WHERE request_hash=?',
                                  (digest,)).fetchone()
            if existing:
                return jsonify(request_id=existing['request_id'], status=existing['status'],
                               created_at=existing['created'], deduplicated=True,
                               repository=body['repository'], base_sha=body['base_sha'])
            request_id = 'trr_'+secrets.token_hex(12)
            created = int(time.time())
            db.execute('INSERT INTO thresh_repair_requests VALUES(?,?,?,?,?,?,?,?,?,?)',
                       (request_id, SITE_ID, body['subject'], link['owner'], body['repository'],
                        body['base_sha'], summary, 'pending-maintainer-review', created, digest))
        return jsonify(request_id=request_id, status='pending-maintainer-review', created_at=created,
                       deduplicated=False, repository=body['repository'], base_sha=body['base_sha']), 201

    app.register_blueprint(bp)
    return True
