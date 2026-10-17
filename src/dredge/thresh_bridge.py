"""Optional Thresh account linking and entitlement authority. © 2026 Sophia Cole.

Disabled unless THRESH_BRIDGE_KEY is configured. Never match accounts by email,
create charges, assign Studio roles, or expose investigation data.
"""
import hashlib
import hmac
import os
import re
import secrets
import time
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

    app.register_blueprint(bp)
    return True
