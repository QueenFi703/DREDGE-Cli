"""Default-off staging reviewer authentication, never a provider impersonation.

Configuration/provisioning is an operator action, not a public signup endpoint.
Only password hashes belong in configuration. No credentials are generated here.
Review tool data lives in a distinct, marker-checked SQLite store; OAuth grants
retain the existing protocol store and are checked against current review access.
"""
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import time
import uuid

from flask import Blueprint, current_app, jsonify, redirect, render_template_string, request, session
from flask_login import current_user, login_user, logout_user
from werkzeug.security import check_password_hash

STAGING_ORIGIN = 'https://chatgpt-staging.dredgeoriongateway.com'
LOGIN_PATH = '/auth/reviewer'
MAX_LIFETIME = 30 * 86400
WINDOW_SECONDS = 300
MAX_ATTEMPTS = 5
OWNER_RE = re.compile(r'review:[0-9a-f]{64}\Z')
HASH_RE = re.compile(r'scrypt:32768:8:1\$[A-Za-z0-9]{16,64}\$[0-9a-f]{128}\Z')
TEMPLATE = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>DREDGE reviewer sign-in</title>
<h1>DREDGE reviewer sample account</h1>
<p>Isolated fictional data only. No payments or AI execution.</p>
{% if signed_in %}<p>Signed in. Return to ChatGPT to connect this account.</p>
<form action="/auth/reviewer/logout" method="post"><input type="hidden" name="csrf_token" value="{{ csrf }}"><button>Sign out</button></form>
{% else %}{% if error %}<p>{{ error }}</p>{% endif %}
<form action="/auth/reviewer" method="post">
<input type="hidden" name="csrf_token" value="{{ csrf }}">
<label>Username <input name="username" autocomplete="username" maxlength="64" required></label>
<label>Password <input type="password" name="password" autocomplete="current-password" maxlength="256" required></label>
<button>Sign in</button></form>{% endif %}</html>'''


def is_review_owner(owner):
    # All reserved review namespace values fail closed, including malformed IDs.
    return isinstance(owner, str) and owner.startswith('review:')


def active_review_owner(app, owner):
    account = app.config.get('REVIEWER_ACCOUNT')
    valid = bool(app.config.get('DREDGE_REVIEWER_LOGIN_ENABLED') is True
                and app.config.get('DREDGE_CHATGPT_ENABLED') is True
                and app.config.get('CHATGPT_ISSUER') == STAGING_ORIGIN
                and isinstance(account, dict) and OWNER_RE.fullmatch(owner or '')
                and hmac.compare_digest(owner, account['owner'])
                and time.time() < account['expires_at'])
    if not valid:
        return False
    try:
        with app.extensions['reviewer_store'].connect() as db:
            row = db.execute('SELECT s.revoked,a.owner,a.expires_at FROM reviewer_account_state s '
                             'JOIN reviewer_active_generation a ON a.owner=s.owner WHERE s.owner=?', (owner,)).fetchone()
            return bool(row and not row['revoked'] and time.time() < row['expires_at'])
    except (KeyError, sqlite3.Error):
        return False


def revoke_review_access(app):
    """Operator-only administrative helper; no HTTP route. Requires separate approval."""
    with app.extensions['reviewer_store'].connect() as db:
        db.execute('UPDATE reviewer_account_state SET revoked=1 WHERE owner=?',
                   (app.config['REVIEWER_ACCOUNT']['owner'],))


def _review_store(path, main_path):
    """Never initialize a review store over an existing unmarked database."""
    target, main = Path(path), Path(main_path)
    if not target.is_absolute() or target.is_symlink() or target.resolve() == main.resolve():
        raise RuntimeError('Reviewer storage must be a distinct absolute non-symlink path.')
    if target.exists() and main.exists() and os.path.samefile(target, main):
        raise RuntimeError('Reviewer storage must not alias the ordinary Studio database.')
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        with sqlite3.connect(target) as db:
            try:
                marker = db.execute('SELECT purpose FROM reviewer_store_marker').fetchall()
            except sqlite3.Error as exc:
                raise RuntimeError('Existing database is not a reviewer fixture store.') from exc
            if marker != [('dredge-fictional-review-v1',)]:
                raise RuntimeError('Existing database is not a reviewer fixture store.')
    else:
        os.close(fd)
        with sqlite3.connect(target) as db:
            db.execute('CREATE TABLE reviewer_store_marker(purpose TEXT PRIMARY KEY)')
            db.execute('INSERT INTO reviewer_store_marker VALUES(?)', ('dredge-fictional-review-v1',))
    from .studio import StudioStore
    from .studio_budgets import initialize_schema as budgets
    from .studio_research_controls import initialize_schema as sources
    from .public_source_revalidation import initialize_schema as checks
    result = StudioStore(target)
    with result.connect() as db:
        budgets(db)
        sources(db)
        checks(db)
        db.execute('CREATE TABLE IF NOT EXISTS reviewer_account_state '
                   '(owner TEXT PRIMARY KEY, revoked INTEGER NOT NULL DEFAULT 0 CHECK(revoked IN (0,1)))')
        db.execute('CREATE TABLE IF NOT EXISTS reviewer_active_generation '
                   '(id INTEGER PRIMARY KEY CHECK(id=1), generation INTEGER NOT NULL, owner TEXT NOT NULL, expires_at INTEGER NOT NULL)')
        db.execute('CREATE TABLE IF NOT EXISTS reviewer_login_attempts '
                   '(id INTEGER PRIMARY KEY CHECK(id=1), window_start REAL NOT NULL, attempts INTEGER NOT NULL)')
    return result


def _seed(store, owner, generation, expiry):
    """Insert explicitly synthetic, owner-bound fixtures once per credential generation."""
    from .studio_budgets import set_budget
    from .studio_research_controls import capture_sources
    records = [
        ('library', 'What services could a fictional library offer?', [],
         'FICTIONAL REVIEW FIXTURE, NOT LIVE RESEARCH: The fictional library Cedar Lantern could offer a reading circle, '
         'a computer-skills session and a craft table. This is invented sample content for demonstrating '
         'saved research retrieval. No real library or patron is involved.'),
        ('sources', 'Demonstrate a bounded public-source check for a fictional training exercise.',
         [dict(id='source-1', title='Missouri DSS: Child Support', url='https://dss.mo.gov/child-support',
               excerpt='', provenance='public_catalog_reference', verification='unverified')],
         'FICTIONAL REVIEW FIXTURE: This record cites a public catalog page [source-1] only to demonstrate '
         'source checks. It contains no case or personal data and no legal advice. '
         'No source verification is claimed until a source check actually completes.')]
    with store.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        active = db.execute('SELECT * FROM reviewer_active_generation WHERE id=1').fetchone()
        if active and (generation < active['generation'] or (generation == active['generation']
                and (owner != active['owner'] or expiry != active['expires_at']))):
            raise RuntimeError('Reviewer credentials require a new monotonic generation; rollback is forbidden.')
        if active is None or generation > active['generation']:
            db.execute('UPDATE reviewer_account_state SET revoked=1')
            db.execute('INSERT OR REPLACE INTO reviewer_active_generation VALUES(1,?,?,?)', (generation, owner, expiry))
        db.execute('INSERT OR IGNORE INTO reviewer_account_state(owner) VALUES(?)', (owner,))
        for label, query, evidence, answer in records:
            ident = str(uuid.uuid5(uuid.NAMESPACE_URL, owner + '/' + label))
            if db.execute('SELECT 1 FROM runs WHERE id=?', (ident,)).fetchone():
                continue
            now = time.time()
            payload = dict(query=query, pipeline_type='standard', execution_mode='demo',
                           data_scope='public_or_fictional', evidence=evidence,
                           fixture_provenance='manually_authored_fictional_review_fixture')
            result = dict(answer=answer, provider_calls=0, token_usage=None, cost_usd=None)
            db.execute('INSERT INTO runs(id,owner,status,payload,result,created,ended) VALUES(?,?,?,?,?,?,?)',
                       (ident, owner, 'completed', json.dumps(payload), json.dumps(result), now, now))
            capture_sources(db, owner, ident, payload)
            store.audit(db, owner, 'fictional_review_fixture_created', ident,
                        {'provenance': 'manually_authored', 'provider_calls': 0})
        # Never reset an existing budget or reservations after a restart.
        if not db.execute('SELECT 1 FROM studio_budget_settings WHERE owner=?', (owner,)).fetchone():
            set_budget(db, owner, enabled=True, limit_micro_usd=0)


def register_reviewer_login(app):
    flag = os.environ.get('DREDGE_REVIEWER_LOGIN_ENABLED', 'false')
    app.config['DREDGE_REVIEWER_LOGIN_ENABLED'] = False
    if flag == 'false':
        return
    if flag != 'true':
        raise RuntimeError('DREDGE_REVIEWER_LOGIN_ENABLED must be true or false.')
    if (os.environ.get('DREDGE_CHATGPT_TUNNEL_ENABLED') != 'true'
            or not app.config.get('DREDGE_CHATGPT_ENABLED')
            or app.config.get('CHATGPT_ISSUER') != STAGING_ORIGIN):
        raise RuntimeError('Reviewer login is restricted to the dedicated DREDGE staging tunnel.')
    username = os.environ.get('DREDGE_REVIEWER_USERNAME', '')
    password_hash = os.environ.get('DREDGE_REVIEWER_PASSWORD_HASH', '')
    if not re.fullmatch(r'[A-Za-z0-9_-]{8,64}', username) or not HASH_RE.fullmatch(password_hash):
        raise RuntimeError('A dedicated username and scrypt:32768:8:1 password hash are required.')
    try:
        expiry = int(os.environ.get('DREDGE_REVIEWER_EXPIRES_AT', ''))
        generation = int(os.environ.get('DREDGE_REVIEWER_GENERATION', ''))
        if not 1 <= generation <= 2147483647:
            raise ValueError('Invalid generation')
    except ValueError as exc:
        raise RuntimeError('Reviewer access needs an explicit expiry timestamp.') from exc
    if not 0 < expiry <= time.time() + MAX_LIFETIME:
        raise RuntimeError('Reviewer access needs a positive expiry no more than 30 days ahead.')
    if expiry <= time.time():
        # A naturally expired review account must not take down personal sign-in
        # on the next service restart. No store or fixtures are opened here.
        return
    path = os.environ.get('DREDGE_REVIEWER_DB_PATH', '')
    review_store = _review_store(path, app.extensions['studio_store'].path)
    # Changing the password invalidates old identities, browser sessions and grants.
    owner = 'review:' + hashlib.sha256((username + '\0' + password_hash + '\0' + str(generation)).encode()).hexdigest()
    app.config['REVIEWER_ACCOUNT'] = dict(username=username, password_hash=password_hash,
                                          owner=owner, expires_at=expiry)
    app.extensions['reviewer_store'] = review_store
    _seed(review_store, owner, generation, expiry)
    app.config['DREDGE_REVIEWER_LOGIN_ENABLED'] = True
    app.config.update(SESSION_COOKIE_SECURE=True, SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE='Lax')

    @app.before_request
    def review_boundary():
        owner_id = session.get('_user_id')
        if not is_review_owner(owner_id):
            return None
        if not active_review_owner(app, owner_id):
            logout_user()
            session.clear()
            return jsonify(error='Review access expired or revoked.'), 401
        allowed = {'/auth/login', LOGIN_PATH, LOGIN_PATH + '/logout', '/chatgpt/oauth/authorize',
                   '/chatgpt/oauth/token', '/chatgpt/oauth/revoke'}
        if request.path not in allowed:
            return jsonify(error='Review accounts are limited to the OAuth-connected sample tools.'), 403

    bp = Blueprint('reviewer_login', __name__)

    @bp.before_request
    def login_boundary():
        if not active_review_owner(app, owner):
            return jsonify(error='Review access unavailable.'), 403
        if request.host_url.rstrip('/') != STAGING_ORIGIN or request.args:
            return jsonify(error='Use the configured review sign-in page.'), 400
        request.max_content_length = 4096
        if request.method == 'POST':
            if request.headers.get('Origin', STAGING_ORIGIN) != STAGING_ORIGIN:
                return jsonify(error='Invalid sign-in origin.'), 403
            if request.mimetype != 'application/x-www-form-urlencoded':
                return jsonify(error='Invalid sign-in form.'), 400
            expected = session.get('reviewer_csrf', '')
            supplied = request.form.get('csrf_token', '')
            if not expected or not re.fullmatch(r'[A-Za-z0-9_-]{43}', supplied) or not hmac.compare_digest(expected, supplied):
                return jsonify(error='Refresh the sign-in page.'), 403

    @bp.after_request
    def headers(response):
        response.headers.update({'Cache-Control': 'no-store', 'Pragma': 'no-cache',
            'Referrer-Policy': 'no-referrer', 'X-Content-Type-Options': 'nosniff', 'X-Frame-Options': 'DENY',
            'Content-Security-Policy': "default-src 'none'; form-action 'self'; frame-ancestors 'none'; base-uri 'none'"})
        return response

    @bp.route(LOGIN_PATH, methods=['GET', 'POST'])
    def sign_in():
        if current_user.is_authenticated and current_user.get_id() != owner:
            return jsonify(error='Use a separate browser session for the review account.'), 409
        error = None
        if request.method == 'POST':
            if (set(request.form) != {'username', 'password', 'csrf_token'}
                    or any(len(request.form.getlist(k)) != 1 for k in request.form)
                    or not 1 <= len(request.form['password']) <= 256
                    or len(request.form['username']) > 64):
                return jsonify(error='Invalid sign-in form.'), 400
            session.pop('reviewer_csrf', None)
            with review_store.connect() as db:
                db.execute('BEGIN IMMEDIATE')
                now = time.time()
                row = db.execute('SELECT * FROM reviewer_login_attempts WHERE id=1').fetchone()
                if row is None or now - row['window_start'] >= WINDOW_SECONDS:
                    db.execute('INSERT OR REPLACE INTO reviewer_login_attempts VALUES(1,?,1)', (now,))
                elif row['attempts'] >= MAX_ATTEMPTS:
                    response = jsonify(error='Too many sign-in attempts. Try again later.')
                    response.headers['Retry-After'] = str(WINDOW_SECONDS)
                    return response, 429
                else:
                    db.execute('UPDATE reviewer_login_attempts SET attempts=attempts+1 WHERE id=1')
            password_ok = check_password_hash(app.config['REVIEWER_ACCOUNT']['password_hash'], request.form['password'])
            username_ok = hmac.compare_digest(username.encode(), request.form['username'].encode())
            if password_ok and username_ok and active_review_owner(app, owner):
                # Rotate all login/session state; retain only the server-created
                # anonymous consent binding needed to resume this OAuth transaction.
                retained = {k: session[k] for k in ('chatgpt_oauth_binding', 'chatgpt_oauth_csrf', 'chatgpt_return_to') if k in session}
                session.clear()
                session.update(retained)
                from .auth import User, remember_studio_session, studio_return_path
                user = User(owner, 'DREDGE review account', '', 'review')
                login_user(user, remember=False, fresh=True)
                remember_studio_session(user)
                with review_store.connect() as db:
                    db.execute('DELETE FROM reviewer_login_attempts WHERE id=1')
                destination = studio_return_path()
                return redirect(destination if destination.startswith('/chatgpt/oauth/authorize?transaction=') else LOGIN_PATH)
            error = 'Sign-in failed.'
        csrf = session.setdefault('reviewer_csrf', secrets.token_urlsafe(32))
        return render_template_string(TEMPLATE, csrf=csrf, error=error,
                                      signed_in=current_user.is_authenticated), 401 if error else 200

    @bp.post(LOGIN_PATH + '/logout')
    def sign_out():
        if set(request.form) != {'csrf_token'} or len(request.form.getlist('csrf_token')) != 1:
            return jsonify(error='Invalid sign-out form.'), 400
        if not current_user.is_authenticated or current_user.get_id() != owner:
            return jsonify(error='No review session.'), 403
        logout_user()
        session.clear()
        return redirect(LOGIN_PATH)

    app.register_blueprint(bp)


def review_access_summary(app, owner):
    if not active_review_owner(app, owner):
        raise PermissionError('Review access expired or revoked.')
    return dict(workspace_role='operator', studio_report_available=True,
                subscription_status='inactive', paid_access=False, paid_ai_included=False,
                subscription_requirement_enabled=False, billing_configured=False,
                billing_environment='disabled', billing_environments={'stripe': 'disabled', 'paypal': 'disabled'},
                subscription_status_source='stored_verified_records', subscription_status_checked_live=False,
                in_chat_purchases_enabled=False, subscription_info_url=None,
                review_account=True, data_environment='isolated_fictional_fixtures',
                entitlement_notes=['Fictional review account; no paid entitlement, payments or AI execution.'])
