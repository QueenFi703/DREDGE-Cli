"""Synthetic account linking; no production credentials, payments, or network calls."""
import sqlite3
import time
from contextlib import contextmanager
from urllib.parse import urlsplit, parse_qs
import pytest
from flask import Flask, session
from flask_login import LoginManager, UserMixin
from dredge.thresh_bridge import register_thresh_bridge, SITE_ID

class Store:
    def __init__(self, path): self.path = str(path)
    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            with connection: yield connection
        finally: connection.close()
class User(UserMixin):
    def __init__(self, user_id): self.id = user_id

@pytest.fixture
def bridge(tmp_path, monkeypatch):
    monkeypatch.setenv('THRESH_BRIDGE_KEY', 'synthetic-only-bridge-secret-key-12345')
    app = Flask(__name__); app.secret_key = 'synthetic-cookie-secret'
    app.config.update(TESTING=True, BILLING_ORIGIN='https://dredge.example.test')
    login = LoginManager(app); login.user_loader(lambda user_id: User(user_id))
    app.add_url_rule('/auth/login', 'auth.login', lambda: 'synthetic sign in')
    store = Store(tmp_path/'bridge.sqlite'); eligible = {'owner-a': True, 'owner-b': False}
    assert register_thresh_bridge(app, store=store, subscription_check=lambda owner: eligible.get(owner,False))
    return app, store, eligible

def identity(subject='site-alice'):
    return {'site_id':SITE_ID,'subject':subject,'entitlement':'thresh:use'}
HEADERS={'Authorization':'Bearer synthetic-only-bridge-secret-key-12345'}
ORIGIN={'Origin':'https://dredge.example.test'}
def client_for(app, owner=None):
    client=app.test_client()
    if owner:
        with client.session_transaction() as s: s['_user_id']=owner; s['_fresh']=True
    return client

def challenge(app, subject='site-alice'):
    r=app.test_client().post('/api/thresh/link-challenges',json=identity(subject),headers=HEADERS)
    assert r.status_code==200
    return parse_qs(urlsplit(r.json['link_url']).query)['challenge'][0]

def confirm(client, token, headers=ORIGIN):
    assert client.get('/thresh/link?challenge='+token).status_code==200
    with client.session_transaction() as s: csrf=s['thresh_csrf']
    return client.post('/thresh/link',data={'csrf':csrf,'challenge':token},headers=headers)

def status(app, subject='site-alice'):
    return app.test_client().post('/api/thresh/entitlement',json=identity(subject),headers=HEADERS)

def test_disabled_by_default_creates_no_routes_or_tables(monkeypatch):
    monkeypatch.delenv('THRESH_BRIDGE_KEY',raising=False)
    app=Flask(__name__)
    assert register_thresh_bridge(app) is False
    assert app.test_client().post('/api/thresh/entitlement').status_code==404

def test_weak_secret_is_rejected(monkeypatch):
    monkeypatch.setenv('THRESH_BRIDGE_KEY','short')
    with pytest.raises(RuntimeError): register_thresh_bridge(Flask(__name__))

def test_unauthorized_authority_requests_are_denied(bridge):
    app,_,_=bridge
    for headers in ({},{'Authorization':'Bearer wrong'},{'Authorization':'ümlaut'}):
        assert app.test_client().post('/api/thresh/entitlement',json=identity(),headers=headers).status_code==401

def test_exact_site_scope_and_server_owned_identity_required(bridge):
    app,_,_=bridge
    for changes in ({'site_id':'other'},{'entitlement':'studio:read'},{'owner':'owner-a'},{'subject':''}):
        data={**identity(),**changes}
        assert app.test_client().post('/api/thresh/entitlement',json=data,headers=HEADERS).status_code==401

def test_anonymous_link_requires_login_and_retains_only_pending_challenge(bridge):
    app,_,_=bridge; token=challenge(app); client=client_for(app)
    r=client.get('/thresh/link?challenge='+token)
    assert r.status_code==302 and r.headers['Location']=='/auth/login'
    with client.session_transaction() as s: assert s['thresh_pending_link']==token
    assert status(app).json['active'] is False

def test_link_requires_explicit_csrf_confirmation_and_origin(bridge):
    app,_,_=bridge; token=challenge(app); client=client_for(app,'owner-a')
    assert confirm(client,token,headers={'Origin':'https://attacker.invalid'}).status_code==403
    assert client.post('/thresh/link',data={'challenge':token,'csrf':'wrong'},headers=ORIGIN).status_code==403
    assert status(app).json['active'] is False

def test_happy_path_uses_proved_owner_and_returns_no_private_account_data(bridge):
    app,_,_=bridge; token=challenge(app); client=client_for(app,'owner-a')
    assert confirm(client,token).status_code==200
    grant=status(app).json
    assert grant['active'] is True and grant['billing_mode']=='live'
    assert grant['subject']=='site-alice' and grant['expires_at']<=int(time.time())+61
    assert 'owner-a' not in str(grant) and 'email' not in grant

def test_link_does_not_grant_an_unpaid_account_access(bridge):
    app,_,_=bridge
    assert confirm(client_for(app,'owner-b'),challenge(app)).status_code==200
    assert status(app).json['active'] is False

def test_entitlement_rechecks_subscription_state(bridge):
    app,_,eligible=bridge
    assert confirm(client_for(app,'owner-a'),challenge(app)).status_code==200
    assert status(app).json['active'] is True
    eligible['owner-a']=False
    assert status(app).json['active'] is False

def test_one_use_and_expired_challenges(bridge):
    app,store,_=bridge; token=challenge(app); client=client_for(app,'owner-a')
    assert confirm(client,token).status_code==200
    assert client.get('/thresh/link?challenge='+token).status_code==400
    token=challenge(app)
    with store.connect() as db: db.execute('UPDATE thresh_link_challenges SET expires=0')
    assert client.get('/thresh/link?challenge='+token).status_code==400

def test_site_identity_cannot_be_reassigned_to_another_owner(bridge):
    app,_,_=bridge
    assert confirm(client_for(app,'owner-a'),challenge(app)).status_code==200
    assert confirm(client_for(app,'owner-b'),challenge(app)).status_code==409
    assert status(app).json['active'] is True

def test_one_owner_cannot_pin_a_second_site_identity(bridge):
    app,_,_=bridge
    assert confirm(client_for(app,'owner-a'),challenge(app)).status_code==200
    assert confirm(client_for(app,'owner-a'),challenge(app,'site-bob')).status_code==409
    assert status(app,'site-bob').json['active'] is False

def test_revocation_denies_access_and_invalidates_older_pending_link(bridge):
    app,_,_=bridge; client=client_for(app,'owner-a')
    assert confirm(client,challenge(app)).status_code==200
    pending=challenge(app)
    with client.session_transaction() as s: csrf=s['thresh_csrf']
    assert client.post('/thresh/unlink',data={'csrf':csrf},headers=ORIGIN).status_code==200
    assert status(app).json['active'] is False
    assert client.get('/thresh/link?challenge='+pending).status_code==400

def test_security_headers_and_unknown_subjects(bridge):
    app,_,_=bridge; response=status(app,'unknown')
    assert response.json['active'] is False
    assert response.headers['Cache-Control']=='no-store'
    assert response.headers['Referrer-Policy']=='no-referrer'

def test_login_return_preserves_existing_studio_route_without_pending_link():
    from dredge.auth import studio_return_path
    app=Flask(__name__); app.secret_key='synthetic-cookie'
    with app.test_request_context('/'):
        session['studio_return_run']='12345678-1234-1234-1234-123456789abc'
        assert studio_return_path()=='/advanced?run=12345678-1234-1234-1234-123456789abc'
    with app.test_request_context('/'):
        session['studio_return_run']='https://attacker.invalid/'
        assert studio_return_path()=='/advanced'

def test_pending_link_returns_only_to_the_fixed_local_confirmation_route():
    from dredge.auth import studio_return_path
    app=Flask(__name__); app.secret_key='synthetic-cookie'
    with app.test_request_context('/'):
        session['thresh_pending_link']='https://attacker.invalid/'
        assert studio_return_path()=='/thresh/link'
