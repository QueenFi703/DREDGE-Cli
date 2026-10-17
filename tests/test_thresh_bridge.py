"""Synthetic account linking; no production credentials, payments, or network calls."""
import base64
import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
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

def repair(app, subject='site-alice', **changes):
    body={**identity(subject),'repository':'QueenFi703/example','base_sha':'a'*40,
          'approval':'SUBMIT_REPAIR_REQUEST','findings':[{'path':'.github/workflows/ci.yml','check':'explicit-permissions'}],**changes}
    return app.test_client().post('/api/thresh/repair-requests',json=body,headers=HEADERS)

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

def test_repair_request_requires_a_linked_paid_owner_and_stores_no_source(bridge):
    app,store,_=bridge
    assert repair(app).status_code==403
    assert confirm(client_for(app,'owner-a'),challenge(app)).status_code==200
    response=repair(app)
    assert response.status_code==201 and response.json['status']=='pending-maintainer-review'
    with store.connect() as db:
        row=db.execute('SELECT * FROM thresh_repair_requests').fetchone()
    assert row['owner']=='owner-a' and 'content' not in row.keys() and 'name: CI' not in row['summary']

def test_repair_request_is_idempotent_and_does_not_mutate_a_repository(bridge):
    app,_,_=bridge
    assert confirm(client_for(app,'owner-a'),challenge(app)).status_code==200
    first=repair(app); second=repair(app)
    assert first.status_code==201 and second.status_code==200
    assert first.json['request_id']==second.json['request_id'] and second.json['deduplicated'] is True

def test_concurrent_identical_repair_requests_create_one_record(bridge):
    app,store,_=bridge
    assert confirm(client_for(app,'owner-a'),challenge(app)).status_code==200
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses=list(pool.map(lambda _: repair(app), range(2)))
    assert sorted(r.status_code for r in responses)==[200,201]
    assert len({r.json['request_id'] for r in responses})==1
    with store.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM thresh_repair_requests').fetchone()[0]==1

def test_repair_request_rechecks_subscription_and_rejects_unbounded_payloads(bridge):
    app,_,eligible=bridge
    assert confirm(client_for(app,'owner-a'),challenge(app)).status_code==200
    eligible['owner-a']=False
    assert repair(app).status_code==403
    eligible['owner-a']=True
    assert repair(app,approval='YES').status_code==401
    assert repair(app,repository='../private').status_code==401
    assert repair(app,findings=[{'path':'secrets.txt','check':'explicit-permissions'}]).status_code==401

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

def test_oauth_metadata_registration_pkce_reviewer_grant_and_mcp(bridge, monkeypatch):
    app,_,_=bridge
    reviewer_code='synthetic-reviewer-code-12345'
    monkeypatch.setenv('THRESH_REVIEWER_CODE_HASH',hashlib.sha256(reviewer_code.encode()).hexdigest())
    monkeypatch.setenv('THRESH_REVIEWER_EXPIRES',str(int(time.time())+3600))
    client=client_for(app,'owner-b')
    metadata=client.get('/.well-known/oauth-authorization-server').json
    assert metadata['code_challenge_methods_supported']==['S256']
    assert metadata['response_modes_supported']==['query']
    assert client.get('/.well-known/openid-configuration').json==metadata
    redirect_uri='https://chatgpt.com/connector/oauth/callback'
    registered=client.post('/oauth/register',json={'redirect_uris':[redirect_uri]})
    assert registered.status_code==201
    client_id=registered.json['client_id']
    verifier='v'*43
    code_challenge=base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b'=').decode()
    query={'client_id':client_id,'redirect_uri':redirect_uri,'response_type':'code','state':'state-1',
           'code_challenge':code_challenge,'code_challenge_method':'S256','scope':'thresh:use'}
    consent=client.get('/oauth/authorize',query_string=query)
    assert consent.status_code==200 and b'Reviewer access code' in consent.data
    with client.session_transaction() as saved: csrf=saved['thresh_oauth_csrf']
    allowed=client.post('/oauth/authorize',data={'csrf':csrf,'decision':'allow','reviewer_code':reviewer_code})
    assert allowed.status_code==302
    code=parse_qs(urlsplit(allowed.location).query)['code'][0]
    token=client.post('/oauth/token',data={'grant_type':'authorization_code','client_id':client_id,
        'redirect_uri':redirect_uri,'code':code,'code_verifier':verifier})
    assert token.status_code==200 and token.json['token_type']=='Bearer'
    workflow={'workflows':[{'path':'.github/workflows/ci.yml','content':'runs-on: windows-latest'}]}
    call={'jsonrpc':'2.0','id':1,'method':'tools/call','params':{'name':'thresh_review_workflows','arguments':workflow}}
    result=client.post('/mcp',json=call,headers={'Authorization':'Bearer '+token.json['access_token']})
    output=json.loads(result.json['result']['content'][0]['text'])
    assert result.status_code==200 and len(output['findings'])==2 and output['writes_performed']==0

def test_oauth_rejects_unsafe_redirect_and_mcp_requires_bearer(bridge):
    app,_,_=bridge; client=app.test_client()
    assert client.post('/oauth/register',json={'redirect_uris':['http://attacker.invalid/callback']}).status_code==400
    call={'jsonrpc':'2.0','id':1,'method':'tools/call','params':{'name':'thresh_review_workflows','arguments':{'workflows':[{'path':'.github/workflows/ci.yml','content':'jobs: {}'}]}}}
    response=client.post('/mcp',json=call)
    assert response.status_code==401 and 'resource_metadata=' in response.headers['WWW-Authenticate']

def test_openai_domain_challenge_is_explicit_and_fail_closed(bridge, monkeypatch):
    app,_,_=bridge; client=app.test_client()
    monkeypatch.delenv('THRESH_OPENAI_DOMAIN_TOKEN',raising=False)
    assert client.get('/.well-known/openai-apps-challenge').status_code==404
    monkeypatch.setenv('THRESH_OPENAI_DOMAIN_TOKEN','synthetic_openai_domain_token_123456789')
    response=client.get('/.well-known/openai-apps-challenge')
    assert response.status_code==200 and response.get_data(as_text=True)=='synthetic_openai_domain_token_123456789'

def test_oauth_login_returns_only_to_fixed_resume_route():
    from dredge.auth import studio_return_path
    app=Flask(__name__); app.secret_key='synthetic-cookie'
    with app.test_request_context('/'):
        session['thresh_oauth_pending']={'redirect_uri':'https://attacker.invalid'}
        assert studio_return_path()=='/oauth/authorize/resume'
