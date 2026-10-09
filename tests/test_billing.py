"""Billing trust boundaries: signed events, account ownership, and replay safety."""
import hashlib
import hmac
import json
import time
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from test_studio import app, client_for

@pytest.fixture
def billing(app):
    app.config.update(STRIPE_PRICE_ID='price_test', STRIPE_WEBHOOK_SECRET='whsec_test', STRIPE_PORTAL_CONFIG='bpc_test')
    sdk = SimpleNamespace(v1=SimpleNamespace(subscriptions=Mock(), invoices=Mock(), prices=Mock(), customers=Mock(),
          checkout=SimpleNamespace(sessions=Mock()), billing_portal=SimpleNamespace(sessions=Mock())))
    app.extensions['stripe_client'] = sdk
    with app.extensions['studio_store'].connect() as db:
        db.execute('INSERT INTO billing_customers(owner,mode,customer) VALUES(?,?,?)', ('test:op','sandbox','cus_test'))
    sdk.v1.subscriptions.retrieve.return_value = {'id':'sub_test','customer':'cus_test','livemode':False,
        'metadata':{'dredge_owner':'test:op'},'items':{'data':[{'price':{'id':'price_test'},'current_period_end':time.time()+3600}]},
        'status':'active','latest_invoice':'in_test'}
    sdk.v1.invoices.retrieve.return_value = {'status':'paid'}
    return app, sdk

def event(client, event_id='evt_test', **extra):
    body = json.dumps({'id':event_id,'type':'customer.subscription.updated','livemode':False,
        'data':{'object':{'id':'sub_test'}}, **extra}).encode()
    timestamp = str(int(time.time()))
    signature = hmac.new(b'whsec_test', timestamp.encode()+b'.'+body, hashlib.sha256).hexdigest()
    return client.post('/api/billing/webhook', data=body, headers={'Stripe-Signature':f't={timestamp},v1={signature}'})

def test_signed_reconciliation_replay_and_reordered_events(billing):
    app, sdk = billing
    customer, _ = client_for(app)
    assert event(app.test_client()).status_code == 200
    assert customer.get('/api/billing/status').json['subscribed']
    assert event(app.test_client()).json['duplicate']
    assert sdk.v1.subscriptions.retrieve.call_count == 1
    sdk.v1.subscriptions.retrieve.return_value['status'] = 'canceled'
    assert event(app.test_client(), 'evt_delayed').status_code == 200
    assert not customer.get('/api/billing/status').json['subscribed']
    assert customer.get('/api/studio/session').json['role'] == 'operator'

@pytest.mark.parametrize('change', ['owner','price','invoice','missing_invoice'])
def test_unverified_entitlement_never_activates(billing, change):
    app, sdk = billing
    sub = sdk.v1.subscriptions.retrieve.return_value
    if change == 'owner': sub['metadata']['dredge_owner'] = 'test:other'
    if change == 'price': sub['items']['data'][0]['price']['id'] = 'price_other'
    if change == 'invoice': sdk.v1.invoices.retrieve.return_value['status'] = 'open'
    if change == 'missing_invoice': sub['latest_invoice'] = None
    assert event(app.test_client()).status_code == 200
    customer, _ = client_for(app)
    assert not customer.get('/api/billing/status').json['subscribed']

def test_signature_mode_csrf_and_portal_ownership(billing):
    app, sdk = billing
    anonymous = app.test_client()
    assert anonymous.post('/api/billing/webhook', data=b'{}').status_code == 400
    assert event(anonymous, livemode=True).status_code == 400
    assert event(anonymous, account='acct_foreign').status_code == 400
    customer, headers = client_for(app, 'test:other')
    assert customer.post('/api/billing/checkout').status_code == 403
    assert customer.post('/api/billing/portal', headers=headers).status_code == 404
    assert not sdk.v1.billing_portal.sessions.create.called
    assert not customer.get('/api/billing/status').json['enforcement']

def test_public_billing_is_safe_before_credentials(app):
    client = app.test_client()
    assert client.get('/billing?checkout=returned').status_code == 200
    signed, _ = client_for(app)
    assert not signed.get('/api/billing/status').json['configured']
    assert not signed.get('/api/billing/status').json['subscribed']

def test_checkout_reuses_session_and_rejects_displayed_price_mismatch(billing):
    app, sdk = billing
    customer, headers = client_for(app)
    price = SimpleNamespace(id='price_test', livemode=False, currency='usd', unit_amount=1900,
        recurring=SimpleNamespace(interval='month',interval_count=1),active=True)
    sdk.v1.prices.retrieve.return_value = price
    sdk.v1.checkout.sessions.create.return_value = SimpleNamespace(id='cs_test',url='https://checkout.stripe.com/test')
    sdk.v1.checkout.sessions.retrieve.return_value = SimpleNamespace(status='open',url='https://checkout.stripe.com/test')
    assert customer.post('/api/billing/checkout', headers=headers).status_code == 200
    assert customer.post('/api/billing/checkout', headers=headers).status_code == 200
    assert sdk.v1.checkout.sessions.create.call_count == 1
    params = sdk.v1.checkout.sessions.create.call_args.args[0]
    assert params['customer'] == 'cus_test'
    assert params['client_reference_id'] == 'test:op'
    assert params['success_url'].startswith('https://dredgeoriongateway.com/')
    price.unit_amount = 9900
    assert customer.post('/api/billing/checkout', headers=headers).status_code == 503
