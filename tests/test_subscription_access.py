"""Provider-neutral account entitlements never grant roles or paid AI usage."""
import time
import pytest
from test_studio import app, client_for, proposal
from test_billing import billing, event
from dredge.billing import subscription_access


def seed_paypal(app, owner='test:op', mode='live', status='active', end=None, cancel=0):
    with app.extensions['studio_store'].connect() as db:
        db.execute('INSERT INTO paypal_subscriptions(id,owner,mode,status,period_end,cancel_at_period_end,binding,plan_id,provider_status) VALUES(?,?,?,?,?,?,?,?,?)',
                   ('I-FICTIONAL', owner, mode, status, end or time.time()+3600, cancel, 'binding', 'P-TEST', 'ACTIVE'))


@pytest.mark.parametrize('provider', ['stripe', 'paypal'])
def test_live_paid_access_is_owner_bound_and_roles_unchanged(app, provider):
    app.config.update(STRIPE_BILLING_MODE='live', PAYPAL_BILLING_MODE='live')
    if provider == 'stripe':
        with app.extensions['studio_store'].connect() as db:
            db.execute('INSERT INTO billing_subscriptions VALUES(?,?,?,?,?,?)',
                       ('sub', 'test:viewer', 'live', 'active', time.time()+3600, 0))
    else:
        seed_paypal(app, owner='test:viewer')
    with app.app_context():
        assert subscription_access('test:viewer') == {'subscription_status':'active','paid_access':True,'paid_ai_included':False}
        assert not subscription_access('test:other')['paid_access']
    viewer, headers = client_for(app, 'test:viewer')
    assert viewer.get('/api/studio/session').json['role'] == 'viewer'
    assert proposal(viewer, headers).status_code == 403


@pytest.mark.parametrize('status', ['trialing','past_due','payment_pending','suspended','canceled','expired'])
def test_non_paid_status_never_grants_live_access(app, status):
    app.config['PAYPAL_BILLING_MODE'] = 'live'
    seed_paypal(app, status=status)
    with app.app_context():
        assert not subscription_access('test:op')['paid_access']


def test_sandbox_expiry_and_paid_cancellation_boundaries(app):
    seed_paypal(app, mode='sandbox')
    with app.app_context():
        assert subscription_access('test:op')['subscription_status'] == 'sandbox_only'
    app.config['PAYPAL_BILLING_MODE'] = 'live'
    seed_paypal(app, end=time.time()-1, cancel=1)
    with app.app_context():
        assert not subscription_access('test:op')['paid_access']
    with app.extensions['studio_store'].connect() as db:
        db.execute("UPDATE paypal_subscriptions SET period_end=? WHERE mode='live'", (time.time()+60,))
    with app.app_context():
        assert subscription_access('test:op')['paid_access']


def test_sandbox_payment_cannot_bypass_live_execution_gate(app):
    app.config.update(BILLING_REQUIRE_SUBSCRIPTION=True, STRIPE_BILLING_MODE='live')
    seed_paypal(app, mode='sandbox')
    customer, headers = client_for(app)
    assert proposal(customer, headers).status_code == 402


def test_stripe_checkout_blocks_other_provider_before_creation(billing):
    app, sdk = billing
    from types import SimpleNamespace
    sdk.v1.prices.retrieve.return_value = SimpleNamespace(id='price_test', livemode=False, currency='usd', unit_amount=1900,
        recurring=SimpleNamespace(interval='month',interval_count=1),active=True)
    seed_paypal(app, mode='sandbox')
    customer, headers = client_for(app)
    assert customer.post('/api/billing/checkout', headers=headers).status_code == 409
    sdk.v1.checkout.sessions.create.assert_not_called()


def test_stripe_reconcile_recovers_missed_webhook_and_ignores_client_owner(billing):
    app, sdk = billing
    with app.extensions['studio_store'].connect() as db:
        db.execute("UPDATE billing_customers SET checkout_id='cs_owned'")
    sdk.v1.checkout.sessions.retrieve.return_value = {'customer':'cus_test', 'client_reference_id':'test:op',
        'livemode':False, 'status':'complete', 'subscription':'sub_test'}
    customer, headers = client_for(app)
    assert customer.post('/api/billing/reconcile', json={'owner':'test:other','subscription':'sub_foreign'}, headers=headers).json['subscribed']
    other, other_headers = client_for(app, 'test:other')
    assert not other.post('/api/billing/reconcile', headers=other_headers).json['subscribed']
    assert sdk.v1.subscriptions.retrieve.call_count == 1


def test_stripe_reconcile_expired_checkout_unlocks_provider_switch(billing):
    app, sdk = billing
    with app.extensions['studio_store'].connect() as db:
        db.execute("UPDATE billing_customers SET checkout_id='cs_expired'")
    sdk.v1.checkout.sessions.retrieve.return_value = {'customer':'cus_test', 'client_reference_id':'test:op',
        'livemode':False, 'status':'expired'}
    customer, headers = client_for(app)
    assert customer.get('/api/billing/status').json['checkout_conflicts']['paypal']
    result = customer.post('/api/billing/reconcile', headers=headers)
    assert result.status_code == 200
    assert not result.json['checkout_conflicts']['paypal']


def test_reconciliation_does_not_transfer_existing_subscription(billing):
    app, sdk = billing
    assert event(app.test_client()).status_code == 200
    with app.extensions['studio_store'].connect() as db:
        db.execute('INSERT INTO billing_customers(owner,mode,customer) VALUES(?,?,?)', ('test:other','sandbox','cus_other'))
    sub = sdk.v1.subscriptions.retrieve.return_value
    sub['customer'] = 'cus_other'; sub['metadata']['dredge_owner'] = 'test:other'
    assert event(app.test_client(), 'evt_rebound').json['ignored']
    with app.extensions['studio_store'].connect() as db:
        record = db.execute('SELECT owner,status FROM billing_subscriptions').fetchone()
        assert dict(record) == {'owner':'test:op','status':'unverified'}


def test_stripe_reconcile_requires_csrf_and_rejects_wrong_checkout_binding(billing):
    app, sdk = billing
    customer, headers = client_for(app)
    assert customer.post('/api/billing/reconcile').status_code == 403
    with app.extensions['studio_store'].connect() as db:
        db.execute("UPDATE billing_customers SET checkout_id='cs_foreign'")
    sdk.v1.checkout.sessions.retrieve.return_value = {'customer':'cus_foreign','client_reference_id':'test:other','livemode':False,'status':'complete','subscription':'sub_foreign'}
    assert customer.post('/api/billing/reconcile', headers=headers).status_code == 409
    sdk.v1.subscriptions.retrieve.assert_not_called()


def stripe_price(sdk):
    from types import SimpleNamespace
    sdk.v1.prices.retrieve.return_value = SimpleNamespace(id='price_test', livemode=False, currency='usd', unit_amount=1900,
        recurring=SimpleNamespace(interval='month',interval_count=1),active=True)


def test_uncertain_stripe_create_keeps_reservation_and_identical_retry(billing):
    import stripe
    from types import SimpleNamespace
    app, sdk = billing
    stripe_price(sdk)
    customer, headers = client_for(app)
    sdk.v1.checkout.sessions.create.side_effect = stripe.APIConnectionError('Fictional timeout')
    assert customer.post('/api/billing/checkout', headers=headers).status_code == 502
    first = sdk.v1.checkout.sessions.create.call_args
    assert customer.get('/api/billing/status').json['checkout_conflicts']['paypal']
    # Restart-like configuration variation must not change the persisted payload.
    app.config['STRIPE_INTEGRATION_IDENTIFIER'] = 'different_after_restart'
    sdk.v1.checkout.sessions.create.side_effect = None
    sdk.v1.checkout.sessions.create.return_value = SimpleNamespace(id='cs_retry',url='https://checkout.stripe.com/retry')
    assert customer.post('/api/billing/checkout', headers=headers).status_code == 200
    assert sdk.v1.checkout.sessions.create.call_args == first


def test_old_uncertain_stripe_checkout_never_reuses_expired_idempotency_key(billing):
    app, sdk = billing
    stripe_price(sdk)
    with app.extensions['studio_store'].connect() as db:
        db.execute('INSERT INTO stripe_checkout_intents(owner,mode,request_id,created) VALUES(?,?,?,?)',
                   ('test:op','sandbox','old-request',time.time()-24*3600))
    customer, headers = client_for(app)
    assert customer.post('/api/billing/checkout', headers=headers).status_code == 409
    sdk.v1.checkout.sessions.create.assert_not_called()
    assert customer.get('/api/billing/status').json['checkout_conflicts']['paypal']


def test_stripe_customer_timeout_also_keeps_cross_provider_reservation(billing):
    import stripe
    app, sdk = billing
    stripe_price(sdk)
    customer, headers = client_for(app, 'test:other')
    sdk.v1.customers.create.side_effect = stripe.APIConnectionError('Fictional timeout')
    assert customer.post('/api/billing/checkout', headers=headers).status_code == 502
    assert customer.get('/api/billing/status').json['checkout_conflicts']['paypal']
    sdk.v1.checkout.sessions.create.assert_not_called()


def test_old_cancel_event_does_not_clear_new_checkout_reservation(billing):
    app, sdk = billing
    with app.extensions['studio_store'].connect() as db:
        db.execute("UPDATE billing_customers SET checkout_id='cs_new'")
        db.execute('INSERT INTO stripe_checkout_intents(owner,mode,request_id,created,checkout_id) VALUES(?,?,?,?,?)',
                   ('test:op','sandbox','new-request',time.time(),'cs_new'))
    sdk.v1.subscriptions.retrieve.return_value['status'] = 'canceled'
    assert event(app.test_client(), 'evt_old_cancel').status_code == 200
    with app.extensions['studio_store'].connect() as db:
        assert db.execute('SELECT checkout_id FROM billing_customers').fetchone()[0] == 'cs_new'
        assert db.execute('SELECT checkout_id FROM stripe_checkout_intents').fetchone()[0] == 'cs_new'
