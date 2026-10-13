"""PayPal provider contract: account binding, payment proof, durable retries, and CSRF."""
import copy
import json
import sqlite3
import time
from datetime import datetime, timezone
from unittest.mock import Mock
from types import SimpleNamespace

import pytest
from test_studio import app, client_for

from dredge.paypal_billing import PayPalError, register_paypal_billing


def stamp(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat().replace('+00:00', 'Z')


def plan():
    return {'id': 'P-TEST', 'status': 'ACTIVE', 'quantity_supported': False,
            'billing_cycles': [{'tenure_type': 'REGULAR', 'sequence': 1, 'total_cycles': 0,
                'frequency': {'interval_unit': 'MONTH', 'interval_count': 1},
                'pricing_scheme': {'fixed_price': {'currency_code': 'USD', 'value': '19.00'}}}],
            'payment_preferences': {'setup_fee': {'currency_code': 'USD', 'value': '0'}},
            'taxes': {'percentage': '0'}}


class FakePayPal:
    mode = 'sandbox'

    def __init__(self):
        self.plans = plan()
        self.sub = None
        self.transactions = []
        self.create_calls = []
        self.retrieve_calls = []
        self.cancel_calls = []
        self.verified = True
        self.fail_create_once = False

    def get_plan(self, plan_id):
        return copy.deepcopy(self.plans)

    def create_subscription(self, payload, request_id):
        self.create_calls.append((copy.deepcopy(payload), request_id))
        if self.sub is None:
            self.sub = {'id': 'I-TEST', 'plan_id': payload['plan_id'], 'custom_id': payload['custom_id'],
                        'status': 'APPROVAL_PENDING', 'quantity': '1', 'plan_overridden': False,
                        'links': [{'rel': 'approve', 'href': 'https://www.sandbox.paypal.com/webapps/billing/subscriptions?ba_token=BA-TEST'}]}
        if self.fail_create_once:
            self.fail_create_once = False
            raise PayPalError('Synthetic timeout after provider creation')
        return copy.deepcopy(self.sub)

    def get_subscription(self, sid):
        self.retrieve_calls.append(sid)
        return copy.deepcopy(self.sub)

    def list_transactions(self, sid, start_time, end_time):
        return {'transactions': copy.deepcopy(self.transactions), 'total_pages': 1}

    def verify_webhook(self, raw_body, headers, webhook_id):
        return self.verified

    def cancel_subscription(self, sid):
        self.cancel_calls.append(sid)
        self.sub['status'] = 'CANCELLED'

    def paid(self, paid_at=None, next_bill=None):
        paid_at = paid_at or time.time() - 60
        next_bill = next_bill or time.time() + 27 * 86400
        self.sub.update(status='ACTIVE', billing_info={
            'last_payment': {'amount': {'currency_code': 'USD', 'value': '19.00'}, 'time': stamp(paid_at)},
            'next_billing_time': stamp(next_bill), 'failed_payments_count': 0,
            'outstanding_balance': {'currency_code': 'USD', 'value': '0.00'}})
        self.transactions = [{'id': 'SALE-TEST', 'status': 'COMPLETED', 'time': stamp(paid_at),
                              'amount_with_breakdown': {'gross_amount': {'currency_code': 'USD', 'value': '19.00'}}}]


@pytest.fixture
def paypal(app):
    if 'paypal_billing' not in app.blueprints:
        register_paypal_billing(app)
    app.config.update(PAYPAL_BILLING_ENABLED=True, PAYPAL_BILLING_MODE='sandbox',
                      PAYPAL_PLAN_ID='P-TEST', PAYPAL_WEBHOOK_ID='WH-TEST')
    sdk = FakePayPal()
    app.extensions['paypal_client'] = sdk
    return app, sdk


def checkout(app, sdk, user='test:op'):
    customer, headers = client_for(app, user)
    response = customer.post('/api/billing/paypal/checkout', headers=headers)
    assert response.status_code == 200, response.json
    return customer, headers


def event(app, event_id='EV-TEST', kind='BILLING.SUBSCRIPTION.UPDATED', resource=None):
    return app.test_client().post('/api/billing/paypal/webhook', json={
        'id': event_id, 'event_type': kind, 'resource': resource or {'id': 'I-TEST'}}, headers={
        'PAYPAL-TRANSMISSION-ID': 'TRANSMISSION-TEST', 'PAYPAL-TRANSMISSION-TIME': stamp(time.time()),
        'PAYPAL-TRANSMISSION-SIG': 'synthetic-signature', 'PAYPAL-AUTH-ALGO': 'SHA256withRSA',
        'PAYPAL-CERT-URL': 'https://api.sandbox.paypal.com/v1/notifications/certs/CERT-TEST'})


def test_create_is_server_bound_and_return_does_not_grant_access(paypal):
    app, sdk = paypal
    customer, headers = checkout(app, sdk)
    assert not customer.get('/api/billing/paypal/status').json['subscribed']
    customer.get('/billing?paypal=returned&subscription_id=I-OTHER')
    assert not customer.get('/api/billing/paypal/status').json['subscribed']
    assert customer.post('/api/billing/paypal/checkout', headers=headers).status_code == 200
    assert len(sdk.create_calls) == 1
    payload, request_id = sdk.create_calls[0]
    assert payload['plan_id'] == 'P-TEST' and payload['quantity'] == '1'
    assert payload['custom_id'] != 'test:op'
    assert payload['application_context']['return_url'] == 'https://dredgeoriongateway.com/billing?paypal=returned'
    assert len(request_id) == 36
    assert 'subscriber' not in payload


def test_paid_webhook_replay_reordered_and_expiration(paypal):
    app, sdk = paypal
    customer, _ = checkout(app, sdk)
    sdk.paid()
    assert event(app).status_code == 200
    assert customer.get('/api/billing/paypal/status').json['subscribed']
    calls = len(sdk.retrieve_calls)
    assert event(app).json['duplicate']
    assert len(sdk.retrieve_calls) == calls
    sdk.sub['status'] = 'SUSPENDED'
    assert event(app, 'EV-LATE', 'BILLING.SUBSCRIPTION.ACTIVATED').status_code == 200
    assert not customer.get('/api/billing/paypal/status').json['subscribed']
    sdk.paid(paid_at=time.time()-40*86400, next_bill=time.time()-86400)
    assert event(app, 'EV-OLD-PAYMENT').status_code == 200
    assert not customer.get('/api/billing/paypal/status').json['subscribed']
    assert customer.get('/api/studio/session').json['role'] == 'operator'


@pytest.mark.parametrize('change', ['unpaid', 'pending', 'refunded', 'wrong_currency', 'wrong_amount',
    'old_payment', 'future_payment', 'outstanding', 'failed_payment', 'wrong_plan', 'override', 'binding', 'quantity'])
def test_unproven_or_mismatched_subscription_never_grants(paypal, change):
    app, sdk = paypal
    customer, _ = checkout(app, sdk)
    sdk.paid()
    if change == 'unpaid': sdk.sub['billing_info'].pop('last_payment')
    if change == 'pending': sdk.transactions[0]['status'] = 'PENDING'
    if change == 'refunded': sdk.transactions[0]['status'] = 'REFUNDED'
    if change == 'wrong_currency': sdk.transactions[0]['amount_with_breakdown']['gross_amount']['currency_code'] = 'EUR'
    if change == 'wrong_amount': sdk.transactions[0]['amount_with_breakdown']['gross_amount']['value'] = '1.00'
    if change == 'old_payment': sdk.paid(paid_at=time.time()-60*86400)
    if change == 'future_payment': sdk.paid(paid_at=time.time()+86400)
    if change == 'outstanding': sdk.sub['billing_info']['outstanding_balance']['value'] = '19.00'
    if change == 'failed_payment': sdk.sub['billing_info']['failed_payments_count'] = 1
    if change == 'wrong_plan': sdk.sub['plan_id'] = 'P-OTHER'
    if change == 'override': sdk.sub['plan_overridden'] = True
    if change == 'binding': sdk.sub['custom_id'] = 'different-account-binding'
    if change == 'quantity': sdk.sub['quantity'] = '2'
    assert event(app).status_code == 200
    assert not customer.get('/api/billing/paypal/status').json['subscribed']


def test_cancellation_keeps_proven_paid_term_and_cannot_resurrect_on_late_event(paypal):
    app, sdk = paypal
    customer, headers = checkout(app, sdk)
    sdk.paid()
    assert event(app).status_code == 200
    response = customer.post('/api/billing/paypal/cancel', headers=headers)
    assert response.status_code == 200
    assert sdk.cancel_calls == ['I-TEST']
    state = customer.get('/api/billing/paypal/status').json
    assert state['subscribed'] and state['subscriptions'][0]['cancel_at_period_end']
    assert not state['can_cancel']
    sdk.sub['billing_info'].pop('next_billing_time')
    assert event(app, 'EV-LATE', 'BILLING.SUBSCRIPTION.ACTIVATED').status_code == 200
    assert customer.get('/api/billing/paypal/status').json['subscribed']
    with app.extensions['studio_store'].connect() as db:
        db.execute('UPDATE paypal_subscriptions SET period_end=?', (time.time()-1,))
    assert not customer.get('/api/billing/paypal/status').json['subscribed']


def test_cross_account_cannot_reconcile_or_cancel_client_supplied_id(paypal):
    app, sdk = paypal
    customer, _ = checkout(app, sdk)
    sdk.paid()
    other, headers = client_for(app, 'test:other')
    before = len(sdk.retrieve_calls)
    for path in ('reconcile', 'cancel'):
        result = other.post('/api/billing/paypal/'+path, headers=headers, json={'subscription_id':'I-TEST'})
        assert result.status_code == 404
    assert len(sdk.retrieve_calls) == before and not sdk.cancel_calls
    assert not other.get('/api/billing/paypal/status').json['subscribed']
    assert event(app).status_code == 200
    assert customer.get('/api/billing/paypal/status').json['subscribed']


def test_auth_csrf_signature_mode_and_disabled_configuration(paypal):
    app, sdk = paypal
    anonymous = app.test_client()
    assert anonymous.post('/api/billing/paypal/checkout').status_code == 401
    customer, headers = client_for(app)
    for route in ('checkout', 'cancel', 'reconcile'):
        assert customer.post('/api/billing/paypal/'+route).status_code == 403
    assert anonymous.post('/api/billing/paypal/webhook', json={}).status_code == 400
    sdk.verified = False
    assert event(app).status_code == 400
    app.config['PAYPAL_BILLING_MODE'] = 'live'
    assert customer.post('/api/billing/paypal/checkout', headers=headers).status_code == 503
    app.config['PAYPAL_BILLING_MODE'] = 'sandbox'
    app.config['PAYPAL_BILLING_ENABLED'] = False
    assert customer.post('/api/billing/paypal/checkout', headers=headers).status_code == 503
    assert event(app).status_code == 503


@pytest.mark.parametrize('field', ['amount', 'currency', 'interval', 'count', 'trial', 'setup', 'tax', 'quantity', 'inactive'])
def test_exact_price_plan_validation(paypal, field):
    app, sdk = paypal
    cycle = sdk.plans['billing_cycles'][0]
    if field == 'amount': cycle['pricing_scheme']['fixed_price']['value'] = '20'
    if field == 'currency': cycle['pricing_scheme']['fixed_price']['currency_code'] = 'EUR'
    if field == 'interval': cycle['frequency']['interval_unit'] = 'YEAR'
    if field == 'count': cycle['frequency']['interval_count'] = 2
    if field == 'trial': sdk.plans['billing_cycles'].append({'tenure_type': 'TRIAL'})
    if field == 'setup': sdk.plans['payment_preferences']['setup_fee']['value'] = '1'
    if field == 'tax': sdk.plans['taxes']['percentage'] = '10'
    if field == 'quantity': sdk.plans['quantity_supported'] = True
    if field == 'inactive': sdk.plans['status'] = 'INACTIVE'
    customer, headers = client_for(app)
    assert customer.post('/api/billing/paypal/checkout', headers=headers).status_code == 503
    assert not sdk.create_calls


def test_create_timeout_retains_durable_idempotency_and_opaque_binding(paypal):
    app, sdk = paypal
    customer, headers = client_for(app)
    sdk.fail_create_once = True
    assert customer.post('/api/billing/paypal/checkout', headers=headers).status_code == 502
    with app.extensions['studio_store'].connect() as db:
        intent = dict(db.execute('SELECT * FROM paypal_checkout_intents').fetchone())
    assert intent['subscription_id'] is None
    assert customer.post('/api/billing/paypal/checkout', headers=headers).status_code == 200
    assert sdk.create_calls[0] == sdk.create_calls[1]
    assert sdk.create_calls[0][1] == intent['request_id']


def test_uncertain_create_past_provider_idempotency_window_fails_closed(paypal):
    app, sdk = paypal
    customer, headers = client_for(app)
    sdk.fail_create_once = True
    assert customer.post('/api/billing/paypal/checkout', headers=headers).status_code == 502
    with app.extensions['studio_store'].connect() as db:
        db.execute('UPDATE paypal_checkout_intents SET created=?', (time.time()-73*3600,))
    assert customer.post('/api/billing/paypal/checkout', headers=headers).status_code == 409
    assert len(sdk.create_calls) == 1


def test_webhook_recovers_create_timeout_using_authoritative_server_binding(paypal):
    app, sdk = paypal
    customer, headers = client_for(app)
    sdk.fail_create_once = True
    assert customer.post('/api/billing/paypal/checkout', headers=headers).status_code == 502
    sdk.paid()
    assert event(app).status_code == 200
    assert customer.get('/api/billing/paypal/status').json['subscribed']
    with app.extensions['studio_store'].connect() as db:
        assert db.execute('SELECT subscription_id FROM paypal_checkout_intents').fetchone()[0] == 'I-TEST'


def test_reconciliation_holds_sqlite_write_lock(paypal):
    app, sdk = paypal
    customer, headers = checkout(app, sdk)
    sdk.paid()
    original = sdk.get_subscription
    def retrieve(sid):
        db = sqlite3.connect(app.extensions['studio_store'].path, timeout=0)
        try:
            with pytest.raises(sqlite3.OperationalError, match='locked'):
                db.execute('BEGIN IMMEDIATE')
        finally:
            db.close()
        return original(sid)
    sdk.get_subscription = retrieve
    assert customer.post('/api/billing/paypal/reconcile', headers=headers).status_code == 200
    assert customer.get('/api/billing/paypal/status').json['subscribed']


def test_stripe_pending_or_active_prevents_paypal_checkout(paypal):
    app, sdk = paypal
    customer, headers = client_for(app)
    with app.extensions['studio_store'].connect() as db:
        db.execute('INSERT INTO billing_customers(owner,mode,customer,checkout_id) VALUES(?,?,?,?)',
                   ('test:op', 'sandbox', 'cus_pending', 'cs_pending'))
    assert customer.post('/api/billing/paypal/checkout', headers=headers).status_code == 409
    assert not sdk.create_calls
    with app.extensions['studio_store'].connect() as db:
        db.execute('UPDATE billing_customers SET checkout_id=NULL')
        db.execute('INSERT INTO billing_subscriptions VALUES(?,?,?,?,?,?)',
                   ('sub_active', 'test:op', 'sandbox', 'active', time.time()+3600, 0))
    assert customer.post('/api/billing/paypal/checkout', headers=headers).status_code == 409
    assert not sdk.create_calls


def test_pending_intent_blocks_stripe_even_after_network_failure(paypal):
    from dredge.billing import subscription_conflict
    app, sdk = paypal
    customer, headers = client_for(app)
    sdk.fail_create_once = True
    assert customer.post('/api/billing/paypal/checkout', headers=headers).status_code == 502
    with app.app_context(), app.extensions['studio_store'].connect() as db:
        assert subscription_conflict(db, 'test:op', 'stripe')
        assert not subscription_conflict(db, 'test:other', 'stripe')


def test_unknown_subscription_and_forged_binding_cannot_claim_entitlement(paypal):
    app, sdk = paypal
    customer, _ = checkout(app, sdk)
    sdk.paid()
    sdk.sub['id'] = 'I-OTHER'
    assert event(app, resource={'id': 'I-OTHER', 'custom_id': sdk.sub['custom_id']}).json['ignored']
    assert not customer.get('/api/billing/paypal/status').json['subscribed']


def test_wrong_plan_or_binding_revokes_previously_granted_access(paypal):
    app, sdk = paypal
    customer, _ = checkout(app, sdk)
    sdk.paid()
    assert event(app).status_code == 200
    assert customer.get('/api/billing/paypal/status').json['subscribed']
    sdk.sub['custom_id'] = 'different-binding'
    assert event(app, 'EV-CHANGED-BINDING').status_code == 200
    assert not customer.get('/api/billing/paypal/status').json['subscribed']


def test_signed_refund_cannot_be_undone_by_stale_completed_payment(paypal):
    app, sdk = paypal
    customer, _ = checkout(app, sdk)
    sdk.paid()
    assert event(app).status_code == 200
    assert event(app, 'EV-REFUND', 'PAYMENT.SALE.REFUNDED', {'id': 'REFUND-TEST', 'sale_id': 'SALE-TEST'}).status_code == 200
    assert not customer.get('/api/billing/paypal/status').json['subscribed']
    assert event(app, 'EV-STALE-PAYMENT', 'PAYMENT.SALE.COMPLETED', {'id': 'SALE-TEST', 'billing_agreement_id': 'I-TEST'}).status_code == 200
    assert not customer.get('/api/billing/paypal/status').json['subscribed']


def test_provider_failure_does_not_consume_event_or_store_half_update(paypal):
    app, sdk = paypal
    customer, _ = checkout(app, sdk)
    sdk.paid()
    original = sdk.get_subscription
    sdk.get_subscription = Mock(side_effect=PayPalError('Synthetic provider outage'))
    assert event(app).status_code == 503
    with app.extensions['studio_store'].connect() as db:
        assert db.execute('SELECT COUNT(*) FROM paypal_events').fetchone()[0] == 0
    sdk.get_subscription = original
    assert event(app).status_code == 200
    assert customer.get('/api/billing/paypal/status').json['subscribed']


def test_untrusted_or_cross_environment_approval_url_is_not_returned(paypal):
    app, sdk = paypal
    customer, headers = checkout(app, sdk)
    for url in ('https://evil.example/pay', 'https://www.paypal.com/pay',
                'https://www.sandbox.paypal.com.evil.example/pay', 'https://user@www.sandbox.paypal.com/pay'):
        sdk.sub['links'][0]['href'] = url
        response = customer.post('/api/billing/paypal/checkout', headers=headers)
        assert response.status_code == 502
        assert 'url' not in response.json


def test_event_modes_json_size_and_id_paths_fail_before_provider_call(paypal):
    app, sdk = paypal
    sdk.verify_webhook = Mock(return_value=True)
    client = app.test_client()
    assert client.post('/api/billing/paypal/webhook', data=b' '*(1024*1024+1)).status_code == 413
    headers = {'PAYPAL-TRANSMISSION-ID': 'tx', 'PAYPAL-TRANSMISSION-TIME': stamp(time.time()),
               'PAYPAL-TRANSMISSION-SIG': 'sig', 'PAYPAL-AUTH-ALGO': 'SHA256withRSA',
               'PAYPAL-CERT-URL': 'https://api.paypal.com/v1/notifications/certs/CERT-TEST'}
    payload = {'id':'EV-TEST','event_type':'BILLING.SUBSCRIPTION.UPDATED','resource':{'id':'I-TEST'}}
    assert client.post('/api/billing/paypal/webhook', json=payload, headers=headers).status_code == 400
    headers['PAYPAL-CERT-URL'] = 'https://api.sandbox.paypal.com/v1/notifications/certs/CERT-TEST'
    duplicate = b'{"id":"EV-ONE","id":"EV-TWO","event_type":"X","resource":{}}'
    assert client.post('/api/billing/paypal/webhook', data=duplicate, headers=headers).status_code == 400
    assert not sdk.verify_webhook.called
    payload['resource']['id'] = 'I-TEST/../../other'
    assert client.post('/api/billing/paypal/webhook', json=payload, headers=headers).json['ignored']
    assert not sdk.retrieve_calls


def test_paypal_registration_requires_explicit_opt_in_and_never_calls_network(app, monkeypatch):
    from flask import Flask
    from dredge.paypal_billing import paypal_configured
    for name in ('PAYPAL_BILLING_ENABLED','PAYPAL_BILLING_MODE'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('PAYPAL_CLIENT_ID', 'fictional-client')
    monkeypatch.setenv('PAYPAL_CLIENT_SECRET', 'fictional-secret')
    monkeypatch.setenv('PAYPAL_PLAN_ID', 'P-TEST')
    monkeypatch.setenv('PAYPAL_WEBHOOK_ID', 'WH-TEST')
    request_mock = Mock(side_effect=AssertionError('Registration must not use network'))
    monkeypatch.setattr('requests.Session.request', request_mock)
    application = Flask('paypal-disabled')
    application.extensions['studio_store'] = app.extensions['studio_store']
    register_paypal_billing(application)
    with application.app_context():
        assert application.config['PAYPAL_BILLING_MODE'] == 'sandbox'
        assert not paypal_configured()
        assert 'paypal_client' not in application.extensions
    assert not request_mock.called


def test_rest_adapter_fixed_origins_idempotency_and_raw_signature_postback():
    from dredge.paypal_billing import PayPalClient
    sdk = PayPalClient('fictional-client', 'fictional-secret', 'sandbox')
    sdk._http = Mock()
    responses = [Mock(status_code=200), Mock(status_code=201), Mock(status_code=200)]
    responses[0].json.return_value = {'access_token':'fictional-token','expires_in':3600}
    responses[1].json.return_value = {'id':'I-TEST'}
    responses[2].json.return_value = {'verification_status':'SUCCESS'}
    sdk._http.request.side_effect = responses
    sdk.create_subscription({'plan_id':'P-TEST'}, 'durable-request')
    raw = b'{ "id": "EV-TEST", "resource": { "amount": 19.00 }, "event_type":"X" }\n'
    headers = {'PAYPAL-TRANSMISSION-ID': 'tx', 'PAYPAL-TRANSMISSION-TIME': stamp(time.time()),
               'PAYPAL-TRANSMISSION-SIG': 'sig', 'PAYPAL-AUTH-ALGO': 'SHA256withRSA',
               'PAYPAL-CERT-URL': 'https://api.sandbox.paypal.com/v1/notifications/certs/CERT-TEST'}
    assert sdk.verify_webhook(raw, headers, 'WH-TEST')
    calls = sdk._http.request.call_args_list
    assert len(calls) == 3
    assert calls[0].args == ('POST', 'https://api-m.sandbox.paypal.com/v1/oauth2/token')
    assert calls[1].kwargs['headers']['PayPal-Request-Id'] == 'durable-request'
    assert calls[2].args == ('POST', 'https://api-m.sandbox.paypal.com/v1/notifications/verify-webhook-signature')
    assert b'"webhook_event":' + raw + b'}' in calls[2].kwargs['data']
    assert json.loads(calls[2].kwargs['data'])['webhook_id'] == 'WH-TEST'
    assert all(call.kwargs['allow_redirects'] is False for call in calls)
    assert all(call.kwargs['timeout'] == (5,15) for call in calls)
    assert not any(call.args[1] == headers['PAYPAL-CERT-URL'] for call in calls)
    with pytest.raises(PayPalError):
        sdk.get_subscription('I-TEST/../../other')
    assert len(sdk._http.request.call_args_list) == 3


def test_rest_adapter_does_not_follow_redirect_or_log_response_secrets():
    from dredge.paypal_billing import PayPalClient
    sdk = PayPalClient('fictional-client', 'fictional-secret', 'live')
    sdk._http = Mock()
    sdk._http.request.return_value = Mock(status_code=302, text='sensitive-provider-response')
    with pytest.raises(PayPalError) as error:
        sdk.get_plan('P-TEST')
    assert 'sensitive-provider-response' not in str(error.value)
    assert sdk._http.request.call_args.args[1].startswith('https://api-m.paypal.com/')


def test_concurrent_checkout_reserves_one_subscription(paypal):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    app, sdk = paypal
    first, first_headers = client_for(app)
    second, second_headers = client_for(app)
    start = threading.Barrier(2)
    def create(client, headers):
        start.wait(timeout=3)
        return client.post('/api/billing/paypal/checkout', headers=headers).status_code
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(lambda pair: create(*pair), [(first,first_headers),(second,second_headers)]))
    assert results == [200,200]
    assert len(sdk.create_calls) == 1
    with app.extensions['studio_store'].connect() as db:
        assert db.execute('SELECT COUNT(*) FROM paypal_checkout_intents').fetchone()[0] == 1
        assert db.execute('SELECT COUNT(*) FROM paypal_subscriptions').fetchone()[0] == 1


@pytest.mark.parametrize('field', ['custom_id', 'id', 'plan_id'])
def test_cancel_rechecks_authoritative_binding_before_external_mutation(paypal, field):
    app, sdk = paypal
    customer, headers = checkout(app, sdk)
    sdk.paid()
    sdk.sub[field] = 'changed-binding'
    assert customer.post('/api/billing/paypal/cancel', headers=headers).status_code == 409
    assert not sdk.cancel_calls


def test_refund_arriving_before_payment_mapping_is_retained(paypal):
    app, sdk = paypal
    # The refund omits a subscription ID and arrives before the completed sale.
    assert event(app, 'EV-EARLY-REFUND', 'PAYMENT.SALE.REFUNDED', {'id':'REFUND-TEST','sale_id':'SALE-TEST'}).status_code == 200
    customer, _ = checkout(app, sdk)
    sdk.paid()
    assert event(app, 'EV-LATE-COMPLETED', 'PAYMENT.SALE.COMPLETED', {'id':'SALE-TEST','billing_agreement_id':'I-TEST'}).status_code == 200
    assert not customer.get('/api/billing/paypal/status').json['subscribed']
    with app.extensions['studio_store'].connect() as db:
        assert db.execute("SELECT status FROM paypal_payments WHERE id='SALE-TEST'").fetchone()[0] == 'REFUNDED'


@pytest.mark.parametrize('value', [None, {}, [None], 'bad-shape'])
def test_malformed_plan_shapes_fail_closed(paypal, value):
    app, sdk = paypal
    sdk.plans['billing_cycles'] = value
    customer, headers = client_for(app)
    assert customer.post('/api/billing/paypal/checkout', headers=headers).status_code == 503
    assert not sdk.create_calls


def test_malformed_certificate_url_is_rejected_without_network(paypal):
    app, sdk = paypal
    sdk.verify_webhook = Mock()
    result = app.test_client().post('/api/billing/paypal/webhook', json={
        'id':'EV-TEST','event_type':'BILLING.SUBSCRIPTION.UPDATED','resource':{'id':'I-TEST'}}, headers={
        'PAYPAL-TRANSMISSION-ID':'tx','PAYPAL-TRANSMISSION-TIME':stamp(time.time()),
        'PAYPAL-TRANSMISSION-SIG':'sig','PAYPAL-AUTH-ALGO':'SHA256withRSA',
        'PAYPAL-CERT-URL':'https://['})
    assert result.status_code == 400
    assert not sdk.verify_webhook.called


def test_timeout_retry_after_remote_approval_preserves_binding_without_approval_link(paypal):
    app, sdk = paypal
    customer, headers = client_for(app)
    sdk.fail_create_once = True
    assert customer.post('/api/billing/paypal/checkout', headers=headers).status_code == 502
    sdk.paid()
    sdk.sub['links'] = []
    assert customer.post('/api/billing/paypal/checkout', headers=headers).status_code == 409
    assert customer.post('/api/billing/paypal/reconcile', headers=headers).status_code == 200
    assert customer.get('/api/billing/paypal/status').json['subscribed']
    assert sdk.create_calls[0] == sdk.create_calls[1]


def test_environment_and_provider_event_ids_are_namespaced(paypal):
    app, sdk = paypal
    customer, _ = checkout(app, sdk)
    sdk.paid()
    with app.extensions['studio_store'].connect() as db:
        db.execute('INSERT INTO paypal_events VALUES(?,?,?,?)', ('EV-TEST','live','test',time.time()))
        db.execute('INSERT INTO billing_events VALUES(?,?,?,?)', ('EV-TEST','sandbox','test',time.time()))
    assert event(app).status_code == 200
    assert customer.get('/api/billing/paypal/status').json['subscribed']


def test_checkout_ignores_client_owner_plan_price_and_subscription_ids(paypal):
    app, sdk = paypal
    customer, headers = client_for(app)
    result = customer.post('/api/billing/paypal/checkout', headers=headers, json={
        'owner':'test:other','plan_id':'P-OTHER','custom_id':'attacker','subscription_id':'I-OTHER','quantity':'100'})
    assert result.status_code == 200
    payload = sdk.create_calls[0][0]
    assert payload['plan_id'] == 'P-TEST' and payload['quantity'] == '1'
    assert payload['custom_id'] != 'attacker'
    with app.extensions['studio_store'].connect() as db:
        assert db.execute('SELECT owner FROM paypal_subscriptions').fetchone()[0] == 'test:op'


def test_price_config_change_cannot_reuse_pending_request_for_different_body(paypal):
    app, sdk = paypal
    customer, headers = client_for(app)
    sdk.fail_create_once = True
    assert customer.post('/api/billing/paypal/checkout', headers=headers).status_code == 502
    app.config['PAYPAL_PLAN_ID'] = 'P-NEW'
    sdk.plans['id'] = 'P-NEW'
    assert customer.post('/api/billing/paypal/checkout', headers=headers).status_code == 409
    assert len(sdk.create_calls) == 1


def test_month_end_anchor_preserves_fully_paid_february_to_march_term(paypal, monkeypatch):
    app, sdk = paypal
    customer, headers = checkout(app, sdk)
    paid_at = datetime(2026, 2, 28, 12, tzinfo=timezone.utc).timestamp()
    next_bill = datetime(2026, 3, 31, 12, tzinfo=timezone.utc).timestamp()
    current = datetime(2026, 3, 30, 12, tzinfo=timezone.utc).timestamp()
    monkeypatch.setattr('dredge.paypal_billing.time', SimpleNamespace(time=lambda: current))
    sdk.paid(paid_at=paid_at, next_bill=next_bill)
    assert customer.post('/api/billing/paypal/reconcile', headers=headers).status_code == 200
    state = customer.get('/api/billing/paypal/status').json
    assert state['subscribed']
    assert state['subscriptions'][0]['period_end'] == next_bill


def test_unpaid_future_next_billing_time_cannot_extend_one_paid_month(paypal):
    app, sdk = paypal
    customer, headers = checkout(app, sdk)
    paid_at = time.time()-60
    sdk.paid(paid_at=paid_at, next_bill=time.time()+90*86400)
    assert customer.post('/api/billing/paypal/reconcile', headers=headers).status_code == 200
    state = customer.get('/api/billing/paypal/status').json
    assert state['subscriptions'][0]['period_end'] == pytest.approx(paid_at+31*86400, abs=0.001)
