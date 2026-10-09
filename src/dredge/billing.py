"""Account-bound Stripe subscriptions. Sandbox by default; billing never assigns roles."""
import hashlib
import hmac
import os
import secrets
import time
from functools import wraps
from urllib.parse import urlsplit

import stripe
from flask import Blueprint, current_app, jsonify, request, send_file, session
from flask_login import current_user, login_required

from .studio import STATIC, store

billing_bp = Blueprint('billing', __name__)
API_VERSION = '2026-08-26.dahlia'
EVENTS = {'checkout.session.completed', 'checkout.session.async_payment_succeeded',
          'customer.subscription.created', 'customer.subscription.updated',
          'customer.subscription.deleted', 'invoice.paid', 'invoice.payment_failed'}


def mode():
    return current_app.config['STRIPE_BILLING_MODE']


def client():
    return current_app.extensions.get('stripe_client')


def configured():
    return bool(client() and current_app.config['STRIPE_PRICE_ID'] and
                current_app.config['STRIPE_WEBHOOK_SECRET'] and current_app.config['STRIPE_PORTAL_CONFIG'])


def authenticated_mutation(fn):
    @wraps(fn)
    @login_required
    def wrapped(*args, **kwargs):
        token = session.get('studio_csrf', '')
        if not token or not hmac.compare_digest(token, request.headers.get('X-CSRF-Token', '')):
            return jsonify(error='Refresh billing before continuing.'), 403
        if not configured():
            return jsonify(error='Sandbox billing setup is awaiting secure configuration.'), 503
        try:
            return fn(*args, **kwargs)
        except stripe.StripeError:
            current_app.logger.warning('Stripe billing request failed; no payment details logged.')
            return jsonify(error='Stripe could not complete this request. Please try again.'), 502
    return wrapped


def has_subscription(owner):
    with store().connect() as db:
        rows = db.execute('SELECT status,period_end FROM billing_subscriptions WHERE owner=? AND mode=?',
                          (owner, mode())).fetchall()
    return any(r['status'] in {'active', 'trialing'} and r['period_end'] > time.time() for r in rows)


def require_subscription():
    # This opt-in gate is separate from identity/role authorization and never enabled in sandbox.
    if current_app.config['BILLING_REQUIRE_SUBSCRIPTION'] and not has_subscription(current_user.get_id()):
        return jsonify(error='An active subscription is required to create or execute runs.', code='subscription_required'), 402


@billing_bp.route('/billing')
def billing_page():
    return send_file(STATIC / 'studio_billing.html')


@billing_bp.route('/api/billing/status')
@login_required
def status():
    session.setdefault('studio_csrf', secrets.token_urlsafe(32))
    with store().connect() as db:
        subscriptions = db.execute('SELECT status,period_end,cancel_at_period_end FROM billing_subscriptions WHERE owner=? AND mode=?',
                                   (current_user.get_id(), mode())).fetchall()
        customer = db.execute('SELECT customer FROM billing_customers WHERE owner=? AND mode=?',
                              (current_user.get_id(), mode())).fetchone()
    return jsonify(mode=mode(), configured=configured(), amount_usd=19, interval='month',
                   subscribed=has_subscription(current_user.get_id()), can_manage=bool(customer and subscriptions),
                   subscriptions=[dict(r) for r in subscriptions], csrf_token=session['studio_csrf'],
                   enforcement=current_app.config['BILLING_REQUIRE_SUBSCRIPTION'])


@billing_bp.route('/api/billing/checkout', methods=['POST'])
@authenticated_mutation
def checkout():
    owner = current_user.get_id()
    price = client().v1.prices.retrieve(current_app.config['STRIPE_PRICE_ID'])
    if (bool(price.livemode) != (mode() == 'live') or price.currency != 'usd' or price.unit_amount != 1900
            or price.recurring.interval != 'month' or price.recurring.interval_count != 1 or not price.active):
        return jsonify(error='Billing price does not match the displayed monthly plan.'), 503
    origin = current_app.config['BILLING_ORIGIN']
    with store().connect() as db:
        db.execute('BEGIN IMMEDIATE')
        if db.execute("SELECT 1 FROM billing_subscriptions WHERE owner=? AND mode=? AND status NOT IN ('canceled','incomplete_expired')",
                      (owner, mode())).fetchone():
            return jsonify(error='Manage your existing subscription instead of starting another.'), 409
        row = db.execute('SELECT * FROM billing_customers WHERE owner=? AND mode=?', (owner, mode())).fetchone()
        if not row:
            key = hashlib.sha256((mode()+owner).encode()).hexdigest()
            customer = client().v1.customers.create({'metadata': {'dredge_owner': owner, 'app': 'dredge_studio'}},
                                                   options={'idempotency_key': 'dredge-customer-'+key})
            db.execute('INSERT INTO billing_customers(owner,mode,customer) VALUES(?,?,?)', (owner, mode(), customer.id))
            row = db.execute('SELECT * FROM billing_customers WHERE owner=? AND mode=?', (owner, mode())).fetchone()
        if row['checkout_id']:
            existing = client().v1.checkout.sessions.retrieve(row['checkout_id'])
            if existing.status == 'open':
                return jsonify(url=existing.url)
            if existing.status == 'complete':
                return jsonify(error='Checkout is complete. Wait for payment confirmation or refresh billing.'), 409
        generation = int(row['generation'])+1
        result = client().v1.checkout.sessions.create({
            'mode': 'subscription', 'customer': row['customer'], 'client_reference_id': owner,
            'line_items': [{'price': price.id, 'quantity': 1}],
            'subscription_data': {'billing_mode': {'type': 'flexible'}, 'metadata': {'dredge_owner': owner, 'app': 'dredge_studio'}},
            'integration_identifier': current_app.config['STRIPE_INTEGRATION_IDENTIFIER'],
            'success_url': origin+'/billing?checkout=returned', 'cancel_url': origin+'/billing?checkout=canceled'},
            options={'idempotency_key': 'dredge-checkout-'+hashlib.sha256((mode()+owner+str(generation)).encode()).hexdigest()})
        db.execute('UPDATE billing_customers SET checkout_id=?,generation=? WHERE owner=? AND mode=?',
                   (result.id, generation, owner, mode()))
    return jsonify(url=result.url)


@billing_bp.route('/api/billing/portal', methods=['POST'])
@authenticated_mutation
def portal():
    with store().connect() as db:
        row = db.execute('SELECT customer FROM billing_customers WHERE owner=? AND mode=?',
                         (current_user.get_id(), mode())).fetchone()
    if not row:
        return jsonify(error='No billing account exists for your sign-in.'), 404
    result = client().v1.billing_portal.sessions.create({'customer': row['customer'],
            'configuration': current_app.config['STRIPE_PORTAL_CONFIG'],
            'return_url': current_app.config['BILLING_ORIGIN']+'/billing'})
    return jsonify(url=result.url)


def subscription_id(kind, obj):
    if kind.startswith('customer.subscription.'):
        return obj['id']
    if kind.startswith('checkout.session.'):
        return obj.get('subscription')
    return obj.get('parent', {}).get('subscription_details', {}).get('subscription') or obj.get('subscription')


@billing_bp.route('/api/billing/webhook', methods=['POST'])
def webhook():
    secret = current_app.config['STRIPE_WEBHOOK_SECRET']
    if not configured():
        return jsonify(error='Billing is not configured.'), 503
    if request.content_length and request.content_length > 1024*1024:
        return jsonify(error='Event too large.'), 413
    try:
        event = stripe.Webhook.construct_event(request.get_data(), request.headers.get('Stripe-Signature', ''), secret).to_dict()
    except (ValueError, stripe.SignatureVerificationError):
        return jsonify(error='Invalid event signature.'), 400
    if bool(event.get('livemode')) != (mode() == 'live') or event.get('account'):
        return jsonify(error='Wrong billing environment.'), 400
    if event['type'] not in EVENTS:
        return jsonify(received=True)
    obj = event['data']['object']
    sid = subscription_id(event['type'], obj)
    if not sid:
        return jsonify(received=True)
    try:
        with store().connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT 1 FROM billing_events WHERE id=? AND mode=?', (event['id'], mode())).fetchone():
                return jsonify(received=True, duplicate=True)
            # Retrieve current state under the write lock: delayed/reordered events cannot resurrect access.
            sub = client().v1.subscriptions.retrieve(sid)
            if hasattr(sub, 'to_dict'):
                sub = sub.to_dict()
            customer = sub['customer']
            row = db.execute('SELECT owner FROM billing_customers WHERE customer=? AND mode=?', (customer, mode())).fetchone()
            owner = sub.get('metadata', {}).get('dredge_owner')
            items = sub['items']['data']
            if (not row or row['owner'] != owner or bool(sub['livemode']) != (mode() == 'live')
                    or not items or any(i['price']['id'] != current_app.config['STRIPE_PRICE_ID'] for i in items)):
                return jsonify(received=True, ignored=True)
            end = max(i.get('current_period_end', sub.get('current_period_end', 0)) for i in items)
            state = sub['status']
            # Active state alone is insufficient if an initial invoice remains unpaid.
            invoice = sub.get('latest_invoice')
            if state == 'active' and not invoice:
                state = 'payment_pending'
            if state == 'active' and invoice:
                bill = client().v1.invoices.retrieve(invoice if isinstance(invoice, str) else invoice['id'])
                if bill['status'] != 'paid':
                    state = 'payment_pending'
            db.execute('INSERT INTO billing_subscriptions(id,owner,mode,status,period_end,cancel_at_period_end) VALUES(?,?,?,?,?,?) '
                       'ON CONFLICT(id,mode) DO UPDATE SET status=excluded.status,period_end=excluded.period_end,cancel_at_period_end=excluded.cancel_at_period_end',
                       (sub['id'], owner, mode(), state, end, int(sub.get('cancel_at_period_end', False))))
            db.execute('INSERT INTO billing_events(id,mode,kind,received) VALUES(?,?,?,?)', (event['id'], mode(), event['type'], time.time()))
            store().audit(db, 'stripe', 'subscription_updated', detail={'mode': mode(), 'owner': owner, 'status': state})
    except stripe.StripeError:
        return jsonify(error='Subscription reconciliation failed; retry this event.'), 503
    return jsonify(received=True)


def register_billing(app):
    billing_mode = os.environ.get('STRIPE_BILLING_MODE', 'sandbox')
    if billing_mode not in {'sandbox', 'live'}:
        raise RuntimeError('STRIPE_BILLING_MODE must be sandbox or live.')
    origin = os.environ.get('BILLING_ORIGIN', 'https://dredgeoriongateway.com').rstrip('/')
    parsed = urlsplit(origin)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.path or parsed.query or parsed.fragment or parsed.username:
        raise RuntimeError('BILLING_ORIGIN must be a trusted HTTPS origin.')
    app.config.update(STRIPE_BILLING_MODE=billing_mode, BILLING_ORIGIN=origin,
                      STRIPE_PRICE_ID=os.environ.get('STRIPE_PRICE_ID', ''),
                      STRIPE_WEBHOOK_SECRET=os.environ.get('STRIPE_WEBHOOK_SECRET', ''),
                      STRIPE_PORTAL_CONFIG=os.environ.get('STRIPE_PORTAL_CONFIG', ''),
                      STRIPE_INTEGRATION_IDENTIFIER='dredge_studio_'+''.join(secrets.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(8)),
                      BILLING_REQUIRE_SUBSCRIPTION=billing_mode == 'live' and os.environ.get('BILLING_REQUIRE_SUBSCRIPTION') == 'true')
    key = os.environ.get('STRIPE_RESTRICTED_KEY', '')
    if key:
        expected = ('rk_live_', 'sk_live_') if billing_mode == 'live' else ('rk_test_', 'sk_test_')
        if not key.startswith(expected):
            raise RuntimeError('Stripe key does not match billing environment.')
        app.extensions['stripe_client'] = stripe.StripeClient(key, stripe_version=API_VERSION, max_network_retries=2, http_client=stripe.RequestsClient(timeout=15))
    with app.extensions['studio_store'].connect() as db:
        db.executescript('''
        CREATE TABLE IF NOT EXISTS billing_customers(owner TEXT,mode TEXT,customer TEXT,checkout_id TEXT,generation INTEGER DEFAULT 0,
          PRIMARY KEY(owner,mode),UNIQUE(customer,mode));
        CREATE TABLE IF NOT EXISTS billing_subscriptions(id TEXT,owner TEXT,mode TEXT,status TEXT,period_end REAL,cancel_at_period_end INTEGER,
          PRIMARY KEY(id,mode));
        CREATE TABLE IF NOT EXISTS billing_events(id TEXT,mode TEXT,kind TEXT,received REAL,PRIMARY KEY(id,mode));
        ''')
    app.register_blueprint(billing_bp)
