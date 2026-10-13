"""Account-bound billing and provider-neutral access. Payments never assign roles."""
import hmac
import json
import os
import secrets
import time
from functools import wraps
from urllib.parse import urlsplit

import stripe
from flask import Blueprint, current_app, jsonify, request, Response, session
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
    """Website status in each provider's configured environment (including sandbox)."""
    from .paypal_billing import paypal_mode
    with store().connect() as db:
        rows = db.execute('SELECT status,period_end FROM billing_subscriptions WHERE owner=? AND mode=?',
                          (owner, mode())).fetchall()
        rows += db.execute('SELECT status,period_end FROM paypal_subscriptions WHERE owner=? AND mode=?',
                           (owner, paypal_mode())).fetchall()
    return any(r['status'] == 'active' and r['period_end'] > time.time() for r in rows)


def subscription_access(owner):
    """Only verified, unexpired live records count as paid access; never infer roles."""
    from .paypal_billing import paypal_mode
    paid = False
    with store().connect() as db:
        for table, environment in (('billing_subscriptions', mode()),
                                   ('paypal_subscriptions', paypal_mode())):
            if environment == 'live':
                paid = paid or bool(db.execute(
                    f"SELECT 1 FROM {table} WHERE owner=? AND mode='live' "
                    "AND status='active' AND period_end>? LIMIT 1", (owner, time.time())).fetchone())
    return {'subscription_status': 'active' if paid else
            'sandbox_only' if mode() == paypal_mode() == 'sandbox' else 'inactive',
            'paid_access': paid, 'paid_ai_included': False}


def subscription_conflict(db, owner, provider):
    """Check the other provider under the checkout write lock, including pending intent."""
    from .paypal_billing import paypal_mode, paypal_checkout_pending
    if provider == 'paypal':
        return bool(db.execute(
            "SELECT 1 FROM billing_subscriptions WHERE owner=? AND mode=? "
            "AND status NOT IN ('canceled','incomplete_expired') LIMIT 1", (owner, mode())).fetchone()
            or db.execute('SELECT 1 FROM billing_customers WHERE owner=? AND mode=? '
                          'AND checkout_id IS NOT NULL', (owner, mode())).fetchone()
            or db.execute('SELECT 1 FROM stripe_checkout_intents WHERE owner=? AND mode=? AND checkout_id IS NULL',
                          (owner, mode())).fetchone())
    if provider == 'stripe':
        return bool(db.execute(
            "SELECT 1 FROM paypal_subscriptions WHERE owner=? AND mode=? "
            "AND status NOT IN ('canceled','expired') "
            "AND NOT (cancel_at_period_end=1 AND period_end<=?) LIMIT 1",
            (owner, paypal_mode(), time.time())).fetchone()
            or paypal_checkout_pending(db, owner, paypal_mode()))
    raise ValueError('Unknown subscription provider.')


def require_subscription():
    # This opt-in gate is separate from identity/role authorization and never enabled in sandbox.
    if current_app.config['BILLING_REQUIRE_SUBSCRIPTION'] and not subscription_access(current_user.get_id())['paid_access']:
        return jsonify(error='An active subscription is required to create or execute runs.', code='subscription_required'), 402


@billing_bp.route('/billing')
def billing_page():
    from .paypal_billing import paypal_mode
    html = (STATIC / 'studio_billing.html').read_text()
    if mode() == 'live' or paypal_mode() == 'live':
        html = html.replace('Sandbox preparation — no real payments enabled.', 'Live billing — real monthly payments.')
        html = html.replace('Test monthly checkout', 'Subscribe for $19/month')
        html = html.replace('Sandbox subscriptions are test records, not paid access. The current workspace remains available according to its assigned roles.', 'Subscription payments are real. Workspace permissions are assigned separately; contact the workspace administrator for operator access before subscribing.')
    return Response(html, mimetype='text/html')


@billing_bp.route('/api/billing/status')
@login_required
def status():
    from .paypal_billing import paypal_configured, paypal_mode
    session.setdefault('studio_csrf', secrets.token_urlsafe(32))
    with store().connect() as db:
        subscriptions = db.execute('SELECT status,period_end,cancel_at_period_end FROM billing_subscriptions WHERE owner=? AND mode=?',
                                   (current_user.get_id(), mode())).fetchall()
        customer = db.execute('SELECT customer FROM billing_customers WHERE owner=? AND mode=?',
                              (current_user.get_id(), mode())).fetchone()
        paypal_subscriptions = db.execute(
            'SELECT status,period_end,cancel_at_period_end,provider_status FROM paypal_subscriptions WHERE owner=? AND mode=?',
            (current_user.get_id(), paypal_mode())).fetchall()
        stripe_conflict = subscription_conflict(db, current_user.get_id(), 'stripe')
        paypal_conflict = subscription_conflict(db, current_user.get_id(), 'paypal')
    return jsonify(mode=mode(), configured=configured(), amount_usd=19, interval='month',
                   subscribed=has_subscription(current_user.get_id()), can_manage=bool(customer and subscriptions),
                   subscriptions=[dict(r) for r in subscriptions], csrf_token=session['studio_csrf'],
                   access=subscription_access(current_user.get_id()),
                   paypal={'mode': paypal_mode(), 'configured': paypal_configured(),
                           'subscriptions': [dict(r) for r in paypal_subscriptions]},
                   checkout_conflicts={'stripe': stripe_conflict, 'paypal': paypal_conflict},
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
    # Commit a provider reservation before any create call. An uncertain Stripe
    # request must also block PayPal until resolved, even after a process restart.
    with store().connect() as db:
        db.execute('BEGIN IMMEDIATE')
        if subscription_conflict(db, owner, 'stripe'):
            return jsonify(error='Manage or reconcile your existing PayPal subscription before starting another.'), 409
        if db.execute("SELECT 1 FROM billing_subscriptions WHERE owner=? AND mode=? AND status NOT IN ('canceled','incomplete_expired')",
                      (owner, mode())).fetchone():
            return jsonify(error='Manage your existing subscription instead of starting another.'), 409
        row = db.execute('SELECT * FROM billing_customers WHERE owner=? AND mode=?', (owner, mode())).fetchone()
        if row and row['checkout_id']:
            existing = client().v1.checkout.sessions.retrieve(row['checkout_id'])
            if existing.status == 'open':
                return jsonify(url=existing.url)
            if existing.status != 'expired':
                return jsonify(error='Checkout is complete or pending. Refresh payment status before continuing.'), 409
            db.execute('UPDATE billing_customers SET checkout_id=NULL WHERE owner=? AND mode=?', (owner, mode()))
            db.execute('DELETE FROM stripe_checkout_intents WHERE owner=? AND mode=? AND checkout_id=?',
                       (owner, mode(), row['checkout_id']))
        intent = db.execute('SELECT * FROM stripe_checkout_intents WHERE owner=? AND mode=?', (owner, mode())).fetchone()
        if not intent:
            db.execute('INSERT INTO stripe_checkout_intents(owner,mode,request_id,created) VALUES(?,?,?,?)',
                       (owner, mode(), secrets.token_urlsafe(24), time.time()))
    # Persist the customer and immutable checkout payload before checkout creation.
    with store().connect() as db:
        db.execute('BEGIN IMMEDIATE')
        intent = db.execute('SELECT * FROM stripe_checkout_intents WHERE owner=? AND mode=?', (owner, mode())).fetchone()
        if time.time() - intent['created'] >= 23 * 3600 and not intent['checkout_id']:
            return jsonify(error='This checkout needs merchant reconciliation before another can be created.'), 409
        row = db.execute('SELECT * FROM billing_customers WHERE owner=? AND mode=?', (owner, mode())).fetchone()
        if not row:
            customer = client().v1.customers.create({'metadata': {'dredge_owner': owner, 'app': 'dredge_studio'}},
                options={'idempotency_key': 'dredge-customer-'+intent['request_id']})
            db.execute('INSERT INTO billing_customers(owner,mode,customer) VALUES(?,?,?)', (owner, mode(), customer.id))
            row = db.execute('SELECT * FROM billing_customers WHERE owner=? AND mode=?', (owner, mode())).fetchone()
        if not intent['payload']:
            payload = {
                'mode': 'subscription', 'customer': row['customer'], 'client_reference_id': owner,
                'line_items': [{'price': price.id, 'quantity': 1}],
                'subscription_data': {'billing_mode': {'type': 'flexible'}, 'metadata': {'dredge_owner': owner, 'app': 'dredge_studio'}},
                'integration_identifier': current_app.config['STRIPE_INTEGRATION_IDENTIFIER'],
                'success_url': origin+'/billing?checkout=returned', 'cancel_url': origin+'/billing?checkout=canceled'}
            db.execute('UPDATE stripe_checkout_intents SET payload=? WHERE owner=? AND mode=?',
                       (json.dumps(payload, sort_keys=True), owner, mode()))
    with store().connect() as db:
        db.execute('BEGIN IMMEDIATE')
        intent = db.execute('SELECT * FROM stripe_checkout_intents WHERE owner=? AND mode=?', (owner, mode())).fetchone()
        if intent['checkout_id']:
            existing = client().v1.checkout.sessions.retrieve(intent['checkout_id'])
            if existing.status == 'open':
                return jsonify(url=existing.url)
            return jsonify(error='Refresh payment status before continuing.'), 409
        if time.time() - intent['created'] >= 23 * 3600:
            return jsonify(error='This checkout needs merchant reconciliation before another can be created.'), 409
        payload = json.loads(intent['payload'])
        if (payload['line_items'][0]['price'] != price.id
                or payload['success_url'] != origin+'/billing?checkout=returned'):
            return jsonify(error='Pending checkout configuration changed. Merchant reconciliation is required.'), 409
        result = client().v1.checkout.sessions.create(payload,
            options={'idempotency_key': 'dredge-checkout-'+intent['request_id']})
        db.execute('UPDATE billing_customers SET checkout_id=?,generation=generation+1 WHERE owner=? AND mode=?',
                   (result.id, owner, mode()))
        db.execute('UPDATE stripe_checkout_intents SET checkout_id=? WHERE owner=? AND mode=?',
                   (result.id, owner, mode()))
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


@billing_bp.route('/api/billing/reconcile', methods=['POST'])
@authenticated_mutation
def reconcile():
    """Recover missed Stripe notifications for this account; never trust browser IDs."""
    owner = current_user.get_id()
    with store().connect() as db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute('SELECT customer,checkout_id FROM billing_customers WHERE owner=? AND mode=?',
                         (owner, mode())).fetchone()
        subscription_ids = {r['id'] for r in db.execute(
            'SELECT id FROM billing_subscriptions WHERE owner=? AND mode=?', (owner, mode())).fetchall()}
        if row and row['checkout_id']:
            checkout_session = client().v1.checkout.sessions.retrieve(row['checkout_id'])
            if hasattr(checkout_session, 'to_dict'):
                checkout_session = checkout_session.to_dict()
            if (checkout_session.get('customer') != row['customer']
                    or checkout_session.get('client_reference_id') != owner
                    or bool(checkout_session.get('livemode')) != (mode() == 'live')):
                return jsonify(error='Checkout account binding could not be verified.'), 409
            if checkout_session.get('status') == 'expired':
                db.execute('UPDATE billing_customers SET checkout_id=NULL WHERE owner=? AND mode=?', (owner, mode()))
                db.execute('DELETE FROM stripe_checkout_intents WHERE owner=? AND mode=? AND checkout_id=?',
                           (owner, mode(), row['checkout_id']))
            if checkout_session.get('subscription'):
                subscription_ids.add(checkout_session['subscription'])
        for sid in subscription_ids:
            if not reconcile_stripe_subscription(db, sid, expected_owner=owner):
                return jsonify(error='Subscription account binding could not be verified.'), 409
        if row and row['checkout_id'] and checkout_session.get('subscription'):
            terminal = db.execute("SELECT 1 FROM billing_subscriptions WHERE id=? AND owner=? AND mode=? AND status IN ('canceled','incomplete_expired')",
                                  (checkout_session['subscription'], owner, mode())).fetchone()
            if terminal:
                db.execute('UPDATE billing_customers SET checkout_id=NULL WHERE owner=? AND mode=?', (owner, mode()))
                db.execute('DELETE FROM stripe_checkout_intents WHERE owner=? AND mode=? AND checkout_id=?',
                           (owner, mode(), row['checkout_id']))
    return status()


def reconcile_stripe_subscription(db, sid, expected_owner=None):
    """Read latest authoritative state while holding the caller's write lock."""
    sub = client().v1.subscriptions.retrieve(sid)
    if hasattr(sub, 'to_dict'):
        sub = sub.to_dict()
    if sub.get('id') != sid:
        return False
    customer = sub['customer']
    row = db.execute('SELECT owner FROM billing_customers WHERE customer=? AND mode=?', (customer, mode())).fetchone()
    owner = sub.get('metadata', {}).get('dredge_owner')
    items = sub['items']['data']
    existing = db.execute('SELECT owner FROM billing_subscriptions WHERE id=? AND mode=?',
                          (sid, mode())).fetchone()
    if (not row or row['owner'] != owner or (expected_owner and owner != expected_owner)
            or (existing and existing['owner'] != owner) or bool(sub['livemode']) != (mode() == 'live')
            or len(items) != 1 or any(i['price']['id'] != current_app.config['STRIPE_PRICE_ID']
            or i.get('quantity', 1) != 1 for i in items)):
        # A previously known subscription that no longer passes binding/plan checks
        # cannot retain paid access until its old cached expiry.
        if existing:
            db.execute("UPDATE billing_subscriptions SET status='unverified',period_end=0 WHERE id=? AND mode=?",
                       (sid, mode()))
        return False
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
    return True


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
            if not reconcile_stripe_subscription(db, sid):
                return jsonify(received=True, ignored=True)
            db.execute('INSERT INTO billing_events(id,mode,kind,received) VALUES(?,?,?,?)', (event['id'], mode(), event['type'], time.time()))
            store().audit(db, 'stripe', 'subscription_updated', detail={'mode': mode()})
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
        CREATE TABLE IF NOT EXISTS stripe_checkout_intents(owner TEXT,mode TEXT,request_id TEXT NOT NULL,created REAL NOT NULL,
          payload TEXT,checkout_id TEXT,PRIMARY KEY(owner,mode));
        CREATE TABLE IF NOT EXISTS billing_events(id TEXT,mode TEXT,kind TEXT,received REAL,PRIMARY KEY(id,mode));
        ''')
    app.register_blueprint(billing_bp)
    from .paypal_billing import register_paypal_billing
    from .subscription_info import register_subscription_info
    register_paypal_billing(app)
    register_subscription_info(app)
    if app.config.get('PAYPAL_BILLING_MODE') == 'live' and os.environ.get('BILLING_REQUIRE_SUBSCRIPTION') == 'true':
        app.config['BILLING_REQUIRE_SUBSCRIPTION'] = True
