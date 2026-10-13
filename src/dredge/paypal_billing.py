"""Account-bound PayPal monthly subscriptions; disabled until explicitly configured.

Only fixed PayPal API origins are contacted. Browser return parameters never create
an entitlement. Provider state and completed payments are read under the SQLite
write lock. This module never assigns workspace roles or enables paid inference.
"""
import hmac
import json
import os
import re
import secrets
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from functools import wraps
from urllib.parse import urlsplit

import requests
from flask import Blueprint, current_app, jsonify, request, session
from flask_login import current_user, login_required

from .studio import store

paypal_bp = Blueprint('paypal_billing', __name__)
API_ORIGINS = {'sandbox': 'https://api-m.sandbox.paypal.com', 'live': 'https://api-m.paypal.com'}
EVENTS = {'BILLING.SUBSCRIPTION.CREATED', 'BILLING.SUBSCRIPTION.ACTIVATED',
          'BILLING.SUBSCRIPTION.UPDATED', 'BILLING.SUBSCRIPTION.CANCELLED',
          'BILLING.SUBSCRIPTION.EXPIRED', 'BILLING.SUBSCRIPTION.SUSPENDED',
          'BILLING.SUBSCRIPTION.PAYMENT.FAILED', 'PAYMENT.SALE.COMPLETED',
          'PAYMENT.SALE.DENIED', 'PAYMENT.SALE.REFUNDED', 'PAYMENT.SALE.REVERSED'}
TERMINAL = {'CANCELLED', 'EXPIRED'}
# PayPal documents a 72-hour subscription-create idempotency retention window.
# Stop a little early; an unresolved request must never create a second charge.
CREATE_RETRY_SECONDS = 71 * 3600


class PayPalError(Exception):
    """Sanitized provider failure. Never contains credentials or a response body."""


def paypal_mode():
    return current_app.config.get('PAYPAL_BILLING_MODE', 'sandbox')


def paypal_client():
    return current_app.extensions.get('paypal_client')


def paypal_configured():
    client = paypal_client()
    return bool(current_app.config.get('PAYPAL_BILLING_ENABLED') is True and client
                and client.mode == paypal_mode() and current_app.config.get('PAYPAL_PLAN_ID')
                and current_app.config.get('PAYPAL_WEBHOOK_ID'))


def _identifier(value, prefix=None):
    return (isinstance(value, str) and bool(re.fullmatch(r'[A-Za-z0-9-]{3,100}', value))
            and (not prefix or value.startswith(prefix)))


def _certificate_url(value, billing_mode):
    if not isinstance(value, str):
        return False
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    hosts = {'api.sandbox.paypal.com', 'api-m.sandbox.paypal.com'} if billing_mode == 'sandbox' else {'api.paypal.com', 'api-m.paypal.com'}
    return bool(parsed.scheme == 'https' and parsed.netloc in hosts and not parsed.query
                and not parsed.fragment and re.fullmatch(r'/v1/notifications/certs/CERT-[A-Za-z0-9-]+', parsed.path))


class PayPalClient:
    """Small fixed-origin REST adapter. No externally supplied URL is fetched."""
    def __init__(self, client_id, client_secret, billing_mode):
        self.mode = billing_mode
        self.origin = API_ORIGINS[billing_mode]
        self._credentials = (client_id, client_secret)
        self._http = requests.Session()
        self._token = None
        self._token_until = 0

    def _send(self, method, path, **kwargs):
        try:
            response = self._http.request(method, self.origin + path, timeout=(5, 15),
                                          allow_redirects=False, **kwargs)
            if not 200 <= response.status_code < 300:
                raise PayPalError('PayPal request failed.')
            if response.status_code == 204:
                return {}
            result = response.json()
            if not isinstance(result, dict):
                raise PayPalError('Invalid PayPal response.')
            return result
        except (requests.RequestException, ValueError) as exc:
            raise PayPalError('PayPal request failed.') from exc

    def _access_token(self):
        if self._token and self._token_until > time.time():
            return self._token
        result = self._send('POST', '/v1/oauth2/token', auth=self._credentials,
                            data={'grant_type': 'client_credentials'}, headers={'Accept': 'application/json'})
        token = result.get('access_token')
        try:
            lifetime = min(float(result['expires_in']), 3600)
        except (KeyError, ValueError, TypeError):
            raise PayPalError('Invalid PayPal authentication response.') from None
        if not isinstance(token, str) or not token or lifetime <= 60:
            raise PayPalError('Invalid PayPal authentication response.')
        self._token, self._token_until = token, time.time() + lifetime - 60
        return token

    def _call(self, method, path, **kwargs):
        headers = {'Authorization': 'Bearer ' + self._access_token(), 'Content-Type': 'application/json'}
        headers.update(kwargs.pop('headers', {}))
        return self._send(method, path, headers=headers, **kwargs)

    def get_plan(self, plan_id):
        if not _identifier(plan_id, 'P-'):
            raise PayPalError('Invalid configured PayPal plan.')
        return self._call('GET', '/v1/billing/plans/' + plan_id)

    def create_subscription(self, payload, request_id):
        return self._call('POST', '/v1/billing/subscriptions', json=payload,
                          headers={'PayPal-Request-Id': request_id, 'Prefer': 'return=representation'})

    def get_subscription(self, sid):
        if not _identifier(sid, 'I-'):
            raise PayPalError('Invalid PayPal subscription identifier.')
        return self._call('GET', '/v1/billing/subscriptions/' + sid, params={'fields': 'last_failed_payment,plan'})

    def list_transactions(self, sid, start_time, end_time):
        if not _identifier(sid, 'I-'):
            raise PayPalError('Invalid PayPal subscription identifier.')
        return self._call('GET', '/v1/billing/subscriptions/' + sid + '/transactions',
                          params={'start_time': start_time, 'end_time': end_time})

    def cancel_subscription(self, sid):
        if not _identifier(sid, 'I-'):
            raise PayPalError('Invalid PayPal subscription identifier.')
        return self._call('POST', '/v1/billing/subscriptions/' + sid + '/cancel',
                          json={'reason': 'Account owner requested cancellation in DREDGE billing.'})

    def verify_webhook(self, raw_body, headers, webhook_id):
        if not _certificate_url(headers.get('PAYPAL-CERT-URL'), self.mode):
            return False
        metadata = {'auth_algo': headers['PAYPAL-AUTH-ALGO'], 'cert_url': headers['PAYPAL-CERT-URL'],
                    'transmission_id': headers['PAYPAL-TRANSMISSION-ID'],
                    'transmission_sig': headers['PAYPAL-TRANSMISSION-SIG'],
                    'transmission_time': headers['PAYPAL-TRANSMISSION-TIME'], 'webhook_id': webhook_id}
        # Preserve the exact original JSON event bytes for PayPal postback verification.
        body = json.dumps(metadata, separators=(',', ':')).encode()[:-1] + b',"webhook_event":' + raw_body + b'}'
        result = self._call('POST', '/v1/notifications/verify-webhook-signature', data=body)
        return result.get('verification_status') == 'SUCCESS'


def _amount(value, expected, currency='USD'):
    try:
        return (isinstance(value, dict) and value.get('currency_code') == currency
                and Decimal(str(value.get('value'))) == Decimal(str(expected)))
    except InvalidOperation:
        return False


def _zero(value):
    try:
        return Decimal(str(value)) == 0
    except InvalidOperation:
        return False


def _plan_valid(plan, *, active=False, embedded=False):
    if not isinstance(plan, dict):
        return False
    cycles = plan.get('billing_cycles', [])
    if (not isinstance(cycles, list) or len(cycles) != 1 or not isinstance(cycles[0], dict)
            or not isinstance(plan.get('payment_preferences', {}), dict)
            or not isinstance(plan.get('taxes', {}), dict)):
        return False
    if (not embedded and plan.get('id') != current_app.config['PAYPAL_PLAN_ID']
            or active and plan.get('status') != 'ACTIVE' or len(cycles) != 1
            or plan.get('quantity_supported', False) is not False):
        return False
    cycle = cycles[0]
    pricing = cycle.get('pricing_scheme', {})
    if not isinstance(pricing, dict):
        return False
    setup = plan.get('payment_preferences', {}).get('setup_fee')
    return bool(cycle.get('tenure_type') == 'REGULAR' and cycle.get('sequence') == 1
                and cycle.get('total_cycles') == 0
                and cycle.get('frequency') == {'interval_unit': 'MONTH', 'interval_count': 1}
                and _amount(pricing.get('fixed_price'), 19) and not pricing.get('tiers')
                and not pricing.get('pricing_model') and (setup is None or _amount(setup, 0))
                and _zero(plan.get('taxes', {}).get('percentage', '0')))


def _timestamp(value):
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        return parsed.timestamp() if parsed.tzinfo else 0
    except (AttributeError, TypeError, ValueError, OverflowError):
        return 0


def _stamp(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat().replace('+00:00', 'Z')


def _paid_term_limit(value):
    # An anchored monthly plan can run Feb 28 -> Mar 31. Adding one calendar
    # month would undergrant those days. A month never exceeds 31 days; the
    # authoritative next billing boundary below can shorten this upper bound.
    return value + 31 * 86400


def authenticated_mutation(fn):
    @wraps(fn)
    @login_required
    def wrapped(*args, **kwargs):
        token = session.get('studio_csrf', '')
        if not token or not hmac.compare_digest(token, request.headers.get('X-CSRF-Token', '')):
            return jsonify(error='Refresh billing before continuing.'), 403
        if not paypal_configured():
            return jsonify(error='PayPal billing is disabled or awaiting secure configuration.'), 503
        try:
            return fn(*args, **kwargs)
        except PayPalError:
            current_app.logger.warning('PayPal billing request failed; no payment details logged.')
            return jsonify(error='PayPal could not complete this request. Please retry or refresh billing.'), 502
    return wrapped


def paypal_checkout_pending(db, owner, billing_mode):
    return bool(db.execute('SELECT 1 FROM paypal_checkout_intents WHERE owner=? AND mode=? AND subscription_id IS NULL',
                           (owner, billing_mode)).fetchone())


def _summary(owner):
    session.setdefault('studio_csrf', secrets.token_urlsafe(32))
    with store().connect() as db:
        rows = db.execute('SELECT status,period_end,cancel_at_period_end,provider_status FROM paypal_subscriptions WHERE owner=? AND mode=?',
                          (owner, paypal_mode())).fetchall()
    return dict(mode=paypal_mode(), configured=paypal_configured(), amount_usd=19, interval='month',
                subscribed=any(r['status'] == 'active' and r['period_end'] > time.time() for r in rows),
                can_cancel=any(r['provider_status'] not in TERMINAL for r in rows),
                subscriptions=[dict(r) for r in rows], csrf_token=session['studio_csrf'])


@paypal_bp.route('/api/billing/paypal/status')
@login_required
def status():
    return jsonify(_summary(current_user.get_id()))


def _approval_url(sub):
    host = 'www.sandbox.paypal.com' if paypal_mode() == 'sandbox' else 'www.paypal.com'
    links = sub.get('links', [])
    if not isinstance(links, list):
        raise PayPalError('Invalid PayPal approval response.')
    for link in links:
        if not isinstance(link, dict) or link.get('rel') != 'approve':
            continue
        value = link.get('href', '')
        try:
            parsed = urlsplit(value)
        except (ValueError, TypeError):
            continue
        if parsed.scheme == 'https' and parsed.netloc == host and not parsed.fragment:
            return value
    raise PayPalError('PayPal approval URL is unavailable.')


def _bind_created(db, intent, sub):
    sid = sub.get('id')
    if (not _identifier(sid, 'I-') or sub.get('custom_id') != intent['binding']
            or sub.get('plan_id') != current_app.config['PAYPAL_PLAN_ID']
            or sub.get('plan_overridden', False) is not False):
        raise PayPalError('PayPal subscription binding did not match.')
    existing = db.execute('SELECT owner,binding FROM paypal_subscriptions WHERE id=? AND mode=?', (sid, paypal_mode())).fetchone()
    if existing and (existing['owner'] != intent['owner'] or existing['binding'] != intent['binding']):
        raise PayPalError('PayPal subscription already belongs to another account.')
    db.execute('INSERT OR IGNORE INTO paypal_subscriptions(id,owner,mode,status,period_end,cancel_at_period_end,binding,plan_id,provider_status) VALUES(?,?,?,?,?,?,?,?,?)',
               (sid, intent['owner'], paypal_mode(), 'payment_pending', 0, 0, intent['binding'], sub['plan_id'], sub.get('status', 'UNKNOWN')))
    db.execute('UPDATE paypal_checkout_intents SET subscription_id=? WHERE request_id=? AND subscription_id IS NULL', (sid, intent['request_id']))
    return sid


@paypal_bp.route('/api/billing/paypal/checkout', methods=['POST'])
@authenticated_mutation
def checkout():
    from .billing import subscription_conflict
    owner = current_user.get_id()
    if not _plan_valid(paypal_client().get_plan(current_app.config['PAYPAL_PLAN_ID']), active=True):
        return jsonify(error='PayPal price does not match the displayed $19 USD monthly plan.'), 503
    with store().connect() as db:
        db.execute('BEGIN IMMEDIATE')
        if subscription_conflict(db, owner, 'paypal'):
            return jsonify(error='Manage or reconcile your existing Stripe subscription first.'), 409
        existing = db.execute("SELECT * FROM paypal_subscriptions WHERE owner=? AND mode=? AND (provider_status NOT IN ('CANCELLED','EXPIRED') OR (status='active' AND period_end>?)) ORDER BY rowid DESC LIMIT 1", (owner, paypal_mode(), time.time())).fetchone()
        if existing:
            sub = paypal_client().get_subscription(existing['id'])
            if (sub.get('id') != existing['id'] or sub.get('custom_id') != existing['binding']
                    or sub.get('plan_id') != existing['plan_id']):
                raise PayPalError('PayPal subscription binding did not match.')
            if sub.get('status') == 'APPROVAL_PENDING':
                return jsonify(url=_approval_url(sub))
            return jsonify(error='Manage or refresh your existing PayPal subscription instead.'), 409
        intent = db.execute('SELECT * FROM paypal_checkout_intents WHERE owner=? AND mode=? AND subscription_id IS NULL ORDER BY created LIMIT 1', (owner, paypal_mode())).fetchone()
        if not intent:
            request_id, binding = str(uuid.uuid4()), secrets.token_urlsafe(32)
            origin = current_app.config['BILLING_ORIGIN']
            payload = {'plan_id': current_app.config['PAYPAL_PLAN_ID'], 'quantity': '1', 'custom_id': binding,
                       'application_context': {'brand_name': 'DREDGE Studio', 'shipping_preference': 'NO_SHIPPING',
                           'user_action': 'SUBSCRIBE_NOW', 'return_url': origin + '/billing?paypal=returned',
                           'cancel_url': origin + '/billing?paypal=canceled'}}
            db.execute('INSERT INTO paypal_checkout_intents(request_id,owner,mode,binding,payload,created) VALUES(?,?,?,?,?,?)',
                       (request_id, owner, paypal_mode(), binding, json.dumps(payload, sort_keys=True), time.time()))
        pending_request_id = intent['request_id'] if intent else request_id
    # Commit the request identity BEFORE the external call, including on timeouts or process failure.
    with store().connect() as db:
        db.execute('BEGIN IMMEDIATE')
        intent = db.execute('SELECT * FROM paypal_checkout_intents WHERE request_id=?', (pending_request_id,)).fetchone()
        if subscription_conflict(db, owner, 'paypal'):
            return jsonify(error='Manage your existing Stripe subscription first.'), 409
        if intent['subscription_id']:
            sub = paypal_client().get_subscription(intent['subscription_id'])
            if sub.get('status') != 'APPROVAL_PENDING':
                return jsonify(error='Refresh your existing PayPal subscription.'), 409
            if sub.get('custom_id') != intent['binding'] or sub.get('id') != intent['subscription_id']:
                raise PayPalError('PayPal subscription binding did not match.')
        else:
            if time.time() - intent['created'] >= CREATE_RETRY_SECONDS:
                return jsonify(error='This checkout needs merchant reconciliation before another can be created.'), 409
            payload = json.loads(intent['payload'])
            if payload['plan_id'] != current_app.config['PAYPAL_PLAN_ID']:
                return jsonify(error='The pending checkout plan changed. Merchant reconciliation is required.'), 409
            sub = paypal_client().create_subscription(payload, intent['request_id'])
            _bind_created(db, intent, sub)
            if sub.get('status') != 'APPROVAL_PENDING':
                return jsonify(error='PayPal approval is complete. Refresh billing to verify payment.'), 409
        return jsonify(url=_approval_url(sub))


def _payment_period(db, sub, existing):
    info = sub.get('billing_info') or {}
    if not isinstance(info, dict):
        return 0
    last = info.get('last_payment') or {}
    failed = info.get('last_failed_payment') or {}
    if not isinstance(last, dict) or not isinstance(failed, dict):
        return 0
    paid_at = _timestamp(last.get('time'))
    now = time.time()
    if (not _amount(last.get('amount'), 19) or not 0 < paid_at <= now
            or _paid_term_limit(paid_at) <= now
            or info.get('failed_payments_count', 0) != 0
            or _timestamp(failed.get('time')) > paid_at
            or info.get('outstanding_balance') is not None and not _amount(info['outstanding_balance'], 0)):
        return 0
    result = paypal_client().list_transactions(sub['id'], _stamp(paid_at - 86400), _stamp(now))
    # Never follow pagination URLs supplied by a provider response. An incomplete
    # payment list must fail closed, not accidentally overlook a reversal.
    if result.get('total_pages', 1) not in (0, 1):
        return 0
    transactions = result.get('transactions', [])
    if not isinstance(transactions, list):
        return 0
    completed = []
    for transaction in transactions:
        if not isinstance(transaction, dict) or not isinstance(transaction.get('amount_with_breakdown', {}), dict):
            continue
        txid = transaction.get('id')
        if not _identifier(txid):
            continue
        when = _timestamp(transaction.get('time'))
        state = transaction.get('status')
        db.execute('INSERT INTO paypal_payments(id,mode,subscription_id,status) VALUES(?,?,?,?) ON CONFLICT(id,mode) DO UPDATE SET status=CASE WHEN paypal_payments.status IN (\'REFUNDED\',\'PARTIALLY_REFUNDED\',\'REVERSED\') THEN paypal_payments.status ELSE excluded.status END',
                   (txid, paypal_mode(), sub['id'], state or 'UNKNOWN'))
        saved = db.execute('SELECT subscription_id,status FROM paypal_payments WHERE id=? AND mode=?', (txid, paypal_mode())).fetchone()
        if (state == 'COMPLETED' and saved['status'] == 'COMPLETED' and saved['subscription_id'] == sub['id']
                and _amount(transaction.get('amount_with_breakdown', {}).get('gross_amount'), 19)
                and abs(when - paid_at) <= 1 and 0 < when <= now):
            completed.append(txid)
    if not completed:
        return 0
    end = _timestamp(info.get('next_billing_time'))
    if sub.get('status') in TERMINAL:
        # PayPal may omit the next billing time after cancellation. Only keep an
        # already verified paid boundary, never extend it from the cancellation.
        end = min(end or existing['period_end'], existing['period_end']) if existing else 0
    return min(end, _paid_term_limit(paid_at)) if end > paid_at else 0


def _reconcile(db, sid):
    # Both owner refresh and verified webhook callers hold BEGIN IMMEDIATE.
    sub = paypal_client().get_subscription(sid)
    if not isinstance(sub, dict) or not isinstance(sub.get('custom_id'), str):
        raise PayPalError('Invalid PayPal subscription response.')
    existing = db.execute('SELECT * FROM paypal_subscriptions WHERE id=? AND mode=?', (sid, paypal_mode())).fetchone()
    if not existing:
        intent = db.execute('SELECT * FROM paypal_checkout_intents WHERE binding=? AND mode=? AND subscription_id IS NULL', (sub.get('custom_id'), paypal_mode())).fetchone()
        if not intent or sub.get('id') != sid:
            return False
        _bind_created(db, intent, sub)
        existing = db.execute('SELECT * FROM paypal_subscriptions WHERE id=? AND mode=?', (sid, paypal_mode())).fetchone()
    state, end, cancel_at_end = 'invalid', 0, 0
    valid = (sub.get('id') == sid and sub.get('custom_id') == existing['binding']
             and sub.get('plan_id') == existing['plan_id'] == current_app.config['PAYPAL_PLAN_ID']
             and sub.get('plan_overridden', False) is False and sub.get('quantity', '1') == '1'
             and (not sub.get('shipping_amount') or _amount(sub['shipping_amount'], 0)))
    if valid:
        valid = _plan_valid(paypal_client().get_plan(current_app.config['PAYPAL_PLAN_ID']))
        if sub.get('plan') is not None:
            valid = valid and _plan_valid(sub['plan'], embedded=True)
    provider_state = sub.get('status', 'UNKNOWN')
    if not isinstance(provider_state, str) or provider_state not in {'APPROVAL_PENDING', 'APPROVED', 'ACTIVE', 'SUSPENDED', 'CANCELLED', 'EXPIRED'}:
        provider_state = 'UNKNOWN'
    # Cancellation and expiry are terminal: delayed provider snapshots never
    # resurrect a recurring agreement even if an old activation event arrives.
    if existing['provider_status'] in TERMINAL:
        provider_state = existing['provider_status']
        sub['status'] = provider_state
    if valid:
        state = provider_state.lower()
        if provider_state in {'ACTIVE', 'CANCELLED'}:
            end = _payment_period(db, sub, existing)
            state = 'active' if end > time.time() else ('canceled' if provider_state == 'CANCELLED' else 'payment_pending')
            cancel_at_end = int(provider_state == 'CANCELLED')
    db.execute('UPDATE paypal_subscriptions SET status=?,period_end=?,cancel_at_period_end=?,provider_status=? WHERE id=? AND mode=?',
               (state, end, cancel_at_end, provider_state, sid, paypal_mode()))
    store().audit(db, 'paypal', 'subscription_updated', detail={'mode': paypal_mode(), 'owner': existing['owner'], 'status': state})
    return True


@paypal_bp.route('/api/billing/paypal/reconcile', methods=['POST'])
@authenticated_mutation
def reconcile():
    owner = current_user.get_id()
    with store().connect() as db:
        db.execute('BEGIN IMMEDIATE')
        rows = db.execute('SELECT id FROM paypal_subscriptions WHERE owner=? AND mode=?', (owner, paypal_mode())).fetchall()
        if not rows:
            return jsonify(error='No PayPal subscription exists for your sign-in.'), 404
        for row in rows:
            _reconcile(db, row['id'])
    return jsonify(_summary(owner))


@paypal_bp.route('/api/billing/paypal/cancel', methods=['POST'])
@authenticated_mutation
def cancel():
    owner = current_user.get_id()
    with store().connect() as db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute("SELECT id FROM paypal_subscriptions WHERE owner=? AND mode=? AND provider_status NOT IN ('CANCELLED','EXPIRED') ORDER BY rowid DESC LIMIT 1", (owner, paypal_mode())).fetchone()
        if not row:
            return jsonify(error='No cancellable PayPal subscription exists for your sign-in.'), 404
        _reconcile(db, row['id'])
        # Re-read after reconciliation to avoid issuing cancel to an already ended agreement.
        state = db.execute('SELECT status,provider_status FROM paypal_subscriptions WHERE id=? AND mode=?', (row['id'], paypal_mode())).fetchone()
        if state['status'] == 'invalid':
            return jsonify(error='PayPal subscription verification failed. Review the agreement in PayPal before making changes.'), 409
        if state['provider_status'] not in TERMINAL:
            paypal_client().cancel_subscription(row['id'])
            # Persist the successful cancellation even if the subsequent GET fails.
            db.execute("UPDATE paypal_subscriptions SET provider_status='CANCELLED',cancel_at_period_end=1 WHERE id=? AND mode=?", (row['id'], paypal_mode()))
    return jsonify(_summary(owner))


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON keys')
        result[key] = value
    return result


def _event_subscription(db, event):
    resource = event['resource']
    if event['event_type'].startswith('BILLING.SUBSCRIPTION.'):
        return resource.get('id')
    sid = resource.get('billing_agreement_id')
    if sid:
        return sid
    txid = resource.get('sale_id') or resource.get('id')
    row = db.execute('SELECT subscription_id FROM paypal_payments WHERE id=? AND mode=?', (txid, paypal_mode())).fetchone()
    return row['subscription_id'] if row else None


@paypal_bp.route('/api/billing/paypal/webhook', methods=['POST'])
def webhook():
    if not paypal_configured():
        return jsonify(error='PayPal billing is not configured.'), 503
    if request.content_length and request.content_length > 1024 * 1024:
        return jsonify(error='Event too large.'), 413
    raw = request.get_data()
    if len(raw) > 1024 * 1024:
        return jsonify(error='Event too large.'), 413
    required = ('PAYPAL-TRANSMISSION-ID', 'PAYPAL-TRANSMISSION-TIME', 'PAYPAL-TRANSMISSION-SIG', 'PAYPAL-AUTH-ALGO', 'PAYPAL-CERT-URL')
    try:
        event = json.loads(raw, object_pairs_hook=_reject_duplicate_keys,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Invalid number')))
        if (not isinstance(event, dict) or not _identifier(event.get('id'))
                or not isinstance(event.get('event_type'), str) or not isinstance(event.get('resource'), dict)
                or not all(0 < len(request.headers.get(key, '')) <= 8192 for key in required)
                or not _certificate_url(request.headers.get('PAYPAL-CERT-URL'), paypal_mode())):
            raise ValueError('Invalid event')
    except (ValueError, TypeError):
        return jsonify(error='Invalid PayPal event.'), 400
    try:
        if not paypal_client().verify_webhook(raw, request.headers, current_app.config['PAYPAL_WEBHOOK_ID']):
            return jsonify(error='Invalid event signature.'), 400
        if event['event_type'] not in EVENTS:
            return jsonify(received=True)
        with store().connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT 1 FROM paypal_events WHERE id=? AND mode=?', (event['id'], paypal_mode())).fetchone():
                return jsonify(received=True, duplicate=True)
            sid = _event_subscription(db, event)
            if event['event_type'] in {'PAYMENT.SALE.REFUNDED', 'PAYMENT.SALE.REVERSED'}:
                resource = event['resource']
                txid = resource.get('sale_id') or resource.get('id')
                if _identifier(txid):
                    db.execute('INSERT INTO paypal_payments(id,mode,subscription_id,status) VALUES(?,?,?,?) ON CONFLICT(id,mode) DO UPDATE SET status=excluded.status',
                               (txid, paypal_mode(), sid if _identifier(sid, 'I-') else None, 'REFUNDED' if event['event_type'].endswith('REFUNDED') else 'REVERSED'))
            if not _identifier(sid, 'I-'):
                # A verified reversal can arrive before the completed-payment
                # event. Retain its tombstone even without a local subscription map.
                db.execute('INSERT INTO paypal_events(id,mode,kind,received) VALUES(?,?,?,?)', (event['id'], paypal_mode(), event['event_type'], time.time()))
                return jsonify(received=True, ignored=True)
            applied = _reconcile(db, sid)
            if not applied:
                return jsonify(received=True, ignored=True)
            db.execute('INSERT INTO paypal_events(id,mode,kind,received) VALUES(?,?,?,?)', (event['id'], paypal_mode(), event['event_type'], time.time()))
    except PayPalError:
        return jsonify(error='PayPal reconciliation failed; retry this event.'), 503
    return jsonify(received=True)


def register_paypal_billing(app):
    billing_mode = os.environ.get('PAYPAL_BILLING_MODE', 'sandbox')
    if billing_mode not in API_ORIGINS:
        raise RuntimeError('PAYPAL_BILLING_MODE must be sandbox or live.')
    enabled = os.environ.get('PAYPAL_BILLING_ENABLED') == 'true'
    app.config.update(PAYPAL_BILLING_MODE=billing_mode, PAYPAL_BILLING_ENABLED=enabled,
                      PAYPAL_PLAN_ID=os.environ.get('PAYPAL_PLAN_ID', ''),
                      PAYPAL_WEBHOOK_ID=os.environ.get('PAYPAL_WEBHOOK_ID', ''))
    client_id, client_secret = os.environ.get('PAYPAL_CLIENT_ID', ''), os.environ.get('PAYPAL_CLIENT_SECRET', '')
    if enabled and client_id and client_secret:
        app.extensions['paypal_client'] = PayPalClient(client_id, client_secret, billing_mode)
    with app.extensions['studio_store'].connect() as db:
        db.executescript('''
        CREATE TABLE IF NOT EXISTS paypal_checkout_intents(
          request_id TEXT PRIMARY KEY,owner TEXT NOT NULL,mode TEXT NOT NULL,binding TEXT NOT NULL UNIQUE,
          payload TEXT NOT NULL,created REAL NOT NULL,subscription_id TEXT);
        CREATE UNIQUE INDEX IF NOT EXISTS paypal_one_pending_checkout ON paypal_checkout_intents(owner,mode) WHERE subscription_id IS NULL;
        CREATE TABLE IF NOT EXISTS paypal_subscriptions(
          id TEXT NOT NULL,owner TEXT NOT NULL,mode TEXT NOT NULL,status TEXT NOT NULL,period_end REAL NOT NULL DEFAULT 0,
          cancel_at_period_end INTEGER NOT NULL DEFAULT 0,binding TEXT NOT NULL,plan_id TEXT NOT NULL,provider_status TEXT NOT NULL,
          PRIMARY KEY(id,mode),UNIQUE(binding,mode));
        CREATE TABLE IF NOT EXISTS paypal_events(id TEXT,mode TEXT,kind TEXT,received REAL,PRIMARY KEY(id,mode));
        CREATE TABLE IF NOT EXISTS paypal_payments(id TEXT,mode TEXT,subscription_id TEXT,status TEXT,PRIMARY KEY(id,mode));
        ''')
    app.register_blueprint(paypal_bp)
