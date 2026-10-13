"""Atomic owner-scoped limits for subsequent protected Studio generation.

These are conservative cost reservations and usage-derived upper bounds, NEVER
provider invoices. No token-counting call, local token estimate, tools, automatic
retry, reset, or external network request is made by this module.

The caller reserves while holding its run-start write transaction and executes
that exact request once. Unknown outcomes remain reserved across process restarts.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import re
import time

from flask import Blueprint, jsonify, request
from flask_login import current_user, login_required


@dataclass(frozen=True)
class Pricing:
    version: str
    model: str
    service_tier: str
    validated_at: int
    expires_at: int
    input_tokens_max: int
    short_context_max: int
    short_input_micro_usd_per_million: int
    short_output_micro_usd_per_million: int
    long_input_micro_usd_per_million: int
    long_output_micro_usd_per_million: int


# Checked against both official sources on 2026-10-13. Input uses the highest
# applicable cache-write rate, even when the provider may actually charge less.
# https://developers.openai.com/api/docs/models/gpt-6-astra
# https://developers.openai.com/api/docs/pricing
PRICING = Pricing(
    version='openai-gpt-6-astra-standard-2026-10-13-v1',
    model='gpt-6-astra', service_tier='default',
    validated_at=1791849600, expires_at=1792454400,  # Oct 13 to Oct 20, 00:00 UTC
    input_tokens_max=1_050_000, short_context_max=272_000,
    short_input_micro_usd_per_million=12_500_000,
    short_output_micro_usd_per_million=50_000_000,
    long_input_micro_usd_per_million=25_000_000,
    long_output_micro_usd_per_million=75_000_000,
)
MAX_LIMIT_MICRO_USD = 1_000_000_000_000
SCOPE = ('Cumulative limit for subsequent protected Studio generation only. No automatic reset. '
         'Excludes prior and uncapped calls, Casework, infrastructure, taxes, and future tools. '
         'Reservations and usage-derived upper bounds are not actual billed cost.')
REQUEST_FIELDS = frozenset(('model', 'store', 'service_tier', 'reasoning', 'instructions',
                            'input', 'max_output_tokens', 'truncation'))


class BudgetError(ValueError):
    def __init__(self, code, message, status_code=409, **details):
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.details = details


def _now():
    return time.time()


def _iso(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat().replace('+00:00', 'Z')


def _pricing():
    price = PRICING
    numeric = ('validated_at', 'expires_at', 'input_tokens_max', 'short_context_max',
               'short_input_micro_usd_per_million', 'short_output_micro_usd_per_million',
               'long_input_micro_usd_per_million', 'long_output_micro_usd_per_million')
    if (not isinstance(price, Pricing) or price.model != 'gpt-6-astra'
            or price.service_tier != 'default' or not isinstance(price.version, str)
            or not price.version or any(type(getattr(price, key)) is not int
                                        or getattr(price, key) <= 0 for key in numeric)
            or price.input_tokens_max != 1_050_000
            or price.short_context_max >= price.input_tokens_max
            or price.long_input_micro_usd_per_million < price.short_input_micro_usd_per_million
            or price.long_output_micro_usd_per_million < price.short_output_micro_usd_per_million
            or price.validated_at >= price.expires_at):
        raise BudgetError('budget_pricing_unavailable',
                          'Verified pricing is unavailable. Protected generation is blocked.')
    if _now() < price.validated_at:
        raise BudgetError('budget_pricing_unavailable',
                          'Pricing is not yet valid. Protected generation is blocked.')
    if _now() >= price.expires_at:
        raise BudgetError('budget_pricing_expired',
                          'Verified pricing has expired. Protected generation is blocked until revalidated.')
    return price


def _price_status():
    try:
        price = _pricing()
        state = 'current'
    except BudgetError as exc:
        price = PRICING
        state = exc.code
    if not isinstance(price, Pricing) or state == 'budget_pricing_unavailable':
        return dict(status=state, version=None, validated_at=None, expires_at=None)
    return dict(status=state, version=price.version, validated_at=_iso(price.validated_at),
                expires_at=_iso(price.expires_at))


def request_hash(request_payload):
    """Canonical identity of the exact server-prepared wire request, not a proposal."""
    encoded = json.dumps(request_payload, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=True, allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _cost(input_tokens, output_tokens, input_rate, output_rate):
    # All amounts are integer micro-USD; round upward without binary floats.
    return (input_tokens * input_rate + output_tokens * output_rate + 999_999) // 1_000_000


def quote_request(request_payload):
    """No provider contact: full-context bound instead of guessing prompt tokens."""
    price = _pricing()
    if not isinstance(request_payload, dict) or set(request_payload) != REQUEST_FIELDS:
        raise BudgetError('budget_unsupported_request',
                          'Protected generation requires the exact supported no-tool request fields.')
    if request_payload['model'] != price.model:
        raise BudgetError('budget_unsupported_model', 'This model has no verified protected-run price.')
    if request_payload['service_tier'] != price.service_tier:
        raise BudgetError('budget_unsupported_tier', 'Only the standard service tier is priced.')
    if (request_payload['store'] is not False or request_payload['reasoning'] != {'effort': 'low'}
            or request_payload['truncation'] != 'disabled'
            or not isinstance(request_payload['instructions'], str)
            or not isinstance(request_payload['input'], str)
            or type(request_payload['max_output_tokens']) is not int
            or not 128 <= request_payload['max_output_tokens'] <= 1200):
        raise BudgetError('budget_unsupported_request',
                          'The request exceeds the protected generation contract.')
    output_max = request_payload['max_output_tokens']
    reserved = _cost(price.input_tokens_max, output_max,
                     price.long_input_micro_usd_per_million,
                     price.long_output_micro_usd_per_million)
    return dict(currency='USD', reserve_micro_usd=reserved, input_token_upper_bound=price.input_tokens_max,
                output_token_upper_bound=output_max, pricing_version=price.version,
                validated_at=_iso(price.validated_at), expires_at=_iso(price.expires_at),
                request_hash=request_hash(request_payload), actual_cost_usd=None,
                cost_status='conservative_reservation',
                note='Full model context reserved without a token-counting call. Output includes '
                     'reasoning tokens. Actual billed cost is unavailable.')


def initialize_schema(db):
    # Individual statements preserve the caller's transaction; executescript would commit it.
    db.execute('''CREATE TABLE IF NOT EXISTS studio_budget_settings (
        owner TEXT PRIMARY KEY, enabled INTEGER NOT NULL CHECK(enabled IN (0,1)),
        limit_micro_usd INTEGER NOT NULL CHECK(limit_micro_usd >= 0),
        halted INTEGER NOT NULL DEFAULT 0 CHECK(halted IN (0,1)),
        halt_reason TEXT, updated REAL NOT NULL)''')
    db.execute('''CREATE TABLE IF NOT EXISTS studio_budget_reservations (
        owner TEXT NOT NULL, run_id TEXT NOT NULL, request_hash TEXT NOT NULL,
        model TEXT NOT NULL, service_tier TEXT NOT NULL, pricing_version TEXT NOT NULL,
        input_tokens_max INTEGER NOT NULL, output_tokens_max INTEGER NOT NULL,
        input_rate_micro_usd_per_million INTEGER NOT NULL,
        output_rate_micro_usd_per_million INTEGER NOT NULL,
        reserve_micro_usd INTEGER NOT NULL CHECK(reserve_micro_usd >= 0),
        held_micro_usd INTEGER NOT NULL CHECK(held_micro_usd >= 0),
        committed_upper_micro_usd INTEGER NOT NULL DEFAULT 0 CHECK(committed_upper_micro_usd >= 0),
        status TEXT NOT NULL CHECK(status IN ('held','committed_upper_bound','violation')),
        issue TEXT, created REAL NOT NULL, finished REAL,
        PRIMARY KEY(owner,run_id))''')
    db.execute('''CREATE TRIGGER IF NOT EXISTS studio_budget_identity_immutable
        BEFORE UPDATE OF owner,run_id,request_hash,model,service_tier,pricing_version,
        input_tokens_max,output_tokens_max,input_rate_micro_usd_per_million,
        output_rate_micro_usd_per_million,reserve_micro_usd,created
        ON studio_budget_reservations BEGIN
        SELECT RAISE(ABORT,'Budget reservation identity is immutable'); END''')
    db.execute('''CREATE TRIGGER IF NOT EXISTS studio_budget_reservations_no_delete
        BEFORE DELETE ON studio_budget_reservations BEGIN
        SELECT RAISE(ABORT,'Budget reservations cannot be reset'); END''')


def _write_lock(db, owner):
    if not db.in_transaction:
        raise RuntimeError('Studio budgets require an active write transaction.')
    if not isinstance(owner, str) or not owner:
        raise ValueError('An authenticated owner is required.')
    # Also acquire a database write lock if a caller accidentally used deferred BEGIN.
    db.execute('UPDATE studio_budget_settings SET owner=owner WHERE owner=?', (owner,))


def budget_status(db, owner):
    if not isinstance(owner, str) or not owner:
        raise ValueError('An authenticated owner is required.')
    row = db.execute('SELECT * FROM studio_budget_settings WHERE owner=?', (owner,)).fetchone()
    totals = db.execute('''SELECT COALESCE(SUM(held_micro_usd),0) AS held,
        COALESCE(SUM(committed_upper_micro_usd),0) AS committed, COUNT(*) AS protected_runs
        FROM studio_budget_reservations WHERE owner=?''', (owner,)).fetchone()
    limit = row['limit_micro_usd'] if row else 0
    used = totals['held'] + totals['committed']
    return dict(currency='USD', enabled=bool(row and row['enabled']), limit_micro_usd=limit,
                held_micro_usd=totals['held'], committed_upper_micro_usd=totals['committed'],
                used_upper_micro_usd=used, available_micro_usd=max(0, limit-used),
                protected_runs=totals['protected_runs'], halted=bool(row and row['halted']),
                halt_reason=row['halt_reason'] if row else None,
                actual_cost_usd=None, cost_status='actual_billing_unavailable',
                pricing=_price_status(), scope=SCOPE)


def set_budget(db, owner, *, enabled, limit_micro_usd):
    _write_lock(db, owner)
    if type(enabled) is not bool or type(limit_micro_usd) is not int or not 0 <= limit_micro_usd <= MAX_LIMIT_MICRO_USD:
        raise BudgetError('budget_invalid_settings',
                          'Provide enabled as a boolean and a nonnegative integer micro-USD limit.', 400)
    current = budget_status(db, owner)
    if limit_micro_usd < current['used_upper_micro_usd']:
        raise BudgetError('budget_below_commitments',
                          'The limit cannot be below existing held and committed upper bounds.',
                          minimum_micro_usd=current['used_upper_micro_usd'])
    db.execute('''INSERT INTO studio_budget_settings(owner,enabled,limit_micro_usd,updated)
        VALUES(?,?,?,?) ON CONFLICT(owner) DO UPDATE SET
        enabled=excluded.enabled,limit_micro_usd=excluded.limit_micro_usd,updated=excluded.updated''',
        (owner, int(enabled), limit_micro_usd, _now()))
    return budget_status(db, owner)


def _reservation_view(row):
    return {key: row[key] for key in ('run_id', 'request_hash', 'pricing_version',
            'reserve_micro_usd', 'held_micro_usd', 'committed_upper_micro_usd', 'status', 'issue')}


def reserve_for_run(db, owner, run_id, request_payload):
    """Caller commits this reservation and running state together, before any call."""
    _write_lock(db, owner)
    if not isinstance(run_id, str) or not run_id:
        raise ValueError('A server-generated run ID is required.')
    current = budget_status(db, owner)
    if not current['enabled']:
        return None
    if current['halted']:
        raise BudgetError('budget_halted',
                          'Protected generation is halted because a prior provider result violated its bound.')
    quote = quote_request(request_payload)
    prior = db.execute('SELECT * FROM studio_budget_reservations WHERE owner=? AND run_id=?',
                       (owner, run_id)).fetchone()
    if prior:
        if prior['request_hash'] != quote['request_hash']:
            raise BudgetError('budget_request_changed', 'This run already has a different immutable request.')
        return _reservation_view(prior)
    if quote['reserve_micro_usd'] > current['available_micro_usd']:
        raise BudgetError('budget_exceeded',
                          'The remaining budget is below this conservative request reservation.', 429,
                          required_micro_usd=quote['reserve_micro_usd'],
                          available_micro_usd=current['available_micro_usd'])
    price = _pricing()
    db.execute('''INSERT INTO studio_budget_reservations(
        owner,run_id,request_hash,model,service_tier,pricing_version,input_tokens_max,
        output_tokens_max,input_rate_micro_usd_per_million,output_rate_micro_usd_per_million,
        reserve_micro_usd,held_micro_usd,status,created) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
        (owner, run_id, quote['request_hash'], price.model, price.service_tier, price.version,
         price.input_tokens_max, request_payload['max_output_tokens'],
         price.long_input_micro_usd_per_million, price.long_output_micro_usd_per_million,
         quote['reserve_micro_usd'], quote['reserve_micro_usd'], 'held', _now()))
    return _reservation_view(db.execute('SELECT * FROM studio_budget_reservations '
                                       'WHERE owner=? AND run_id=?', (owner, run_id)).fetchone())


def _receipt_check(row, result):
    """Return (issue, violation, usage); absence cannot be interpreted as zero."""
    if not isinstance(result, dict):
        return 'unknown_provider_outcome', False, None
    if result.get('usage_invalid') is True:
        return 'invalid_provider_usage', True, None
    expected = dict(provider='OpenAI', model=row['model'], service_tier=row['service_tier'],
                    request_hash=row['request_hash'])
    for key, value in expected.items():
        if result.get(key) is not None and result[key] != value:
            return 'provider_identity_mismatch', True, None
    calls = result.get('provider_calls')
    if calls is not None and (type(calls) is not int or calls != 1):
        return 'provider_call_count_mismatch', True, None
    usage = result.get('usage')
    if isinstance(usage, dict):
        for key in ('input_tokens', 'output_tokens', 'total_tokens'):
            if key in usage and (type(usage[key]) is not int or usage[key] < 0):
                return 'invalid_provider_usage', True, None
        if usage.get('input_tokens', 0) > row['input_tokens_max'] or usage.get('output_tokens', 0) > row['output_tokens_max']:
            return 'provider_usage_exceeded_bound', True, None
        if all(key in usage for key in ('input_tokens', 'output_tokens', 'total_tokens')):
            if usage['total_tokens'] != usage['input_tokens'] + usage['output_tokens']:
                return 'provider_usage_mismatch', True, None
            for group, key, total in (('input_tokens_details', 'cached_tokens', 'input_tokens'),
                                      ('output_tokens_details', 'reasoning_tokens', 'output_tokens')):
                if group in usage:
                    detail = usage[group]
                    if not isinstance(detail, dict) or (key in detail and
                            (type(detail[key]) is not int or not 0 <= detail[key] <= usage[total])):
                        return 'provider_usage_mismatch', True, None
    if (any(result.get(key) != value for key, value in expected.items()) or calls != 1
            or result.get('provider_status') != 'completed'
            or not isinstance(result.get('response_id'), str)
            or not re.fullmatch(r'[a-zA-Z0-9._:-]{1,200}', result['response_id'])
            or not isinstance(usage, dict)
            or not all(key in usage for key in ('input_tokens', 'output_tokens', 'total_tokens'))):
        return 'unknown_provider_outcome', False, None
    return None, False, usage


def finish_reservation(db, owner, run_id, result):
    """Commit an upper bound only from complete terminal metadata, never an invoice.

    This does not throw for provider anomalies: its halt and original full hold
    must commit with the run result instead of being lost in a rollback.
    """
    _write_lock(db, owner)
    row = db.execute('SELECT * FROM studio_budget_reservations WHERE owner=? AND run_id=?',
                     (owner, run_id)).fetchone()
    if not row:
        return None
    if row['status'] != 'held':
        return _reservation_view(row)
    issue, violation, usage = _receipt_check(row, result)
    if violation:
        db.execute('UPDATE studio_budget_settings SET halted=1,halt_reason=?,updated=? WHERE owner=?',
                   (issue, _now(), owner))
        db.execute("UPDATE studio_budget_reservations SET status='violation',issue=?,finished=? "
                   'WHERE owner=? AND run_id=?', (issue, _now(), owner, run_id))
    elif issue:
        db.execute('UPDATE studio_budget_reservations SET issue=? WHERE owner=? AND run_id=?',
                   (issue, owner, run_id))
    else:
        bound = _cost(usage['input_tokens'], usage['output_tokens'],
                      row['input_rate_micro_usd_per_million'], row['output_rate_micro_usd_per_million'])
        # Use the reserved snapshot and worst supported tier, including for small
        # inputs and cached tokens. A price refresh cannot retroactively shrink it.
        db.execute("UPDATE studio_budget_reservations SET status='committed_upper_bound',"
                   'held_micro_usd=0,committed_upper_micro_usd=?,issue=NULL,finished=? '
                   'WHERE owner=? AND run_id=?', (bound, _now(), owner, run_id))
    return _reservation_view(db.execute('SELECT * FROM studio_budget_reservations '
                                       'WHERE owner=? AND run_id=?', (owner, run_id)).fetchone())


def register_studio_budgets(app):
    from .studio import protect_mutations, role, store
    with app.extensions['studio_store'].connect() as db:
        initialize_schema(db)
    bp = Blueprint('studio_budgets', __name__)
    bp.before_request(protect_mutations)

    @bp.after_request
    def private_response(response):
        response.headers['Cache-Control'] = 'no-store'
        return response

    @bp.route('/api/studio/budget', methods=['GET', 'POST'])
    @login_required
    def budget_settings():
        if request.args:
            return jsonify(error='Budget scope comes only from the signed-in account.',
                           code='budget_invalid_scope'), 400
        owner = current_user.get_id()
        if request.method == 'GET':
            with store().connect() as db:
                return jsonify(budget_status(db, owner))
        if role() not in {'operator', 'admin'}:
            return jsonify(error='An operator role is required to change your budget.'), 403
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or set(data) != {'enabled', 'limit_micro_usd'}:
            return jsonify(error='Provide only enabled and limit_micro_usd.',
                           code='budget_invalid_settings'), 400
        try:
            with store().connect() as db:
                db.execute('BEGIN IMMEDIATE')
                result = set_budget(db, owner, **data)
                store().audit(db, owner, 'studio_budget_updated', detail={
                    'enabled': result['enabled'], 'limit_micro_usd': result['limit_micro_usd']})
            return jsonify(result)
        except BudgetError as exc:
            return jsonify(error=str(exc), code=exc.code, **exc.details), exc.status_code

    app.register_blueprint(bp)
