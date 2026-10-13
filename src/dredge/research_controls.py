"""Offline planning primitives. No routes, fetching, provider calls, or pricing defaults.

Only trusted server adapters may supply snapshots, token bounds, prices, or receipts.
These types are not authentication or proof that an external source was fetched.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_CEILING, localcontext
import hashlib
import json
import sqlite3


MAX_INTEGER = 2**63 - 1


def integer(value, name):
    if type(value) is not int or not 0 <= value <= MAX_INTEGER:
        raise ValueError(f'{name} must be a nonnegative bounded integer')
    return value


def label(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError('A nonempty bounded identifier is required')
    return value


@dataclass(frozen=True)
class Scope:
    account_id: str
    case_id: str

    def __post_init__(self):
        label(self.account_id)
        label(self.case_id)

    @property
    def key(self):
        return json.dumps([self.account_id, self.case_id], separators=(',', ':'))


@dataclass(frozen=True)
class Snapshot:
    scope: Scope
    source_id: str
    digest: str
    observed_at: int
    receipt_id: str
    collector_version: str

    def __post_init__(self):
        label(self.source_id)
        label(self.receipt_id)
        label(self.collector_version)
        integer(self.observed_at, 'observed_at')
        if not isinstance(self.digest, str) or len(self.digest) != 64 or any(
            c not in '0123456789abcdef' for c in self.digest
        ):
            raise ValueError('Expected a SHA-256 content digest')


def snapshot(scope, source_id, content, observed_at, receipt_id, collector_version):
    """Hash exact bytes supplied by an authorized collector; never fetch a URL.

    A receipt must identify a successful source observation, not a search citation,
    cached answer, client claim, or HTTP failure. Enforce that at the adapter boundary.
    """
    if not isinstance(content, bytes):
        raise ValueError('Snapshot content must be bytes')
    return Snapshot(scope, source_id, hashlib.sha256(content).hexdigest(), observed_at,
                    receipt_id, collector_version)


def compare_source(scope, previous, current, verification_started_at, now):
    """Compare observations during this verification, never infer freshness from age."""
    integer(verification_started_at, 'verification_started_at')
    integer(now, 'now')
    if verification_started_at > now:
        raise ValueError('Verification cannot start in the future')
    if previous is None or current is None:
        return 'unknown'
    if previous.scope != scope or current.scope != scope:
        raise ValueError('Snapshot scope mismatch')
    if (previous.source_id != current.source_id
            or previous.collector_version != current.collector_version
            or previous.observed_at > current.observed_at
            or previous.receipt_id == current.receipt_id
            or not verification_started_at <= current.observed_at <= now):
        return 'unknown'
    return 'unchanged' if previous.digest == current.digest else 'changed'


def plan_sections(dependencies, source_states):
    """Return a plan, not a rewritten answer or permission to reuse an answer.

    Dependencies must be a complete, trusted section-to-source map. Unmapped sections
    and unknown sources are blocked; claim correctness still needs human review.
    """
    if any(state not in {'changed', 'unchanged', 'unknown'} for state in source_states.values()):
        raise ValueError('Invalid source state')
    plan = {'refresh': [], 'reuse_candidates': [], 'blocked': []}
    for section, sources in dependencies.items():
        label(section)
        if not isinstance(sources, (list, tuple, set)):
            raise ValueError('Section dependencies must be source identifiers')
        states = [source_states.get(label(source), 'unknown') for source in sources]
        bucket = ('blocked' if not states or 'unknown' in states else
                  'refresh' if 'changed' in states else 'reuse_candidates')
        plan[bucket].append(section)
    return plan


@dataclass(frozen=True)
class PriceCard:
    """Server-verified USD prices, including every billable input/tool component."""
    model: str
    version: str
    valid_from: int
    valid_until: int
    input_usd_per_million: str
    output_usd_per_million: str
    tool_call_micro_usd: int


@dataclass(frozen=True)
class Bounds:
    """Certified execution limits, not character-count token guesses.

    Input must cover system instructions, history and all tool-fed/context tokens.
    Output includes reasoning tokens; tools must have a separately enforceable cap.
    """
    input_tokens: int
    output_tokens: int
    tool_calls: int

    def __post_init__(self):
        for name in ('input_tokens', 'output_tokens', 'tool_calls'):
            integer(getattr(self, name), name)


@dataclass(frozen=True)
class Quote:
    model: str
    price_version: str
    valid_until: int
    bounds: Bounds
    upper_micro_usd: int


def quote_cost(model, bounds, prices, now):
    """Fail closed if trusted pricing or certified bounds are unavailable."""
    label(model)
    integer(now, 'now')
    if not isinstance(bounds, Bounds) or not isinstance(prices, PriceCard):
        raise ValueError('Trusted pricing and certified token/tool bounds are required')
    label(prices.version)
    integer(prices.valid_from, 'valid_from')
    integer(prices.valid_until, 'valid_until')
    if prices.model != model or not prices.valid_from <= now < prices.valid_until:
        raise ValueError('Pricing unavailable for this model and time')
    rates = []
    for value in (prices.input_usd_per_million, prices.output_usd_per_million):
        if not isinstance(value, str) or len(value) > 64:
            raise ValueError('Prices must be finite nonnegative decimal strings')
        try:
            rate = Decimal(value)
        except InvalidOperation:
            raise ValueError('Invalid price') from None
        if (not rate.is_finite() or rate < 0 or rate > MAX_INTEGER
                or rate.as_tuple().exponent < -18):
            raise ValueError('Invalid price')
        rates.append(rate)
    integer(prices.tool_call_micro_usd, 'tool_call_micro_usd')
    # One USD per million tokens equals one micro-USD per token. Round UP.
    with localcontext() as context:
        context.prec = 128
        amount = (bounds.input_tokens * rates[0] + bounds.output_tokens * rates[1]
                  + bounds.tool_calls * prices.tool_call_micro_usd)
        total = integer(int(amount.to_integral_value(rounding=ROUND_CEILING)), 'quote')
    return Quote(model, prices.version, prices.valid_until, bounds, total)


class BudgetLedger:
    """SQLite single-database reservations, atomic across connections/processes.

    The app must authenticate Scope and run ID before calling. This is not a global
    provider/account limit: all participating execution paths must use this ledger.
    No reservation expires automatically; unknown completion keeps funds held.
    """
    def __init__(self, path):
        self.path = str(path)
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS research_budgets (
                    scope TEXT PRIMARY KEY, limit_micro INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS research_budget_halts (
                    scope TEXT PRIMARY KEY);
                CREATE TABLE IF NOT EXISTS research_reservations (
                    scope TEXT NOT NULL, run_id TEXT NOT NULL, fingerprint TEXT NOT NULL,
                    reserved_micro INTEGER NOT NULL, settled_micro INTEGER,
                    receipt_id TEXT, PRIMARY KEY(scope, run_id));
                CREATE UNIQUE INDEX IF NOT EXISTS research_final_receipts
                    ON research_reservations(receipt_id) WHERE receipt_id IS NOT NULL;
            ''')

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def used(db, scope):
        return db.execute('''SELECT COALESCE(SUM(COALESCE(settled_micro, reserved_micro)), 0)
                             FROM research_reservations WHERE scope=?''', (scope.key,)).fetchone()[0]

    def set_limit(self, scope, limit_micro):
        """Use only for an authenticated user's explicit choice of this scope's limit."""
        integer(limit_micro, 'limit_micro')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if self.used(db, scope) > limit_micro:
                raise ValueError('Limit is below spent or reserved funds')
            db.execute('INSERT INTO research_budgets VALUES (?, ?) ON CONFLICT(scope) '
                       'DO UPDATE SET limit_micro=excluded.limit_micro', (scope.key, limit_micro))

    def reserve(self, scope, run_id, quote, now):
        """Returns True only for a new reservation; False means do NOT execute again."""
        label(run_id)
        integer(now, 'now')
        if not isinstance(quote, Quote) or now >= quote.valid_until:
            raise ValueError('A current trusted quote is required')
        integer(quote.upper_micro_usd, 'upper_micro_usd')
        fingerprint = json.dumps([quote.model, quote.price_version, quote.valid_until,
                                  quote.bounds.input_tokens, quote.bounds.output_tokens,
                                  quote.bounds.tool_calls, quote.upper_micro_usd])
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            prior = db.execute('SELECT fingerprint FROM research_reservations '
                               'WHERE scope=? AND run_id=?', (scope.key, run_id)).fetchone()
            if prior:
                if prior[0] != fingerprint:
                    raise ValueError('Run reservation already has a different quote')
                return False
            if db.execute('SELECT 1 FROM research_budget_halts WHERE scope=?',
                          (scope.key,)).fetchone():
                raise ValueError('Budget halted pending charge investigation')
            budget = db.execute('SELECT limit_micro FROM research_budgets WHERE scope=?',
                                (scope.key,)).fetchone()
            if not budget or self.used(db, scope) + quote.upper_micro_usd > budget[0]:
                raise ValueError('Budget unavailable or exhausted')
            db.execute('INSERT INTO research_reservations '
                       '(scope, run_id, fingerprint, reserved_micro) VALUES (?, ?, ?, ?)',
                       (scope.key, run_id, fingerprint, quote.upper_micro_usd))
            return True

    def settle(self, scope, run_id, actual_micro, receipt_id):
        """Reconcile only an authoritative final charge, never estimated token cost.

        Zero is valid only with trusted proof no charge occurred. Unknown billing
        must not call this method. Overruns reject reconciliation and retain the
        full hold and durably halt new reservations for that scope. There is no
        automatic unfreeze; investigation and reconciliation are required.
        """
        integer(actual_micro, 'actual_micro')
        label(run_id)
        label(receipt_id)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT reserved_micro, settled_micro, receipt_id '
                             'FROM research_reservations WHERE scope=? AND run_id=?',
                             (scope.key, run_id)).fetchone()
            if not row:
                raise ValueError('Reservation not found')
            if row[1] is not None:
                if (row[1], row[2]) != (actual_micro, receipt_id):
                    raise ValueError('Conflicting final charge')
                return
            if actual_micro > row[0]:
                db.execute('INSERT OR IGNORE INTO research_budget_halts VALUES (?)',
                           (scope.key,))
                db.commit()
                raise ValueError('Charge exceeds certified bound; budget halted')
            db.execute('UPDATE research_reservations SET settled_micro=?, receipt_id=? '
                       'WHERE scope=? AND run_id=?', (actual_micro, receipt_id, scope.key, run_id))
