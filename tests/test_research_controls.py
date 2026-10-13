"""Offline tests only: no real collectors, billing, provider calls or route wiring."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from dredge.research_controls import (
    Bounds, BudgetLedger, PriceCard, Scope, compare_source, plan_sections, quote_cost, snapshot,
)

SCOPE = Scope('account-a', 'case-a')
CARD = PriceCard('model-fixture', 'fixture-v1', 100, 200, '2.5', '10', 150)
BOUNDS = Bounds(100, 20, 1)


def quote():
    return quote_cost('model-fixture', BOUNDS, CARD, 110)


def observed(content=b'content', at=110, receipt='new', scope=SCOPE, version='v1'):
    return snapshot(scope, 'source-a', content, at, receipt, version)


def test_source_digest_verification_not_age():
    old = observed(at=10, receipt='old')
    assert compare_source(SCOPE, old, observed(), 100, 120) == 'unchanged'
    assert compare_source(SCOPE, old, observed(b'changed'), 100, 120) == 'changed'
    assert compare_source(SCOPE, old, old, 100, 120) == 'unknown'
    assert compare_source(SCOPE, old, observed(at=99), 100, 120) == 'unknown'
    assert compare_source(SCOPE, old, observed(at=121), 100, 120) == 'unknown'
    assert compare_source(SCOPE, old, observed(receipt='old'), 100, 120) == 'unknown'
    assert compare_source(SCOPE, old, observed(version='v2'), 100, 120) == 'unknown'
    assert compare_source(SCOPE, old, None, 100, 120) == 'unknown'
    assert compare_source(SCOPE, None, observed(), 100, 120) == 'unknown'
    with pytest.raises(ValueError):
        compare_source(SCOPE, old, observed(scope=Scope('account-b', 'case-a')), 100, 120)
    with pytest.raises(ValueError):
        compare_source(SCOPE, old, observed(scope=Scope('account-a', 'case-b')), 100, 120)
    with pytest.raises(ValueError):
        replace(old, digest='invalid')


def test_refresh_only_changed_dependencies_and_block_unknown():
    assert plan_sections({'a': ['one'], 'b': ['one', 'two'], 'c': ['missing'], 'd': []},
                         {'one': 'unchanged', 'two': 'changed'}) == {
        'refresh': ['b'], 'reuse_candidates': ['a'], 'blocked': ['c', 'd']}
    assert plan_sections({'a': ['one', 'two']}, {'one': 'changed'})['blocked'] == ['a']
    with pytest.raises(ValueError):
        plan_sections({'a': ['one']}, {'one': 'probably_fresh'})


def test_quote_includes_tool_cost_rounds_up_and_is_not_actual_billing():
    assert quote().upper_micro_usd == 600  # 250 input + 200 output + 150 tool
    tiny = replace(CARD, input_usd_per_million='0.00001')
    assert quote_cost('model-fixture', Bounds(1, 0, 0), tiny, 110).upper_micro_usd == 1
    assert quote_cost('model-fixture', Bounds(0, 0, 0), CARD, 110).upper_micro_usd == 0


@pytest.mark.parametrize('card', [None, replace(CARD, model='other'),
    replace(CARD, valid_until=110), replace(CARD, valid_from=111),
    replace(CARD, input_usd_per_million='NaN'),
    replace(CARD, output_usd_per_million='Infinity'),
    replace(CARD, output_usd_per_million='-1'),
    replace(CARD, tool_call_micro_usd=None)])
def test_untrusted_or_unavailable_pricing_fails_closed(card):
    with pytest.raises(ValueError):
        quote_cost('model-fixture', BOUNDS, card, 110)


@pytest.mark.parametrize('value', [-1, True, 1.5, 2**63])
def test_invalid_bounds(value):
    with pytest.raises(ValueError):
        Bounds(value, 10, 1)


def test_atomic_parallel_reservations_and_persistence(tmp_path):
    path = tmp_path / 'ledger.db'
    ledger = BudgetLedger(path)
    ledger.set_limit(SCOPE, 600)
    def reserve(number):
        try:
            return BudgetLedger(path).reserve(SCOPE, str(number), quote(), 120)
        except ValueError:
            return False
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(reserve, range(16)))
    assert sum(results) == 1
    with pytest.raises(ValueError):
        BudgetLedger(path).reserve(SCOPE, 'after-restart', quote(), 120)
    with pytest.raises(ValueError):
        ledger.set_limit(SCOPE, 599)


def test_same_run_idempotency_does_not_authorize_second_dispatch(tmp_path):
    ledger = BudgetLedger(tmp_path / 'ledger.db')
    ledger.set_limit(SCOPE, 10000)
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: ledger.reserve(SCOPE, 'same', quote(), 120), range(8)))
    assert sum(results) == 1
    with pytest.raises(ValueError):
        ledger.reserve(SCOPE, 'same', replace(quote(), price_version='tampered'), 120)
    with pytest.raises(ValueError):
        ledger.reserve(SCOPE, 'same', quote(), 200)


def test_scope_isolation_canonical_keys_and_missing_budget(tmp_path):
    ledger = BudgetLedger(tmp_path / 'ledger.db')
    ledger.set_limit(SCOPE, 600)
    assert ledger.reserve(SCOPE, 'same', quote(), 120)
    for other in [Scope('account-b', 'case-a'), Scope('account-a', 'case-b')]:
        with pytest.raises(ValueError):
            ledger.reserve(other, 'same', quote(), 120)
        with pytest.raises(ValueError):
            ledger.settle(other, 'same', 0, 'receipt')
        ledger.set_limit(other, 600)
        assert ledger.reserve(other, 'same', quote(), 120)
    assert Scope('a:b', 'c').key != Scope('a', 'b:c').key


def test_uncertain_usage_stays_held_final_charge_releases_difference(tmp_path):
    ledger = BudgetLedger(tmp_path / 'ledger.db')
    ledger.set_limit(SCOPE, 900)
    ledger.reserve(SCOPE, 'first', quote(), 120)
    # A timeout or cancellation after dispatch cannot release this hold.
    with pytest.raises(ValueError):
        ledger.reserve(SCOPE, 'second', quote(), 120)
    ledger.settle(SCOPE, 'first', 300, 'final-charge-first')
    ledger.settle(SCOPE, 'first', 300, 'final-charge-first')
    assert ledger.reserve(SCOPE, 'second', quote(), 120)
    with pytest.raises(ValueError):
        ledger.settle(SCOPE, 'first', 0, 'different-receipt')


def test_trusted_no_charge_receipt_can_release_hold(tmp_path):
    ledger = BudgetLedger(tmp_path / 'ledger.db')
    ledger.set_limit(SCOPE, 600)
    ledger.reserve(SCOPE, 'cancelled', quote(), 120)
    ledger.settle(SCOPE, 'cancelled', 0, 'server-proved-never-dispatched')
    assert ledger.reserve(SCOPE, 'next', quote(), 120)
    assert not ledger.reserve(SCOPE, 'cancelled', quote(), 120)


def test_overrun_durably_halts_scope_even_after_limit_increase(tmp_path):
    path = tmp_path / 'ledger.db'
    ledger = BudgetLedger(path)
    ledger.set_limit(SCOPE, 6000)
    ledger.reserve(SCOPE, 'first', quote(), 120)
    with pytest.raises(ValueError, match='halted'):
        ledger.settle(SCOPE, 'first', 601, 'overrun')
    restarted = BudgetLedger(path)
    restarted.set_limit(SCOPE, 10000)
    with pytest.raises(ValueError, match='halted'):
        restarted.reserve(SCOPE, 'second', quote(), 120)


def test_concurrent_cap_edits_cannot_undercut_reservations(tmp_path):
    ledger = BudgetLedger(tmp_path / 'ledger.db')
    ledger.set_limit(SCOPE, 600)
    def lower():
        try:
            ledger.set_limit(SCOPE, 0)
            return True
        except ValueError:
            return False
    def reserve():
        try:
            return ledger.reserve(SCOPE, 'first', quote(), 120)
        except ValueError:
            return False
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(lower)
        second = executor.submit(reserve)
        assert first.result() != second.result()


def test_reject_numerical_underflow_and_overflow():
    with pytest.raises(ValueError):
        quote_cost('model-fixture', BOUNDS,
                   replace(CARD, input_usd_per_million='1e-999999999'), 110)
    with pytest.raises(ValueError):
        quote_cost('model-fixture', Bounds(2**63-1, 0, 0), CARD, 110)


def test_one_final_receipt_cannot_settle_two_reservations(tmp_path):
    import sqlite3
    ledger = BudgetLedger(tmp_path / 'ledger.db')
    ledger.set_limit(SCOPE, 1200)
    ledger.reserve(SCOPE, 'first', quote(), 120)
    ledger.reserve(SCOPE, 'second', quote(), 120)
    ledger.settle(SCOPE, 'first', 300, 'charge-1')
    with pytest.raises(sqlite3.IntegrityError):
        ledger.settle(SCOPE, 'second', 300, 'charge-1')
    with pytest.raises(ValueError):
        ledger.set_limit(SCOPE, 600)  # first settled 300 + second still held 600
