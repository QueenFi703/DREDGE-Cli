"""Budget contract tests use temporary SQLite stores and never contact a provider."""
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import threading

import pytest

from dredge import studio_budgets as budgets
from dredge.studio import StudioStore
from test_studio import app, client_for


@pytest.fixture(autouse=True)
def current_prices(monkeypatch):
    monkeypatch.setattr(budgets, '_now', lambda: 1791856800.0)  # 2026-10-13 UTC


@pytest.fixture
def request_payload():
    return dict(model='gpt-6-astra', store=False, service_tier='default',
                reasoning={'effort': 'low'}, truncation='disabled', instructions='Public fictional question only.',
                input='A tiny fictional question.', max_output_tokens=1200)


@pytest.fixture
def ledger(tmp_path):
    result = StudioStore(tmp_path / 'budget.sqlite3')
    with result.connect() as db:
        budgets.initialize_schema(db)
        db.execute('BEGIN IMMEDIATE')
        budgets.set_budget(db, 'owner', enabled=True, limit_micro_usd=30_000_000)
    return result


def reserve(ledger, payload, run_id='run', owner='owner'):
    with ledger.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        return budgets.reserve_for_run(db, owner, run_id, payload)


def finish(ledger, result, run_id='run', owner='owner'):
    with ledger.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        return budgets.finish_reservation(db, owner, run_id, result)


def status(ledger, owner='owner'):
    with ledger.connect() as db:
        return budgets.budget_status(db, owner)


def receipt(payload, **changes):
    result = dict(provider='OpenAI', provider_calls=1, response_id='resp_test',
                  model='gpt-6-astra', service_tier='default', provider_status='completed',
                  request_hash=budgets.request_hash(payload),
                  usage={'input_tokens': 100, 'output_tokens': 20, 'total_tokens': 120})
    result.update(changes)
    return result


def test_quote_reserves_full_context_without_guessing_input_tokens(request_payload):
    quoted = budgets.quote_request(request_payload)
    assert quoted['reserve_micro_usd'] == 26_340_000
    assert quoted['input_token_upper_bound'] == 1_050_000
    assert quoted['output_token_upper_bound'] == 1200
    assert quoted['actual_cost_usd'] is None
    request_payload['input'] = '文' * 12000
    assert budgets.quote_request(request_payload)['reserve_micro_usd'] == 26_340_000


def test_enabled_budget_really_allows_sufficiently_budgeted_request(ledger, request_payload):
    reservation = reserve(ledger, request_payload)
    assert reservation['status'] == 'held'
    assert status(ledger)['held_micro_usd'] == 26_340_000
    assert status(ledger)['available_micro_usd'] == 3_660_000
    with pytest.raises(budgets.BudgetError, match='remaining') as caught:
        reserve(ledger, request_payload, 'run-2')
    assert caught.value.code == 'budget_exceeded'


def test_valid_complete_usage_frees_only_verified_excess(ledger, request_payload):
    reserve(ledger, request_payload)
    finished = finish(ledger, receipt(request_payload))
    assert finished['status'] == 'committed_upper_bound'
    # Highest supported long-context input/write rate and output rate, never a bill.
    assert status(ledger)['committed_upper_micro_usd'] == 4000
    assert status(ledger)['held_micro_usd'] == 0
    assert status(ledger)['actual_cost_usd'] is None
    assert reserve(ledger, request_payload, 'run-2')['status'] == 'held'


def test_cross_connection_concurrency_cannot_reserve_twice(ledger, request_payload):
    barrier = threading.Barrier(2)
    def attempt(i):
        barrier.wait()
        try:
            reserve(ledger, request_payload, f'run-{i}')
            return 'allowed'
        except budgets.BudgetError as exc:
            return exc.code
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = sorted(pool.map(attempt, range(2)))
    assert outcomes == ['allowed', 'budget_exceeded']
    assert status(ledger)['held_micro_usd'] == 26_340_000


@pytest.mark.parametrize('changes', [
    {'usage': None}, {'usage': {}}, {'usage': {'input_tokens': 0, 'output_tokens': 0}},
    {'provider_status': 'incomplete'}, {'response_id': None}, {'request_hash': None},
    {'provider_calls': None}, {'service_tier': None},
])
def test_unknown_outcomes_keep_entire_hold(ledger, request_payload, changes):
    reserve(ledger, request_payload)
    finish(ledger, receipt(request_payload, **changes))
    assert status(ledger)['held_micro_usd'] == 26_340_000
    assert status(ledger)['committed_upper_micro_usd'] == 0


@pytest.mark.parametrize('changes', [
    {'model': 'gpt-6-luna'}, {'service_tier': 'priority'}, {'provider_calls': 2},
    {'request_hash': '0' * 64},
    {'usage': {'input_tokens': 1_050_001, 'output_tokens': 0, 'total_tokens': 1_050_001}},
    {'usage': {'input_tokens': 0, 'output_tokens': 1201, 'total_tokens': 1201}},
    {'usage': {'input_tokens': 1, 'output_tokens': 1, 'total_tokens': 0}},
    {'usage': {'input_tokens': False, 'output_tokens': 0, 'total_tokens': 0}},
])
def test_contradictory_metadata_halts_future_protected_calls(ledger, request_payload, changes):
    reserve(ledger, request_payload)
    finish(ledger, receipt(request_payload, **changes))
    assert status(ledger)['halted'] is True
    assert status(ledger)['held_micro_usd'] == 26_340_000
    with pytest.raises(budgets.BudgetError) as caught:
        reserve(ledger, request_payload, 'second')
    assert caught.value.code == 'budget_halted'


def test_zero_usage_requires_complete_identity_before_releasing(ledger, request_payload):
    reserve(ledger, request_payload)
    finish(ledger, receipt(request_payload, response_id=None,
                          usage={'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}))
    assert status(ledger)['held_micro_usd'] == 26_340_000
    finish(ledger, receipt(request_payload,
                          usage={'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}))
    assert status(ledger)['held_micro_usd'] == 0


@pytest.mark.parametrize('changes', [
    {'tools': []}, {'conversation': 'private'}, {'previous_response_id': 'resp_other'},
    {'background': False}, {'stream': False}, {'model': 'gpt-6-luna'},
    {'service_tier': 'auto'}, {'store': True}, {'store': 0},
    {'reasoning': {'effort': 'high'}}, {'input': []}, {'instructions': []},
    {'max_output_tokens': True}, {'max_output_tokens': 127}, {'max_output_tokens': 1201},
])
def test_unpriced_or_expanded_request_shape_fails_closed(ledger, request_payload, changes):
    request_payload.update(changes)
    with pytest.raises(budgets.BudgetError):
        reserve(ledger, request_payload)
    assert status(ledger)['held_micro_usd'] == 0


def test_expired_and_missing_prices_are_distinct(ledger, request_payload, monkeypatch):
    monkeypatch.setattr(budgets, '_now', lambda: budgets.PRICING.expires_at)
    with pytest.raises(budgets.BudgetError) as caught:
        reserve(ledger, request_payload)
    assert caught.value.code == 'budget_pricing_expired'
    monkeypatch.setattr(budgets, 'PRICING', None)
    with pytest.raises(budgets.BudgetError) as caught:
        reserve(ledger, request_payload)
    assert caught.value.code == 'budget_pricing_unavailable'


def test_disabled_owner_is_uncapped_and_retains_proposed_limit(ledger, request_payload):
    reserve(ledger, request_payload)
    with ledger.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        budgets.set_budget(db, 'owner', enabled=False, limit_micro_usd=31_000_000)
    assert reserve(ledger, {'unknown': 'unsupported'}, 'uncapped') is None
    assert status(ledger)['limit_micro_usd'] == 31_000_000
    assert status(ledger)['held_micro_usd'] == 26_340_000
    with ledger.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        budgets.set_budget(db, 'owner', enabled=True, limit_micro_usd=31_000_000)
    assert status(ledger)['held_micro_usd'] == 26_340_000


def test_budget_cannot_be_lowered_below_commitments_even_while_disabled(ledger, request_payload):
    reserve(ledger, request_payload)
    for enabled in (True, False):
        with ledger.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            with pytest.raises(budgets.BudgetError) as caught:
                budgets.set_budget(db, 'owner', enabled=enabled, limit_micro_usd=1)
            assert caught.value.code == 'budget_below_commitments'
    assert status(ledger)['limit_micro_usd'] == 30_000_000


def test_owner_and_hash_are_immutable_and_settlement_is_idempotent(ledger, request_payload):
    held = reserve(ledger, request_payload)
    assert reserve(ledger, request_payload) == held
    assert finish(ledger, receipt(request_payload), owner='other') is None
    assert status(ledger)['held_micro_usd'] == 26_340_000
    finish(ledger, receipt(request_payload))
    assert finish(ledger, receipt(request_payload))['committed_upper_micro_usd'] == 4000
    request_payload['input'] += ' changed'
    with pytest.raises(budgets.BudgetError) as caught:
        reserve(ledger, request_payload)
    assert caught.value.code == 'budget_request_changed'


def test_schema_refuses_immutable_identity_change(ledger, request_payload):
    reserve(ledger, request_payload)
    with ledger.connect() as db:
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("UPDATE studio_budget_reservations SET request_hash='changed'")


def test_setting_and_reservation_race_is_serialized(ledger, request_payload):
    barrier = threading.Barrier(2)
    def operation(kind):
        barrier.wait()
        try:
            if kind == 'reserve':
                reserve(ledger, request_payload)
            else:
                with ledger.connect() as db:
                    db.execute('BEGIN IMMEDIATE')
                    budgets.set_budget(db, 'owner', enabled=True, limit_micro_usd=1)
            return kind
        except budgets.BudgetError as exc:
            return exc.code
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = set(pool.map(operation, ('reserve', 'lower')))
    assert outcomes in ({'reserve', 'budget_below_commitments'}, {'lower', 'budget_exceeded'})
    current = status(ledger)
    assert current['held_micro_usd'] + current['committed_upper_micro_usd'] <= current['limit_micro_usd']


def test_endpoint_owner_scope_roles_csrf_and_exact_schema(app):
    if 'studio_budgets' not in app.blueprints:
        budgets.register_studio_budgets(app)
    client, headers = client_for(app)
    assert app.test_client().get('/api/studio/budget').status_code == 401
    assert client.post('/api/studio/budget', json={'enabled': True, 'limit_micro_usd': 30_000_000}).status_code == 403
    good = {'enabled': True, 'limit_micro_usd': 30_000_000}
    assert client.post('/api/studio/budget', json=good, headers=headers).status_code == 200
    for extra in ({'owner': 'test:other'}, {'run_id': 'run'}, {'role': 'admin'}):
        assert client.post('/api/studio/budget', json=dict(good, **extra), headers=headers).status_code == 400
    other, other_headers = client_for(app, 'test:other')
    assert other.get('/api/studio/budget?owner=test:op').status_code == 400
    assert other.get('/api/studio/budget').json['enabled'] is False
    viewer, viewer_headers = client_for(app, 'test:viewer')
    assert viewer.post('/api/studio/budget', json=good, headers=viewer_headers).status_code == 403
    admin, admin_headers = client_for(app, 'test:admin')
    assert admin.post('/api/studio/budget', json=good, headers=admin_headers).status_code == 200
    assert client.get('/api/studio/session').json['role'] == 'operator'


def _process_reserve(path, payload, run_id):
    """A fresh OS process opens its own connection to the durable ledger."""
    budgets._now = lambda: 1791856800.0
    local = StudioStore(path)
    try:
        reserve(local, payload, run_id)
        return 'allowed'
    except budgets.BudgetError as exc:
        return exc.code


def test_reservations_are_atomic_across_real_processes(ledger, request_payload):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs = [pool.submit(_process_reserve, ledger.path, request_payload, f'process-{i}')
                for i in range(2)]
        assert sorted(job.result(timeout=15) for job in jobs) == ['allowed', 'budget_exceeded']
    assert status(ledger)['protected_runs'] == 1


def test_held_reservation_survives_reopen_without_auto_reset(ledger, request_payload):
    reserve(ledger, request_payload)
    reloaded = StudioStore(ledger.path)
    with reloaded.connect() as db:
        budgets.initialize_schema(db)
    assert status(reloaded)['held_micro_usd'] == 26_340_000
    assert status(reloaded)['protected_runs'] == 1


def test_transaction_rollback_cannot_leak_reservations(ledger, request_payload):
    with pytest.raises(RuntimeError):
        with ledger.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            budgets.reserve_for_run(db, 'owner', 'rolled-back', request_payload)
            raise RuntimeError('Run start failed before dispatch')
    assert status(ledger)['held_micro_usd'] == 0
    with ledger.connect() as db:
        with pytest.raises(RuntimeError):
            budgets.reserve_for_run(db, 'owner', 'no-transaction', request_payload)


def test_unknown_reservation_can_finish_after_price_expiration(ledger, request_payload, monkeypatch):
    reserve(ledger, request_payload)
    monkeypatch.setattr(budgets, '_now', lambda: budgets.PRICING.expires_at + 1)
    assert finish(ledger, receipt(request_payload))['committed_upper_micro_usd'] == 4000
    assert status(ledger)['pricing']['status'] == 'budget_pricing_expired'


@pytest.mark.parametrize('changes', [
    {'model': 'gpt-6-luna'}, {'service_tier': 'priority'},
    {'long_input_micro_usd_per_million': 0}, {'long_output_micro_usd_per_million': -1},
    {'input_tokens_max': 10}, {'expires_at': None},
])
def test_malformed_pricing_is_fail_closed_without_breaking_status(ledger, request_payload, monkeypatch, changes):
    from dataclasses import replace
    monkeypatch.setattr(budgets, 'PRICING', replace(budgets.PRICING, **changes))
    with pytest.raises(budgets.BudgetError) as caught:
        reserve(ledger, request_payload)
    assert caught.value.code == 'budget_pricing_unavailable'
    assert status(ledger)['pricing']['status'] == 'budget_pricing_unavailable'


@pytest.mark.parametrize('enabled,limit', [(1, 1), ('true', 1), (True, True),
    (True, -1), (True, 1.5), (True, '30000000'), (True, 10**30)])
def test_settings_reject_coercion_and_unsafe_integer_ranges(ledger, enabled, limit):
    with ledger.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        with pytest.raises(budgets.BudgetError) as caught:
            budgets.set_budget(db, 'owner', enabled=enabled, limit_micro_usd=limit)
        assert caught.value.code == 'budget_invalid_settings'
    assert status(ledger)['limit_micro_usd'] == 30_000_000


def test_saved_limit_is_not_lost_when_disabled_before_first_protected_run(ledger):
    with ledger.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        budgets.set_budget(db, 'new-owner', enabled=False, limit_micro_usd=500_000)
    assert status(ledger, 'new-owner')['enabled'] is False
    assert status(ledger, 'new-owner')['limit_micro_usd'] == 500_000


def test_enabling_or_disabling_does_not_clear_a_provider_mismatch_halt(ledger, request_payload):
    reserve(ledger, request_payload)
    finish(ledger, receipt(request_payload, model='unexpected'))
    with ledger.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        budgets.set_budget(db, 'owner', enabled=False, limit_micro_usd=100_000_000)
        budgets.set_budget(db, 'owner', enabled=True, limit_micro_usd=100_000_000)
    assert status(ledger)['halted'] is True
    with pytest.raises(budgets.BudgetError) as caught:
        reserve(ledger, request_payload, 'second')
    assert caught.value.code == 'budget_halted'
