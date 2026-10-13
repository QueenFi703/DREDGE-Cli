"""Real Studio budget routes/transactions with a fake Responses HTTP boundary."""
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest
import requests

from dredge import studio_budgets as budgets
from dredge import studio_provider
from test_studio import app, approve, client_for
from test_studio_live import live, live_proposal, execute


@pytest.fixture(autouse=True)
def current_budget_prices(monkeypatch):
    monkeypatch.setattr(budgets, '_now', lambda: 1791856800.0)


def configure(client, headers, limit, enabled=True):
    response = client.post('/api/studio/budget', json={
        'enabled': enabled, 'limit_micro_usd': limit}, headers=headers)
    assert response.status_code == 200
    return response.json


def counts(app):
    with app.extensions['studio_store'].connect() as db:
        return {
            'reservations': db.execute('SELECT COUNT(*) FROM studio_budget_reservations').fetchone()[0],
            'dedup': db.execute('SELECT COUNT(*) FROM studio_live_dedup').fetchone()[0],
        }


def test_budgeted_generation_dispatches_frozen_request_and_releases_verified_excess(app, live):
    calls, state, _ = live
    state['data']['service_tier'] = 'default'
    client, headers = client_for(app)
    assert configure(client, headers, 30_000_000)['enabled'] is True
    run = live_proposal(client, headers).json
    result = execute(app, client, headers, run['id'])
    assert result.status_code == 200
    assert len(calls) == 1
    sent = calls[0][1]['json']
    assert sent['truncation'] == 'disabled'
    assert set(sent) == budgets.REQUEST_FIELDS
    assert result.json['result']['request_hash'] == budgets.request_hash(sent)
    assert result.json['result']['service_tier'] == 'default'
    with app.extensions['studio_store'].connect() as db:
        row = db.execute('SELECT * FROM studio_budget_reservations WHERE run_id=?', (run['id'],)).fetchone()
    assert row['request_hash'] == budgets.request_hash(sent)
    assert row['reserve_micro_usd'] == 26_340_000
    status = client.get('/api/studio/budget').json
    assert status['held_micro_usd'] == 0
    assert status['committed_upper_micro_usd'] == 4000
    assert status['available_micro_usd'] == 29_996_000
    assert status['actual_cost_usd'] is None


def test_one_dollar_blocks_without_provider_or_dedup_and_can_retry_after_increase(app, live):
    client, headers = client_for(app)
    configure(client, headers, 1_000_000)
    run = live_proposal(client, headers).json
    assert approve(app, run['id']).status_code == 200
    result = client.post(f"/api/studio/runs/{run['id']}/execute", json={}, headers=headers)
    assert result.status_code == 429
    assert result.json['code'] == 'budget_exceeded'
    assert counts(app) == {'reservations': 0, 'dedup': 0}
    assert not live[0]
    assert client.get(f"/api/studio/runs/{run['id']}").json['status'] == 'approved'
    configure(client, headers, 30_000_000)
    assert client.post(f"/api/studio/runs/{run['id']}/execute", json={}, headers=headers).status_code == 200
    assert len(live[0]) == 1


def test_timeout_holds_budget_and_blocks_a_different_new_request(app, live):
    live[1]['exception'] = requests.Timeout('fictional timeout')
    client, headers = client_for(app)
    configure(client, headers, 30_000_000)
    first = live_proposal(client, headers).json
    assert execute(app, client, headers, first['id']).status_code == 500
    assert client.get('/api/studio/budget').json['held_micro_usd'] == 26_340_000
    second = live_proposal(client, headers, query='A different fictional question.').json
    result = execute(app, client, headers, second['id'])
    assert result.status_code == 429 and result.json['code'] == 'budget_exceeded'
    assert len(live[0]) == 1
    assert counts(app) == {'reservations': 1, 'dedup': 1}


def test_success_without_observed_service_tier_retains_full_reservation(app, live):
    client, headers = client_for(app)
    configure(client, headers, 30_000_000)
    run = live_proposal(client, headers).json
    assert execute(app, client, headers, run['id']).status_code == 200
    status = client.get('/api/studio/budget').json
    assert status['held_micro_usd'] == 26_340_000
    assert status['committed_upper_micro_usd'] == 0
    assert status['halted'] is False


def test_duplicate_rejection_does_not_leave_a_phantom_budget_hold(app, live):
    live[1]['data']['service_tier'] = 'default'
    client, headers = client_for(app)
    configure(client, headers, 30_000_000)
    first = live_proposal(client, headers).json
    assert execute(app, client, headers, first['id']).status_code == 200
    second = live_proposal(client, headers).json
    result = execute(app, client, headers, second['id'])
    assert result.status_code == 409 and result.json['code'] == 'duplicate_live_run'
    assert len(live[0]) == 1
    status = client.get('/api/studio/budget').json
    assert status['held_micro_usd'] == 0
    assert status['committed_upper_micro_usd'] == 4000
    assert counts(app) == {'reservations': 1, 'dedup': 1}


def test_budget_and_unknown_holds_are_independent_for_each_owner(app, live):
    live[1]['exception'] = requests.Timeout('fictional timeout')
    for owner in ('test:op', 'test:other'):
        client, headers = client_for(app, owner)
        configure(client, headers, 30_000_000)
        run = live_proposal(client, headers).json
        assert execute(app, client, headers, run['id']).status_code == 500
        assert client.get('/api/studio/budget').json['held_micro_usd'] == 26_340_000
    assert len(live[0]) == 2
    assert counts(app) == {'reservations': 2, 'dedup': 2}


def test_enabling_cap_during_uncapped_inflight_call_does_not_backfill_ledger(app, live, monkeypatch):
    live[1]['data']['service_tier'] = 'default'
    client, headers = client_for(app)
    configure(client, headers, 1_000_000, enabled=False)
    run = live_proposal(client, headers).json
    assert approve(app, run['id']).status_code == 200
    entered, release = threading.Event(), threading.Event()
    original_post = studio_provider.requests.post

    def blocked_provider(*args, **kwargs):
        entered.set()
        assert release.wait(10), 'test must release the fake provider'
        return original_post(*args, **kwargs)

    monkeypatch.setattr(studio_provider.requests, 'post', blocked_provider)
    settings_client, settings_headers = client_for(app)
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(client.post, f"/api/studio/runs/{run['id']}/execute", json={}, headers=headers)
        try:
            assert entered.wait(10), 'uncapped request should reach the fake provider'
            configure(settings_client, settings_headers, 1_000_000, enabled=True)
        finally:
            release.set()
        assert result.result(timeout=10).status_code == 200
    status = settings_client.get('/api/studio/budget').json
    assert status['protected_runs'] == 0 and status['used_upper_micro_usd'] == 0
    assert status['enabled'] is True
    assert 'uncapped calls' in status['scope']
    next_run = live_proposal(settings_client, settings_headers, query='A new protected question.').json
    rejected = execute(app, settings_client, settings_headers, next_run['id'])
    assert rejected.status_code == 429 and rejected.json['code'] == 'budget_exceeded'
    assert len(live[0]) == 1


def test_two_inflight_protected_calls_cannot_overspend_shared_owner_limit(app, live, monkeypatch):
    live[1]['data']['service_tier'] = 'default'
    first_client, first_headers = client_for(app)
    second_client, second_headers = client_for(app)
    configure(first_client, first_headers, 30_000_000)
    first = live_proposal(first_client, first_headers).json
    second = live_proposal(second_client, second_headers, query='Another concurrent question.').json
    assert approve(app, first['id']).status_code == 200
    assert approve(app, second['id']).status_code == 200
    entered, release = threading.Event(), threading.Event()
    original_post = studio_provider.requests.post

    def blocked_provider(*args, **kwargs):
        entered.set()
        assert release.wait(10), 'test must release the fake provider'
        return original_post(*args, **kwargs)

    monkeypatch.setattr(studio_provider.requests, 'post', blocked_provider)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first_result = pool.submit(first_client.post, f"/api/studio/runs/{first['id']}/execute",
                                   json={}, headers=first_headers)
        try:
            assert entered.wait(10)
            rejected = second_client.post(f"/api/studio/runs/{second['id']}/execute",
                                          json={}, headers=second_headers)
            assert rejected.status_code == 429 and rejected.json['code'] == 'budget_exceeded'
            assert counts(app) == {'reservations': 1, 'dedup': 1}
        finally:
            release.set()
        assert first_result.result(timeout=10).status_code == 200
    assert len(live[0]) == 1


@pytest.mark.parametrize('provider_change', [
    {'usage': {'input_tokens': 100, 'output_tokens': 20, 'total_tokens': 120, 'output_tokens_details': {'reasoning_tokens': -1}}},
    {'usage': {'input_tokens': -1, 'output_tokens': 20, 'total_tokens': 19}},
    {'usage': {'input_tokens': 100, 'output_tokens': 20, 'total_tokens': 120, 'input_tokens_details': 'malformed'}},
    {'model': 'unexpected-model'}, {'service_tier': 'priority'},
    {'usage': {'input_tokens': 10, 'output_tokens': 1201, 'total_tokens': 1211}},
])
def test_provider_mismatch_halts_later_protected_generation(app, live, provider_change):
    live[1]['data']['service_tier'] = 'default'
    live[1]['data'].update(provider_change)
    client, headers = client_for(app)
    configure(client, headers, 100_000_000)
    first = live_proposal(client, headers).json
    execute(app, client, headers, first['id'])
    status = client.get('/api/studio/budget').json
    assert status['halted'] is True and status['held_micro_usd'] == 26_340_000
    second = live_proposal(client, headers, query='Should be halted.').json
    rejected = execute(app, client, headers, second['id'])
    assert rejected.status_code == 409 and rejected.json['code'] == 'budget_halted'
    assert len(live[0]) == 1
