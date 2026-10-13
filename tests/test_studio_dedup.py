"""Exact-input duplicate prevention through approved Studio HTTP execution."""
from concurrent.futures import ThreadPoolExecutor

import requests

from test_studio import app, client_for, approve
from test_studio_live import live, live_proposal, execute


def test_separate_identical_proposals_make_one_provider_call(app, live):
    calls, _, _ = live
    client, headers = client_for(app)
    first = live_proposal(client, headers).json['id']
    second = live_proposal(client, headers).json['id']
    assert execute(app, client, headers, first).status_code == 200
    result = execute(app, client, headers, second)
    assert result.status_code == 409
    assert result.json['code'] == 'duplicate_live_run'
    assert result.json['existing_run_id'] == first
    assert len(calls) == 1
    report = client.get('/api/studio/report').json
    assert report['duplicate_requests_blocked'] == 1
    assert report['savings_status'] == 'not_measured'
    assert report['token_usage'] == 120


def test_parallel_distinct_run_ids_are_atomic(app, live):
    calls, _, _ = live
    clients = [client_for(app) for _ in range(2)]
    ids = [live_proposal(c, h).json['id'] for c, h in clients]
    for ident in ids:
        assert approve(app, ident).status_code == 200
    def invoke(item):
        (c, h), ident = item
        return c.post(f'/api/studio/runs/{ident}/execute', json={}, headers=h).status_code
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(invoke, zip(clients, ids))) == [200, 409]
    assert len(calls) == 1


def test_timeout_retains_reservation_and_never_retries(app, live):
    calls, state, _ = live
    state['exception'] = requests.Timeout()
    c, h = client_for(app)
    one = live_proposal(c, h).json['id']
    assert execute(app, c, h, one).status_code == 500
    two = live_proposal(c, h).json['id']
    assert execute(app, c, h, two).status_code == 409
    assert len(calls) == 1


def test_changed_evidence_and_other_owner_are_not_duplicates(app, live):
    calls, _, _ = live
    c, h = client_for(app)
    assert execute(app, c, h, live_proposal(c, h).json['id']).status_code == 200
    changed = live_proposal(c, h, evidence=[{'title': 'Other source', 'url': 'https://example.org',
                                           'excerpt': 'A changed fictional fact.'}]).json['id']
    assert execute(app, c, h, changed).status_code == 200
    other, oh = client_for(app, 'test:other')
    assert execute(app, other, oh, live_proposal(other, oh).json['id']).status_code == 200
    assert len(calls) == 3
    assert other.get('/api/studio/report').json['duplicate_requests_blocked'] == 0


def test_expired_reservation_allows_reviewed_rerun(app, live):
    calls, _, _ = live
    c, h = client_for(app)
    assert execute(app, c, h, live_proposal(c, h).json['id']).status_code == 200
    with app.extensions['studio_store'].connect() as db:
        db.execute('UPDATE studio_live_dedup SET reserved_at=reserved_at-86401')
    assert execute(app, c, h, live_proposal(c, h).json['id']).status_code == 200
    assert len(calls) == 2


def test_fingerprint_preserves_context_model_options_and_source_versions():
    from copy import deepcopy
    from dredge.studio_dedup import execution_fingerprint
    base = {'query': 'Fictional question', 'model': 'fictional-model', 'max_output_tokens': 1200,
            'data_scope': 'public_or_fictional', 'consent': True, 'execution_mode': 'live',
            'evidence': [{'id': 'source-1', 'title': 'Title', 'url': 'https://example.org',
                          'excerpt': 'Recorded content', 'attached_at': 10, 'version': 'v1'}]}
    original = execution_fingerprint(base)
    for name, value in [('model', 'different-model'), ('max_output_tokens', 800),
                        ('data_scope', 'private'), ('case_id', 'fictional-case'),
                        ('agency_id', 'fictional-agency')]:
        changed = deepcopy(base)
        changed[name] = value
        assert execution_fingerprint(changed) != original
    changed = deepcopy(base)
    changed['evidence'][0]['version'] = 'v2'
    assert execution_fingerprint(changed) != original
    changed = deepcopy(base)
    changed['evidence'][0]['attached_at'] = 20
    assert execution_fingerprint(changed) == original
    assert base['evidence'][0]['attached_at'] == 10
