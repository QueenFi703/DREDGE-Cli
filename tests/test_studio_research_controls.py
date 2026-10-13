"""Supplied-text comparison only; no remote fetch, execution or dollar allowance."""
import json
import pytest
from test_studio import app, client_for, proposal


def saved_answer(app, c, h, evidence=None):
    options = {'evidence': evidence} if evidence is not None else {}
    run_id = proposal(c, h, **options).json['id']
    with app.extensions['studio_store'].connect() as db:
        db.execute('UPDATE runs SET status=?,result=? WHERE id=?',
                   ('completed', json.dumps({'answer': 'First paragraph [source-1].\n\nUncited claim.'}), run_id))
    return run_id


def test_persisted_source_versions_and_changed_section_plan(app):
    c, h = client_for(app)
    old = saved_answer(app, c, h)
    new = proposal(c, h).json['id']
    with app.extensions['studio_store'].connect() as db:
        assert db.execute('SELECT COUNT(*) FROM studio_source_versions').fetchone()[0] == 2
    body = {'base_run_id': old, 'new_run_id': new}
    assert c.post('/api/studio/research/refresh-plan', json=body).status_code == 403
    result = c.post('/api/studio/research/refresh-plan', json=body, headers=h)
    assert result.status_code == 200
    assert result.json['sections'][0]['action'] == 'unchanged_input_review'
    assert result.json['sections'][1]['action'] == 'blocked'
    assert result.json['automatic_reuse_allowed'] is False
    assert result.json['web_freshness'] == 'not_verified'
    changed = proposal(c, h, evidence=[{'title': 'Public reference',
        'url': 'https://docs.python.org/3/library/sqlite3.html', 'excerpt': 'Changed record'}]).json['id']
    result = c.post('/api/studio/research/refresh-plan', headers=h,
                    json={'base_run_id': old, 'new_run_id': changed})
    assert result.json['sections'][0]['action'] == 'changed_input_review'


def test_changed_query_source_set_and_cross_owner_block(app):
    c, h = client_for(app)
    old = saved_answer(app, c, h)
    for new in [proposal(c, h, query='A different question').json['id'],
                proposal(c, h, evidence=[]).json['id']]:
        result = c.post('/api/studio/research/refresh-plan', headers=h,
                        json={'base_run_id': old, 'new_run_id': new})
        assert all(section['action'] == 'blocked' for section in result.json['sections'])
    admin, ah = client_for(app, 'test:admin')
    other = proposal(admin, ah).json['id']
    assert c.post('/api/studio/research/refresh-plan', headers=h,
                  json={'base_run_id': old, 'new_run_id': other}).status_code == 404
    assert admin.post('/api/studio/research/refresh-plan', headers=ah,
                      json={'base_run_id': old, 'new_run_id': other}).status_code == 404


def test_control_page_has_truthful_scope(app):
    c, h = client_for(app)
    response = c.get('/studio/research-controls')
    assert response.status_code == 200
    assert b'Compare saved supplied sources' in response.data
    assert b'does not show a line-by-line diff' in response.data
    assert response.headers['Cache-Control'] == 'no-store'


def test_context_version_and_legacy_context_unknown_block(app):
    c, h = client_for(app)
    old = saved_answer(app, c, h)
    new = proposal(c, h).json['id']
    body = {'base_run_id': old, 'new_run_id': new}
    with app.extensions['studio_store'].connect() as db:
        db.execute('UPDATE studio_source_versions SET version=? WHERE run_id=?', ('other', new))
    result = c.post('/api/studio/research/refresh-plan', headers=h, json=body)
    assert result.json['source_states']['source-1'] == 'unknown_version'
    assert all(section['action'] == 'blocked' for section in result.json['sections'])
    with app.extensions['studio_store'].connect() as db:
        db.execute('DELETE FROM studio_source_context WHERE run_id=?', (old,))
    result = c.post('/api/studio/research/refresh-plan', headers=h, json=body)
    assert result.json['proposal_context_matches'] is False


@pytest.mark.parametrize('payload', ['{}', '[]', 'not json', '{"evidence":[{}]}'])
def test_malformed_historical_sources_fail_closed(app, payload):
    c, h = client_for(app)
    old, new = saved_answer(app, c, h), proposal(c, h).json['id']
    with app.extensions['studio_store'].connect() as db:
        db.execute('UPDATE runs SET payload=? WHERE id=?', (payload, old))
    response = c.post('/api/studio/research/refresh-plan', headers=h,
                      json={'base_run_id':old,'new_run_id':new})
    assert response.status_code == 409


@pytest.mark.parametrize('answer', [None, {}, 'x'*64001, '\n\n'.join(['p']*101)])
def test_unbounded_or_malformed_answers_fail_closed(app, answer):
    c, h = client_for(app)
    old, new = saved_answer(app, c, h), proposal(c, h).json['id']
    with app.extensions['studio_store'].connect() as db:
        db.execute('UPDATE runs SET result=? WHERE id=?', (json.dumps({'answer':answer}), old))
    assert c.post('/api/studio/research/refresh-plan', headers=h,
                  json={'base_run_id':old,'new_run_id':new}).status_code == 409


def test_matching_proposals_never_claim_execution_context_or_remote_freshness(app):
    c, h = client_for(app)
    old, new = saved_answer(app, c, h), proposal(c, h).json['id']
    result = c.post('/api/studio/research/refresh-plan', headers=h,
                    json={'base_run_id':old,'new_run_id':new}).json
    assert result['proposal_context_matches'] is True
    assert result['execution_context_verified'] is False
    assert result['automatic_reuse_allowed'] is False
    assert result['web_freshness'] == 'not_verified'
    assert 'plan_compatible' not in result
    assert result['provider_calls'] == 0


def test_no_cap_or_cost_preflight_surface_is_shipped(app):
    c, h = client_for(app)
    assert c.get('/api/studio/research/cap').status_code == 404
    assert c.post('/api/studio/research/cap', headers=h, json={'enabled':True,'limit_micro_usd':1}).status_code == 404
    assert c.get('/api/studio/runs/fixture/preflight').status_code == 404
    with app.extensions['studio_store'].connect() as db:
        assert not db.execute("SELECT 1 FROM sqlite_master WHERE name='studio_research_caps'").fetchone()


def test_page_and_comparison_require_signin(app):
    assert app.test_client().get('/studio/research-controls').status_code == 302
    assert app.test_client().post('/api/studio/research/refresh-plan',json={}).status_code == 401


def test_newly_captured_hashes_are_not_rewritten_by_comparison(app):
    c, h = client_for(app)
    old, new = saved_answer(app, c, h), proposal(c, h).json['id']
    with app.extensions['studio_store'].connect() as db:
        before = [tuple(r) for r in db.execute('SELECT * FROM studio_source_versions ORDER BY run_id')]
    assert c.post('/api/studio/research/refresh-plan',headers=h,
                  json={'base_run_id':old,'new_run_id':new}).status_code == 200
    with app.extensions['studio_store'].connect() as db:
        assert before == [tuple(r) for r in db.execute('SELECT * FROM studio_source_versions ORDER BY run_id')]


def test_valid_unicode_sources_survive_stored_json_escape_expansion(app):
    c, h = client_for(app)
    payload = {'execution_mode':'demo','query':'Compare these fictional inputs.',
               'evidence':[{'title':'Public source','url':'https://example.test/source',
                            'excerpt':'😀'*3800} for _ in range(3)]}
    encoded = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    assert len(encoded) < 65536
    ids = []
    for _ in range(2):
        response = c.post('/api/studio/runs', headers=h, data=encoded, content_type='application/json')
        assert response.status_code == 201
        ids.append(response.json['id'])
    with app.extensions['studio_store'].connect() as db:
        assert len(db.execute('SELECT payload FROM runs WHERE id=?', (ids[0],)).fetchone()[0]) > 128000
        db.execute('UPDATE runs SET status=?,result=? WHERE id=?',
                   ('completed', json.dumps({'answer':'Review [source-1].'}), ids[0]))
    response = c.post('/api/studio/research/refresh-plan', headers=h,
                      json={'base_run_id':ids[0], 'new_run_id':ids[1]})
    assert response.status_code == 200
    assert response.json['source_states']['source-1'] == 'unchanged_supplied_record'
