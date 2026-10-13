"""Saved research reuse never executes research or certifies source freshness."""
import json

import pytest

from test_chatgpt_runs import app, call, seed, OWNER, OTHER, run_id

TOOL = 'find_my_saved_research'
NOW = 1_000_000


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    monkeypatch.setattr('dredge.chatgpt_research.time.time', lambda: NOW)


def research(app, number=1, age=3600, **kwargs):
    ident = seed(app, number, created=NOW-age-2, **kwargs)
    with app.extensions['studio_store'].connect() as db:
        db.execute('UPDATE runs SET started=?,ended=? WHERE id=?',
                   (NOW-age-1, NOW-age, ident))
    return ident


def test_reuse_is_bounded_owner_scoped_and_not_freshness_proof(app):
    ident = research(app)
    research(app, 2, owner=OTHER)
    research(app, 3, status='failed')
    research(app, 4, payload={'data_scope': 'private'})
    research(app, 5, age=8*86400)
    result = call(app, TOOL, {'query': 'PUBLIC'})
    assert [c['id'] for c in result['candidates']] == [ident]
    candidate = result['candidates'][0]
    assert candidate['age_seconds'] == 3600
    assert candidate['freshness_status'] == 'sources_not_revalidated'
    assert candidate['summary'] == 'A fictional public answer.'
    assert result['new_research_executed'] is False
    assert result['savings_status'] == 'not_measured'
    assert 'PRIVATE' not in json.dumps(result)


def test_only_exposed_summary_is_searched_not_prompt_or_hidden_tail(app):
    research(app, result={'answer': 'a'*2000 + ' SECRETTAIL'})
    assert not call(app, TOOL, {'query': 'SECRETTAIL'})['candidates']
    assert not call(app, TOOL, {'query': 'PRIVATE PROMPT'})['candidates']


@pytest.mark.parametrize('ended', [None, NOW+1, 1, float('inf')])
def test_unknown_future_or_inconsistent_completion_dates_are_excluded(app, ended):
    ident = research(app)
    with app.extensions['studio_store'].connect() as db:
        db.execute('UPDATE runs SET ended=? WHERE id=?', (ended, ident))
    assert not call(app, TOOL, {'query': 'public'})['candidates']


def test_age_window_boundary_and_demo_labels(app):
    research(app, age=86400, payload={'data_scope': 'public_or_fictional', 'execution_mode': 'demo'})
    result = call(app, TOOL, {'query': 'public', 'max_age_days': 1})
    assert result['candidates'][0]['mode'] == 'demo'
    assert result['candidates'][0]['age_seconds'] == 86400


@pytest.mark.parametrize('arguments', [None, [], {}, {'query': ''}, {'query': '  '},
    {'query': 'a'*201}, {'query': '\ud800'}, {'query': 'public', 'owner': OTHER},
    {'query': 'public', 'max_age_days': True}, {'query': 'public', 'max_age_days': 0},
    {'query': 'public', 'max_age_days': 366}, {'query': 'public', 'offset': -1},
    {'query': 'public', 'limit': 21}])
def test_invalid_arguments(app, arguments):
    with pytest.raises(ValueError):
        call(app, TOOL, arguments)


def test_scope_and_enable_flag_fail_closed(app):
    with pytest.raises(PermissionError):
        call(app, TOOL, {'query': 'public'}, scopes=frozenset({'studio:read'}))
    with pytest.raises(PermissionError):
        call(app, TOOL, {'query': 'public'}, owner=None)
    app.config['DREDGE_CHATGPT_RUN_TOOLS_ENABLED'] = False
    with pytest.raises(PermissionError):
        call(app, TOOL, {'query': 'public'})


def test_pagination_is_over_scanned_runs_not_just_matches(app):
    for number in range(1, 4):
        research(app, number, age=number*3600)
    first = call(app, TOOL, {'query': 'no match', 'limit': 2})
    assert first['candidates'] == []
    assert first['has_more'] is True
    assert first['next_offset'] == 2
    second = call(app, TOOL, {'query': 'public', 'limit': 2, 'offset': 2})
    assert [c['id'] for c in second['candidates']] == [run_id(3)]
    assert second['has_more'] is False
