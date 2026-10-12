"""Owner-bound recorded topology through the real OAuth/MCP read interface."""
import json

import pytest
from starlette.testclient import TestClient

from test_chatgpt_runs import app, call, seed, run_id, OWNER, OTHER
from test_chatgpt_run_scopes import make_app, issue_token, rpc, EXPANDED_SCOPE, ISSUER
from dredge.chatgpt_transport import create_chatgpt_app

TOOL = 'get_my_studio_execution_graph'


def trace(app, ident, event):
    app.extensions['studio_store'].trace(ident, event)


def test_recorded_topology_latest_state_and_no_raw_content(app):
    ident = seed(app)
    trace(app, ident, {'id': 'PRIVATE SOURCE', 'type': 'execute',
                      'status': 'running', 'mode': 'live', 'dependencies': [],
                      'answer': 'PRIVATE ANSWER', 'request_id': 'PRIVATE REQUEST'})
    trace(app, ident, {'id': 'PRIVATE SOURCE', 'type': 'execute',
                      'status': 'completed', 'mode': 'live', 'dependencies': [],
                      'duration_ms': 123.5})
    trace(app, ident, {'id': 'PRIVATE TARGET', 'type': 'normalize',
                      'status': 'completed', 'mode': 'local',
                      'dependencies': ['PRIVATE SOURCE'], 'duration_ms': 2})
    output = call(app, TOOL, {'run_id': ident})
    assert output['graph_status'] == 'recorded'
    assert output['nodes'][0]['status'] == 'completed'
    assert output['nodes'][0]['duration_ms'] == 123.5
    assert output['edges'] == [{'source': 'node_1', 'target': 'node_2'}]
    assert output['truncated'] is False
    assert 'PRIVATE' not in json.dumps(output)
    assert 'preserved' not in output and 'retired' not in output


def test_graph_owner_only_even_for_admin_and_no_trace_is_unknown(app):
    own = seed(app)
    foreign = seed(app, 2, owner=OTHER)
    app.config['STUDIO_ROLES'] = {OWNER: 'admin'}
    for ident in (foreign, run_id(999)):
        with pytest.raises(ValueError, match='Run not found'):
            call(app, TOOL, {'run_id': ident})
    output = call(app, TOOL, {'run_id': own})
    assert output['graph_status'] == 'unavailable'
    assert output['nodes'] == output['edges'] == []


def test_graph_bounded_and_malformed_events_do_not_leak(app):
    ident = seed(app)
    for n in range(502):
        trace(app, ident, {'id': 'node', 'type': 'PRIVATE TYPE', 'status': 'PRIVATE',
                          'mode': 'PRIVATE', 'dependencies': [], 'duration_ms': -1})
    trace(app, ident, {'id': ['PRIVATE'], 'answer': 'PRIVATE'})
    output = call(app, TOOL, {'run_id': ident})
    assert output['truncated'] is True
    assert output['nodes'] == [{'id': 'node_1', 'type': 'unknown', 'status': 'unknown',
                                'mode': 'unknown', 'duration_ms': None}]
    assert 'PRIVATE' not in json.dumps(output)


@pytest.mark.parametrize('arguments', [{}, [], {'run_id': '../private'},
                                      {'run_id': run_id(1), 'owner': OTHER}])
def test_graph_rejects_invalid_arguments(app, arguments):
    with pytest.raises(ValueError):
        call(app, TOOL, arguments)


def test_node_limit_and_missing_dependencies_are_explicit(app):
    ident = seed(app)
    for n in range(101):
        trace(app, ident, {'id': str(n), 'type': 'execute', 'status': 'completed',
                          'mode': 'local', 'dependencies': ['not-recorded'],
                          'duration_ms': float('nan')})
    # Nonfinite event JSON fails closed, rather than disclosing any raw text.
    assert call(app, TOOL, {'run_id': ident})['graph_status'] == 'unavailable'
    for n in range(101):
        trace(app, ident, {'id': str(n), 'type': 'execute', 'status': 'completed',
                          'mode': 'local', 'dependencies': ['not-recorded']})
    output = call(app, TOOL, {'run_id': ident})
    assert len(output['nodes']) == 100
    assert output['edges'] == [] and output['truncated'] is True


def test_real_mcp_graph_requires_expanded_grant_and_is_readonly(make_app):
    app = make_app('true')
    ident = seed(app, owner='github:fictional-owner')
    trace(app, ident, {'id': 'source', 'type': 'execute', 'status': 'completed',
                      'mode': 'live', 'dependencies': []})
    base = issue_token(app)
    expanded = issue_token(app, EXPANDED_SCOPE)
    with app.extensions['studio_store'].connect() as db:
        before = {table: db.execute('SELECT COUNT(*) FROM ' + table).fetchone()[0]
                  for table in ('runs', 'trace_events', 'audit')}
    with TestClient(create_chatgpt_app(app), base_url=ISSUER) as client:
        assert TOOL not in {t['name'] for t in rpc(client, base).json()['result']['tools']}
        params = {'name': TOOL, 'arguments': {'run_id': ident}}
        assert rpc(client, base, 'tools/call', params).json()['result']['isError'] is True
        result = rpc(client, expanded, 'tools/call', params).json()['result']
        assert result['isError'] is False
        assert result['structuredContent']['graph_status'] == 'recorded'
        params['arguments']['owner'] = OTHER
        assert rpc(client, expanded, 'tools/call', params).json()['result']['isError'] is True
    with app.extensions['studio_store'].connect() as db:
        after = {table: db.execute('SELECT COUNT(*) FROM ' + table).fetchone()[0]
                 for table in before}
    assert before == after
    app.config['DREDGE_CHATGPT_RUN_TOOLS_ENABLED'] = False
    with pytest.raises(PermissionError):
        call(app, TOOL, {'run_id': ident})
