"""Real local consent and MCP controls; fictional owners and stub public collection."""
import json
import time

import pytest
from starlette.testclient import TestClient

from test_chatgpt_run_scopes import (make_app, issue_token, signed_browser, consent,
                                     rpc, call, ISSUER, BASE_SCOPE)
from dredge.chatgpt_controls import BUDGET_READ_SCOPE, BUDGET_SCOPE, RESEARCH_SCOPE
from dredge.chatgpt_foundation import tool_definitions
from dredge.chatgpt_oauth import verify_access_token
from dredge.chatgpt_transport import create_chatgpt_app
from dredge.public_source_collector import CATALOG, VERSION

OWNER = 'github:fictional-owner'
RUN = '00000000-0000-0000-0000-000000000001'
BUDGET = f'{BASE_SCOPE} {BUDGET_READ_SCOPE} {BUDGET_SCOPE}'
RESEARCH = f'{BASE_SCOPE} studio:runs:read {RESEARCH_SCOPE}'
CATALOG_ID = next(iter(CATALOG))


def enabled(make_app, monkeypatch):
    monkeypatch.setenv('DREDGE_CHATGPT_CONTROLS_ENABLED', 'true')
    app = make_app()
    app.config['STUDIO_ROLES'] = {OWNER: 'operator'}
    return app


def seed(app, owner=OWNER, data_scope='public_or_fictional'):
    with app.extensions['studio_store'].connect() as db:
        db.execute('INSERT INTO runs(id,owner,status,payload,result,created) VALUES(?,?,?,?,?,?)',
                   (RUN, owner, 'completed', json.dumps({'data_scope': data_scope, 'query': 'PRIVATE_QUERY',
                    'evidence': [{'id': 'source-1', 'title': 'PRIVATE_TITLE', 'url': CATALOG[CATALOG_ID][1],
                                  'excerpt': 'PRIVATE_EXCERPT'}]}),
                    json.dumps({'answer': 'PRIVATE_PARAGRAPH [source-1]'}), time.time()))


def result(client, token, name, args=None):
    response = call(client, token, name, args)
    assert response.status_code == 200, response.text
    return response.json()['result']


def test_flag_off_and_strict_enabled(make_app, monkeypatch):
    app = make_app()
    assert app.config['DREDGE_CHATGPT_CONTROLS_ENABLED'] is False
    assert 'update_my_studio_budget' not in {x['name'] for x in tool_definitions(app)}
    monkeypatch.setenv('DREDGE_CHATGPT_CONTROLS_ENABLED', 'yes')
    with pytest.raises(RuntimeError, match='CONTROLS_ENABLED'):
        make_app()


def test_discovery_old_grants_and_exact_annotations(make_app, monkeypatch):
    app = enabled(make_app, monkeypatch)
    token = issue_token(app)
    with TestClient(create_chatgpt_app(app), base_url=ISSUER) as client:
        definitions = {x['name']: x for x in rpc(client, token).json()['result']['tools']}
        assert len(definitions) == 7  # Original two plus controls, run reads still disabled.
        for name in ['update_my_studio_budget', 'check_my_studio_public_sources']:
            assert definitions[name]['annotations']['readOnlyHint'] is False
        assert definitions['check_my_studio_public_sources']['annotations']['openWorldHint'] is True
        assert definitions['update_my_studio_budget']['annotations']['destructiveHint'] is True
        assert definitions['get_my_studio_budget']['annotations']['readOnlyHint'] is True
        for name, args in [('get_my_studio_budget', {}),
                           ('update_my_studio_budget', {'currency':'USD','enabled':True,'limit_micro_usd':10}),
                           ('check_my_studio_public_sources', {'run_id':RUN,'catalog_ids':[CATALOG_ID]}),
                           ('get_my_studio_source_summary', {'run_id':RUN})]:
            response = result(client, token, name, args)
            assert response['isError'] is True
            assert 'insufficient_scope' in response['_meta']['mcp/www_authenticate'][0]
        assert result(client, token, 'list_approved_public_sources')['isError'] is False
    assert verify_access_token(app, token)['scope'] == BASE_SCOPE


@pytest.mark.parametrize('args', [
    {'currency':'USD','enabled':True,'limit_micro_usd':True},
    {'currency':'USD','enabled':True,'limit_micro_usd':1.5},
    {'currency':'EUR','enabled':True,'limit_micro_usd':10},
    {'currency':'USD','enabled':1,'limit_micro_usd':10},
    {'currency':'USD','enabled':True,'limit_micro_usd':1_000_000_000_001},
    {'currency':'USD','enabled':True,'limit_micro_usd':-1},
    {'currency':'USD','enabled':True,'limit_micro_usd':10,'owner':'google:other'},
])
def test_invalid_budget_never_prompts_consent(make_app, monkeypatch, args):
    app = enabled(make_app, monkeypatch)
    token = issue_token(app)
    with TestClient(create_chatgpt_app(app), base_url=ISSUER) as client:
        response = result(client, token, 'update_my_studio_budget', args)
        assert response['isError'] is True
        assert '_meta' not in response


def test_budget_scopes_role_owner_and_consent(make_app, monkeypatch):
    app = enabled(make_app, monkeypatch)
    text, _ = consent(signed_browser(app), BUDGET + ' ' + RESEARCH_SCOPE)
    assert 'enable or disable' in text.text and 'save source observations' in text.text
    assert 'held reservations' in text.text and 'Allow selected access' in text.text
    token = issue_token(app, BUDGET)
    other = issue_token(app, BUDGET, owner='google:other')
    only_write = issue_token(app, BASE_SCOPE + ' ' + BUDGET_SCOPE)
    args = {'currency':'USD','enabled':True,'limit_micro_usd':5_000_000}
    with TestClient(create_chatgpt_app(app), base_url=ISSUER) as client:
        assert result(client, only_write, 'update_my_studio_budget', args)['isError']
        assert result(client, other, 'update_my_studio_budget', args)['isError']
        updated = result(client, token, 'update_my_studio_budget', args)
        assert updated['structuredContent']['limit_micro_usd'] == 5_000_000
        assert result(client, other, 'get_my_studio_budget')['structuredContent']['limit_micro_usd'] == 0
        app.config['STUDIO_ROLES'][OWNER] = 'viewer'
        assert result(client, token, 'update_my_studio_budget', args)['isError']
        assert not result(client, token, 'get_my_studio_budget')['isError']


def test_source_check_projection_history_and_owner(make_app, monkeypatch):
    app = enabled(make_app, monkeypatch)
    seed(app)
    fetched = []
    def collect(key):
        fetched.append(key)
        return dict(url=CATALOG[key][1], title='PUBLIC_TITLE', observed_at=time.time(),
                    digest='a'*64, collector_version=VERSION, bytes_observed=15, text='PUBLIC_SOURCE_EXCERPT')
    monkeypatch.setattr('dredge.public_source_revalidation.collect', collect)
    token = issue_token(app, RESEARCH)
    other = issue_token(app, RESEARCH, owner='google:other')
    app.config['STUDIO_ROLES']['google:other'] = 'admin'
    args = {'run_id':RUN,'catalog_ids':[CATALOG_ID]}
    with TestClient(create_chatgpt_app(app), base_url=ISSUER) as client:
        assert result(client, other, 'check_my_studio_public_sources', args)['isError']
        assert not fetched
        checked = result(client, token, 'check_my_studio_public_sources', args)
        assert checked['isError'] is False, checked
        assert checked['structuredContent']['sources'][0]['state'] == 'first_observation'
        assert checked['structuredContent']['sections'] == [{'id':'section-1','action':'blocked'}]
        history = result(client, token, 'get_my_studio_source_summary', {'run_id':RUN})
        assert len(history['structuredContent']['checks']) == 1
        assert len(fetched) == 1
        assert result(client, other, 'get_my_studio_source_summary', {'run_id':RUN})['isError']
        for forbidden in ('PRIVATE_', 'PUBLIC_SOURCE_EXCERPT', 'PUBLIC_TITLE', 'excerpt', 'paragraph'):
            assert forbidden not in json.dumps([checked, history])


@pytest.mark.parametrize('scope', ['private', None, 'unknown'])
def test_private_run_never_fetched(make_app, monkeypatch, scope):
    app = enabled(make_app, monkeypatch)
    seed(app, data_scope=scope)
    monkeypatch.setattr('dredge.public_source_revalidation.collect', lambda key: pytest.fail('Private fetch'))
    token = issue_token(app, RESEARCH)
    with TestClient(create_chatgpt_app(app), base_url=ISSUER) as client:
        assert result(client, token, 'check_my_studio_public_sources', {'run_id':RUN,'catalog_ids':[CATALOG_ID]})['isError']
        assert result(client, token, 'get_my_studio_source_summary', {'run_id':RUN})['isError']


@pytest.mark.parametrize('ids', [['https://evil.test'], ['not-approved'], [CATALOG_ID,CATALOG_ID], [], [1]])
def test_invalid_catalog_never_prompts_or_fetches(make_app, monkeypatch, ids):
    app = enabled(make_app, monkeypatch)
    token = issue_token(app)
    with TestClient(create_chatgpt_app(app), base_url=ISSUER) as client:
        response = result(client, token, 'check_my_studio_public_sources', {'run_id':RUN,'catalog_ids':ids})
        assert response['isError'] and '_meta' not in response


def test_controls_disable_invalidates_expanded_token_not_base(make_app, monkeypatch):
    app = enabled(make_app, monkeypatch)
    base, expanded = issue_token(app), issue_token(app, BUDGET)
    app.config['DREDGE_CHATGPT_CONTROLS_ENABLED'] = False
    assert verify_access_token(app, base)
    assert verify_access_token(app, expanded) is None


def test_budget_stale_revoked_and_expired_tokens_cannot_mutate(make_app, monkeypatch):
    import hashlib
    app = enabled(make_app, monkeypatch)
    tokens = [issue_token(app, BUDGET) for _ in range(2)]
    with app.extensions['studio_store'].connect() as db:
        db.execute('UPDATE chatgpt_oauth_tokens SET revoked_at=? WHERE token_hash=?',
                   (time.time(), hashlib.sha256(tokens[0].encode()).hexdigest()))
        db.execute('UPDATE chatgpt_oauth_tokens SET expires_at=0 WHERE token_hash=?',
                   (hashlib.sha256(tokens[1].encode()).hexdigest(),))
    with TestClient(create_chatgpt_app(app), base_url=ISSUER) as client:
        for token in tokens:
            assert call(client, token, 'update_my_studio_budget',
                        {'currency':'USD','enabled':True,'limit_micro_usd':100}).status_code == 401
    with app.extensions['studio_store'].connect() as db:
        assert db.execute('SELECT COUNT(*) FROM studio_budget_settings').fetchone()[0] == 0


def test_projection_is_bounded_and_ignores_malformed_text_fields():
    from dredge.chatgpt_controls import _summary
    output = _summary({'check_id':'PRIVATE', 'run_id':'PRIVATE',
        'sources':[{'state': ['PRIVATE'], 'digest':'PRIVATE'*1000, 'source_id':'PRIVATE',
                    'observed_at':'PRIVATE', 'url':{}, 'excerpt':'PRIVATE'}]*100,
        'sections':[{'id':'PRIVATE', 'action':{'text':'PRIVATE'}, 'text':'PRIVATE'}]*1000})
    assert len(output['sources']) == 3 and len(output['sections']) == 100
    assert 'PRIVATE' not in json.dumps(output)


@pytest.mark.parametrize('flag, expected', [('true','review requested Studio permissions'),
                                           ('false','review the request for read-only Studio access')])
def test_tunnel_login_wording_without_new_routes(make_app, monkeypatch, flag, expected):
    from dredge.chatgpt_tunnel import create_tunnel_app
    monkeypatch.setenv('DREDGE_CHATGPT_CONTROLS_ENABLED', flag)
    make_app()
    monkeypatch.setenv('DREDGE_CHATGPT_TUNNEL_ENABLED', 'true')
    monkeypatch.setattr('dredge.chatgpt_tunnel._verify_socket', lambda: None)
    with TestClient(create_tunnel_app(), base_url=ISSUER, client=None) as client:
        headers = {'X-Forwarded-Proto':'https'}
        assert expected in client.get('/auth/login', headers=headers).text
        for path in ['/api/studio/budget', '/api/studio/research/revalidate']:
            assert client.post(path, headers=headers, json={}).status_code == 404
