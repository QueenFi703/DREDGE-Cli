"""Synthetic OAuth renewal tests; no real credentials or provider requests."""
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from test_chatgpt_oauth import make_app, sign_in, authorize, exchange, ISSUER, RESOURCE
from dredge.chatgpt_oauth import verify_access_token

def issue(app):
    client = app.test_client()
    sign_in(client)
    response = exchange(client, authorize(client))
    assert response.status_code == 200
    token = response.get_json()
    assert 'refresh_token' in token
    return token

def refresh(app, token, **changes):
    data = dict(grant_type='refresh_token', client_id='fictional-chatgpt',
                refresh_token=token, resource=RESOURCE)
    data.update(changes)
    return app.test_client().post('/chatgpt/oauth/token', data=data, base_url=ISSUER)

def test_expired_access_renews_without_browser_session(tmp_path):
    app = make_app(tmp_path/'db')
    old = issue(app)
    with app.extensions['studio_store'].connect() as db:
        db.execute('UPDATE chatgpt_oauth_tokens SET expires_at=?', (int(time.time())-1,))
    assert verify_access_token(app, old['access_token']) is None
    new = refresh(app, old['refresh_token']).get_json()
    assert new['refresh_token'] != old['refresh_token']
    assert new['expires_in'] == 3600
    assert verify_access_token(app, new['access_token'])['owner'] == 'github:fictional-owner'
    assert new['scope'] == old['scope']

def test_refresh_persists_across_app_restart_and_stores_only_digests(tmp_path):
    path = tmp_path/'db'; app = make_app(path); token = issue(app)
    restarted = make_app(path)
    assert refresh(restarted, token['refresh_token']).status_code == 200
    assert token['refresh_token'].encode() not in path.read_bytes()
    assert token['access_token'].encode() not in path.read_bytes()

def test_replay_revokes_descendant_tokens_but_not_other_connection(tmp_path):
    app = make_app(tmp_path/'db'); old = issue(app); other = issue(app)
    new = refresh(app, old['refresh_token']).get_json()
    assert refresh(app, old['refresh_token']).status_code == 400
    assert verify_access_token(app, new['access_token']) is None
    assert refresh(app, new['refresh_token']).status_code == 400
    assert verify_access_token(app, other['access_token']) is not None

def test_expiry_is_absolute_and_does_not_slide(tmp_path):
    app = make_app(tmp_path/'db'); old = issue(app)
    with app.extensions['studio_store'].connect() as db:
        initial = db.execute('SELECT expires_at FROM chatgpt_oauth_refresh_tokens').fetchone()[0]
    new = refresh(app, old['refresh_token']).get_json()
    with app.extensions['studio_store'].connect() as db:
        assert {r[0] for r in db.execute('SELECT expires_at FROM chatgpt_oauth_refresh_tokens')} == {initial}
        db.execute('UPDATE chatgpt_oauth_refresh_tokens SET expires_at=?', (int(time.time())-1,))
    assert refresh(app, new['refresh_token']).status_code == 400

def test_wrong_client_resource_and_scope_cannot_refresh(tmp_path):
    app = make_app(tmp_path/'db'); old = issue(app)
    for changes in ({'client_id':'other'}, {'resource':ISSUER+'/other'}, {'scope':'studio:read studio:runs:read'}):
        assert refresh(app, old['refresh_token'], **changes).status_code in {400,401}
    assert refresh(app, old['refresh_token']).status_code == 200

def test_refresh_revocation_invalidates_whole_connection(tmp_path):
    app = make_app(tmp_path/'db'); old = issue(app)
    new = refresh(app, old['refresh_token']).get_json()
    response = app.test_client().post('/chatgpt/oauth/revoke', base_url=ISSUER,
        data={'client_id':'fictional-chatgpt','token':new['refresh_token'],'token_type_hint':'refresh_token'})
    assert response.status_code == 200
    assert verify_access_token(app, new['access_token']) is None
    assert refresh(app, new['refresh_token']).status_code == 400

def test_access_revocation_prevents_refresh(tmp_path):
    app = make_app(tmp_path/'db'); old = issue(app)
    assert app.test_client().post('/chatgpt/oauth/revoke', base_url=ISSUER,
        data={'client_id':'fictional-chatgpt','token':old['access_token']}).status_code == 200
    assert refresh(app, old['refresh_token']).status_code == 400

def test_concurrent_refresh_has_one_winner_and_reuse_revokes_family(tmp_path):
    app = make_app(tmp_path/'db'); old = issue(app); barrier = Barrier(2)
    def run():
        barrier.wait(timeout=5)
        return refresh(app, old['refresh_token'])
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _:run(), range(2)))
    assert sorted(r.status_code for r in responses) == [200,400]
    winner = next(r.get_json() for r in responses if r.status_code == 200)
    assert verify_access_token(app, winner['access_token']) is None

def test_policy_change_rejects_refresh(tmp_path):
    app = make_app(tmp_path/'db'); old = issue(app)
    app.config['CHATGPT_CLIENT_ID'] = 'different'
    assert refresh(app, old['refresh_token'],client_id='different').status_code == 400

def test_code_replay_revokes_refresh_as_well_as_access(tmp_path):
    app = make_app(tmp_path/'db'); client = app.test_client(); sign_in(client)
    code = authorize(client); old = exchange(client, code).get_json()
    assert exchange(client, code).status_code == 400
    assert refresh(app, old['refresh_token']).status_code == 400

def test_scope_narrowing_cannot_be_reexpanded(tmp_path):
    app = make_app(tmp_path/'db'); app.config['DREDGE_CHATGPT_RUN_TOOLS_ENABLED'] = True
    client = app.test_client(); sign_in(client)
    old = exchange(client, authorize(client,scope='studio:read studio:runs:read')).get_json()
    narrow = refresh(app, old['refresh_token'],scope='studio:read').get_json()
    assert narrow['scope'] == 'studio:read'
    assert refresh(app,narrow['refresh_token'],scope='studio:read studio:runs:read').status_code == 400
    assert refresh(app,narrow['refresh_token']).status_code == 200

def test_current_scope_policy_is_checked_on_renewal(tmp_path):
    app = make_app(tmp_path/'db'); app.config['DREDGE_CHATGPT_RUN_TOOLS_ENABLED'] = True
    client = app.test_client(); sign_in(client)
    old = exchange(client, authorize(client,scope='studio:read studio:runs:read')).get_json()
    app.config['DREDGE_CHATGPT_RUN_TOOLS_ENABLED'] = False
    assert refresh(app,old['refresh_token']).status_code == 400

def test_refresh_rejects_identity_override_and_wrong_host(tmp_path):
    app = make_app(tmp_path/'db'); old = issue(app)
    assert refresh(app,old['refresh_token'],owner='github:other').status_code == 400
    data=dict(grant_type='refresh_token',client_id='fictional-chatgpt',
              refresh_token=old['refresh_token'],resource=RESOURCE)
    assert app.test_client().post('/chatgpt/oauth/token',data=data,base_url='https://other.test').status_code == 400
    assert refresh(app,old['refresh_token']).status_code == 200

def test_upgrade_preserves_existing_access_tokens(tmp_path):
    path = tmp_path/'db'; app = make_app(path); old = issue(app)
    with app.extensions['studio_store'].connect() as db:
        db.execute('DROP TABLE chatgpt_oauth_refresh_tokens')
    upgraded = make_app(path)
    assert verify_access_token(upgraded,old['access_token']) is not None
    assert refresh(upgraded,old['refresh_token']).status_code == 400
    assert issue(upgraded)['refresh_token']

def test_refresh_debug_logs_do_not_expose_credentials(tmp_path,caplog):
    import logging
    app = make_app(tmp_path/'db')
    with caplog.at_level(logging.DEBUG,logger='authlib'):
        old = issue(app)
        new = refresh(app,old['refresh_token']).get_json()
    for token in (old,new):
        for key in ('access_token','refresh_token'):
            assert token[key] not in caplog.text
