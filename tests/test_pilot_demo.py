"""The narrative preview must not touch authentication, records or providers."""
from test_studio import app


def test_pilot_is_public_read_only_and_has_no_authenticated_scripts(app):
    client = app.test_client()
    response = client.get('/pilot-demo')
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'studio_pilot.js' in html
    assert 'studio_casework.js' not in html and 'src="/static/studio.js' not in html
    assert '<form' not in html and 'Fictional training demonstration' in html
    assert client.post('/pilot-demo').status_code == 405
    with app.extensions['studio_store'].connect() as db:
        assert db.execute('SELECT COUNT(*) FROM runs').fetchone()[0] == 0
    with client.session_transaction() as session:
        assert '_user_id' not in session


def test_pilot_assets_and_preview_launch_link(app):
    client = app.test_client()
    for asset in ['studio_pilot.js', 'studio_pilot.css']:
        assert client.get('/static/'+asset).status_code == 200
    assert 'href="/pilot-demo"' in client.get('/preview').get_data(as_text=True)
    script = client.get('/static/studio_pilot.js').get_data(as_text=True)
    for forbidden in ['fetch(', 'XMLHttpRequest', 'localStorage', 'sessionStorage', 'sendBeacon', 'speechSynthesis']:
        assert forbidden not in script


def test_opening_scene_is_readable_without_javascript(app):
    html = app.test_client().get('/pilot-demo').get_data(as_text=True)
    assert 'Welcome to the case record</h2>' in html
    assert 'One working record</h3>' in html
    assert 'Fictional information only.' in html
    assert '<button disabled id="play"' in html
    assert 'only the static opening view is shown' in html
    assert 'Static opening view. Playback controls activate' in html
