"""Tests for the DREDGE x Dolly server."""
import json
from unittest.mock import patch
from dredge.server import create_app
from dredge.auth import User, _users


def _make_authed_client(app):
    """Return a test client pre-seeded with an authenticated session."""
    user = User(
        user_id="test:001",
        name="Test User",
        email="test@example.com",
        provider="test",
    )
    _users["test:001"] = user

    client = app.test_client()
    with client.session_transaction() as sess:
        sess["_user_id"] = user.id
        sess["_fresh"] = True
    return client


def test_server_creation():
    """Test that the Flask app can be created."""
    app = create_app()
    assert app is not None


def test_root_endpoint():
    """Test the root endpoint returns API information (authenticated)."""
    app = create_app()
    client = _make_authed_client(app)

    response = client.get('/')
    assert response.status_code == 200

    data = json.loads(response.data)
    assert data['name'] == 'DREDGE x Dolly'
    assert 'version' in data
    assert 'endpoints' in data


def test_health_endpoint():
    """Test the health check endpoint (public — no auth required)."""
    app = create_app()
    client = app.test_client()

    response = client.get('/health')
    assert response.status_code == 200

    data = json.loads(response.data)
    assert data['status'] == 'healthy'
    assert 'version' in data


def test_lift_endpoint_success():
    """Test the lift endpoint with valid input (authenticated)."""
    app = create_app()
    client = _make_authed_client(app)

    payload = {'insight_text': 'Digital memory must be human-reachable.'}
    response = client.post(
        '/lift',
        data=json.dumps(payload),
        content_type='application/json'
    )

    assert response.status_code == 200

    data = json.loads(response.data)
    assert 'id' in data
    assert data['lifted'] is True
    assert data['text'] == payload['insight_text']


def test_lift_endpoint_missing_field():
    """Test the lift endpoint with missing required field (authenticated)."""
    app = create_app()
    client = _make_authed_client(app)

    payload = {}
    response = client.post(
        '/lift',
        data=json.dumps(payload),
        content_type='application/json'
    )

    assert response.status_code == 400

    data = json.loads(response.data)
    assert 'error' in data


def test_login_page_accessible():
    """Login page must be publicly accessible (unauthenticated)."""
    app = create_app()
    client = app.test_client()
    response = client.get('/auth/login')
    assert response.status_code == 200


def test_auth_status_unauthenticated():
    """/auth/status returns authenticated=False for anonymous requests."""
    app = create_app()
    client = app.test_client()
    response = client.get('/auth/status')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['authenticated'] is False


def test_protected_route_redirects_to_login():
    """Unauthenticated requests to protected routes redirect to login."""
    app = create_app()
    client = app.test_client()
    response = client.get('/')
    assert response.status_code == 302
    assert '/auth/login' in response.headers.get('Location', '')


