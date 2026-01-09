"""Tests for the DREDGE x Dolly server."""
import json
import pytest
from dredge.server import create_app


@pytest.fixture
def app():
    """Create and configure a test Flask app instance."""
    return create_app()


@pytest.fixture
def client(app):
    """Create a test client for the Flask app."""
    return app.test_client()


def test_server_creation(app):
    """Test that the Flask app can be created."""
    assert app is not None


def test_root_endpoint(client):
    """Test the root endpoint returns API information."""
    response = client.get('/')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert data['name'] == 'DREDGE x Dolly'
    assert 'version' in data
    assert 'endpoints' in data


def test_health_endpoint(client):
    """Test the health check endpoint."""
    response = client.get('/health')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert data['status'] == 'healthy'
    assert 'version' in data


def test_lift_endpoint_success(client):
    """Test the lift endpoint with valid input."""
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


def test_lift_endpoint_missing_field(client):
    """Test the lift endpoint with missing required field."""
    payload = {}
    response = client.post(
        '/lift',
        data=json.dumps(payload),
        content_type='application/json'
    )
    
    assert response.status_code == 400
    
    data = json.loads(response.data)
    assert 'error' in data
