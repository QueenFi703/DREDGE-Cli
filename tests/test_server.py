"""Tests for the DREDGE x Dolly server."""
import json
from dredge.server import create_app


def test_server_creation():
    """Test that the Flask app can be created."""
    app = create_app()
    assert app is not None


def test_root_endpoint():
    """Test the root endpoint returns API information."""
    app = create_app()
    client = app.test_client()
    
    response = client.get('/')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert data['name'] == 'DREDGE x Dolly'
    assert 'version' in data
    assert 'endpoints' in data


def test_health_endpoint():
    """Test the health check endpoint."""
    app = create_app()
    client = app.test_client()
    
    response = client.get('/health')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert data['status'] == 'healthy'
    assert 'version' in data


def test_lift_endpoint_success():
    """Test the lift endpoint with valid input."""
    app = create_app()
    client = app.test_client()
    
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
    """Test the lift endpoint with missing required field."""
    app = create_app()
    client = app.test_client()
    
    payload = {}
    response = client.post(
        '/lift',
        data=json.dumps(payload),
        content_type='application/json'
    )
    
    assert response.status_code == 400
    
    data = json.loads(response.data)
    assert 'error' in data
