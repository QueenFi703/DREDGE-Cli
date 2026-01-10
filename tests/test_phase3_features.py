"""Tests for Phase III features: Hot reload and enhanced metrics."""
import json
import pytest
from dredge.server import create_app


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app = create_app()
    return app.test_client()


def test_metrics_endpoint_exists(client):
    """Test that /metrics endpoint is available."""
    response = client.get('/metrics')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert "uptime_seconds" in data
    assert "request_count" in data
    assert "lift_count" in data
    assert "latency_ms" in data


def test_metrics_structure(client):
    """Test metrics endpoint returns proper structure."""
    response = client.get('/metrics')
    data = json.loads(response.data)
    
    # Check latency metrics
    assert "p50" in data["latency_ms"]
    assert "p95" in data["latency_ms"]
    assert "p99" in data["latency_ms"]
    assert "mean" in data["latency_ms"]
    
    # Check other metrics
    assert isinstance(data["uptime_seconds"], (int, float))
    assert isinstance(data["request_count"], int)
    assert isinstance(data["lift_count"], int)


def test_enhanced_health_endpoint(client):
    """Test enhanced health endpoint with checks."""
    response = client.get('/health')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert data["status"] in ["healthy", "degraded"]
    assert "checks" in data
    assert "uptime_seconds" in data["checks"]


def test_metrics_track_lift_requests(client):
    """Test that metrics track lift endpoint usage."""
    # Get initial metrics
    response = client.get('/metrics')
    initial_data = json.loads(response.data)
    initial_lift_count = initial_data["lift_count"]
    
    # Make a lift request
    payload = {'insight_text': 'Test for metrics tracking'}
    client.post('/lift', data=json.dumps(payload), content_type='application/json')
    
    # Check metrics again
    response = client.get('/metrics')
    new_data = json.loads(response.data)
    
    assert new_data["lift_count"] == initial_lift_count + 1
    assert new_data["request_count"] > initial_data["request_count"]


def test_metrics_latency_tracking(client):
    """Test that latency metrics are tracked."""
    # Make several requests
    for _ in range(5):
        client.get('/health')
    
    response = client.get('/metrics')
    data = json.loads(response.data)
    
    # Should have latency data now
    assert data["latency_ms"]["mean"] >= 0
    assert data["latency_ms"]["p50"] >= 0


def test_root_endpoint_includes_metrics(client):
    """Test that root endpoint lists /metrics."""
    response = client.get('/')
    data = json.loads(response.data)
    
    assert "/metrics" in data["endpoints"]
    assert data["endpoints"]["/metrics"] == "Performance metrics"


def test_health_requests_per_second(client):
    """Test that health endpoint calculates requests per second."""
    # Make some requests
    for _ in range(3):
        client.get('/health')
    
    response = client.get('/health')
    data = json.loads(response.data)
    
    assert "requests_per_second" in data["checks"]
    assert data["checks"]["requests_per_second"] > 0
