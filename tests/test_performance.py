"""Performance tests for DREDGE x Dolly server."""
import json
import re
import time
import pytest
from dredge.server import create_app


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app = create_app()
    return app.test_client()


def test_lift_endpoint_performance(client):
    """Test that the lift endpoint processes requests efficiently.
    
    Note: Uses relaxed threshold for CI/shared runner compatibility.
    Performance should inform, not fail builds unnecessarily.
    """
    # Warm up
    payload = {'insight_text': 'Performance test insight.'}
    client.post('/lift', data=json.dumps(payload), content_type='application/json')
    
    # Measure time for multiple requests
    num_requests = 100
    start_time = time.time()
    
    for i in range(num_requests):
        payload = {'insight_text': f'Performance test insight {i}.'}
        response = client.post(
            '/lift',
            data=json.dumps(payload),
            content_type='application/json'
        )
        assert response.status_code == 200
    
    elapsed_time = time.time() - start_time
    avg_time_per_request = elapsed_time / num_requests
    
    # Relaxed threshold: 25ms per request (accounts for CI variance)
    # With optimized hash, typical performance is <1ms on good hardware
    assert avg_time_per_request < 0.025, \
        f"Average request time too slow: {avg_time_per_request:.4f}s"
    
    print(f"\n✓ Processed {num_requests} requests in {elapsed_time:.3f}s")
    print(f"✓ Average time per request: {avg_time_per_request*1000:.2f}ms")


def test_hash_id_consistency(client):
    """Test that the same input produces the same ID consistently."""
    payload = {'insight_text': 'Consistent test input.'}
    
    # Make multiple requests with the same input
    ids = []
    for _ in range(5):
        response = client.post(
            '/lift',
            data=json.dumps(payload),
            content_type='application/json'
        )
        data = json.loads(response.data)
        ids.append(data['id'])
    
    # All IDs should be identical for the same input
    assert len(set(ids)) == 1, "Hash IDs should be consistent for the same input"


def test_hash_id_uniqueness(client):
    """Test that different inputs produce different IDs."""
    ids = set()
    
    for i in range(100):
        payload = {'insight_text': f'Unique test input {i}.'}
        response = client.post(
            '/lift',
            data=json.dumps(payload),
            content_type='application/json'
        )
        data = json.loads(response.data)
        ids.add(data['id'])
    
    # Should have 100 unique IDs (hash collisions are extremely unlikely)
    assert len(ids) >= 99, f"Expected ~100 unique IDs, got {len(ids)}"


def test_compact_json_response(client):
    """Test that JSON responses use compact encoding (no extra whitespace).
    
    Uses structural comparison rather than substring matching to avoid
    false positives when user content contains separator patterns.
    """
    response = client.get('/health')
    assert response.status_code == 200
    
    response_text = response.data.decode('utf-8').strip()
    
    # Verify the response is valid JSON
    data = json.loads(response_text)
    assert data['status'] == 'healthy'
    
    # Structural test: compact should be smaller than pretty-printed
    compact = json.dumps(data, separators=(',', ':'))
    pretty = json.dumps(data, separators=(', ', ': '))
    
    assert len(response_text) == len(compact), \
        f"Response size {len(response_text)} != compact size {len(compact)}"
    assert len(compact) < len(pretty), \
        "Compact JSON should be smaller than pretty-printed"
    
    print(f"\n✓ Response size (compact): {len(response_text)} bytes")
    
    compression_ratio = (1 - len(compact) / len(pretty)) * 100
    print(f"✓ Size reduction: {compression_ratio:.1f}% smaller than non-compact JSON")


def test_id_shape_contract(client):
    """Test that IDs conform to the expected format: 16 hex characters.
    
    This locks the identity contract and ensures IDs are always valid
    64-bit hexadecimal representations.
    """
    payload = {'insight_text': 'Test insight for ID validation.'}
    response = client.post(
        '/lift',
        data=json.dumps(payload),
        content_type='application/json'
    )
    
    assert response.status_code == 200
    data = json.loads(response.data)
    
    # ID must be exactly 16 hexadecimal characters (64 bits)
    assert re.fullmatch(r'[0-9a-f]{16}', data['id']), \
        f"ID '{data['id']}' does not match expected format [0-9a-f]{{16}}"
    
    print(f"\n✓ ID format validated: {data['id']}")


def test_collision_behavior(client):
    """Document collision behavior: same content produces same ID.
    
    This is a synthetic test that demonstrates the deterministic nature
    of the hash function. In a real collision scenario (different inputs
    producing the same ID), the last write would win as IDs are labels,
    not proofs. For the 64-bit space, collisions are extremely unlikely
    at modest scale (<5e9 items).
    """
    # Same input should produce same ID (deterministic)
    payload1 = {'insight_text': 'Identical content'}
    payload2 = {'insight_text': 'Identical content'}
    
    response1 = client.post(
        '/lift',
        data=json.dumps(payload1),
        content_type='application/json'
    )
    response2 = client.post(
        '/lift',
        data=json.dumps(payload2),
        content_type='application/json'
    )
    
    data1 = json.loads(response1.data)
    data2 = json.loads(response2.data)
    
    # Deterministic: same content → same ID
    assert data1['id'] == data2['id'], \
        "Same content should produce same ID (deterministic hash)"
    
    # Different content should produce different IDs (high probability)
    payload3 = {'insight_text': 'Different content'}
    response3 = client.post(
        '/lift',
        data=json.dumps(payload3),
        content_type='application/json'
    )
    data3 = json.loads(response3.data)
    
    assert data1['id'] != data3['id'], \
        "Different content should produce different IDs"
    
    print(f"\n✓ Collision behavior validated: deterministic and collision-resistant")
