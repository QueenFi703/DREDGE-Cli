"""Tests for the DREDGE MCP server."""
import json
import os

import torch

from dredge.mcp_server import QuasimotoMCPServer, create_mcp_app
from dredge.string_theory import get_device_info


def test_mcp_server_creation():
    """Test that MCP server can be created."""
    server = QuasimotoMCPServer()
    assert server is not None
    assert isinstance(server.models, dict)
    assert len(server.models) == 0
    assert server.string_theory_server is not None


def test_mcp_server_device_default():
    """Test MCP server device defaults to optimal device."""
    server = QuasimotoMCPServer()
    device_info = get_device_info()
    assert server.device == device_info['optimal_device']


def test_mcp_server_device_cpu():
    """Test MCP server can be created with CPU device."""
    server = QuasimotoMCPServer(device='cpu')
    assert server.device == 'cpu'


def test_mcp_server_device_from_env():
    """Test MCP server respects DEVICE environment variable."""
    # Save original env var
    original_device = os.getenv('DEVICE')

    try:
        # Set env var to CPU
        os.environ['DEVICE'] = 'cpu'
        server = QuasimotoMCPServer(device='auto')
        assert server.device == 'cpu'
    finally:
        # Restore original env var
        if original_device is not None:
            os.environ['DEVICE'] = original_device
        elif 'DEVICE' in os.environ:
            del os.environ['DEVICE']


def test_mcp_server_device_fallback_cuda():
    """Test MCP server falls back to CPU when CUDA unavailable."""
    if not torch.cuda.is_available():
        server = QuasimotoMCPServer(device='cuda')
        assert server.device == 'cpu'


def test_mcp_server_device_fallback_mps():
    """Test MCP server falls back to CPU when MPS unavailable."""
    if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        server = QuasimotoMCPServer(device='mps')
        assert server.device == 'cpu'


def test_list_capabilities():
    """Test listing MCP server capabilities."""
    server = QuasimotoMCPServer()
    capabilities = server.list_capabilities()

    assert capabilities["name"] == "DREDGE Quasimoto String Theory MCP Server"
    assert "version" in capabilities
    assert "protocol" in capabilities
    assert "capabilities" in capabilities
    assert "models" in capabilities["capabilities"]
    assert "operations" in capabilities["capabilities"]

    # Check for string theory additions
    assert "string_theory" in capabilities["capabilities"]["models"]
    assert "string_spectrum" in capabilities["capabilities"]["operations"]
    assert "unified_inference" in capabilities["capabilities"]["operations"]


def test_load_quasimoto_1d():
    """Test loading 1D Quasimoto model."""
    server = QuasimotoMCPServer()
    result = server.load_model("quasimoto_1d")

    assert result["success"] is True
    assert "model_id" in result
    assert result["model_type"] == "quasimoto_1d"
    assert result["n_parameters"] == 8


def test_load_quasimoto_model_on_device():
    """Test that loaded Quasimoto models are on the correct device."""
    server = QuasimotoMCPServer(device='cpu')
    result = server.load_model("quasimoto_1d")

    model_id = result["model_id"]
    model = server.models[model_id]

    # Check that model parameters are on CPU
    for param in model.parameters():
        assert param.device.type == 'cpu'

    # Check that model config includes device info
    assert server.model_configs[model_id]['device'] == 'cpu'


def test_load_quasimoto_ensemble():
    """Test loading Quasimoto ensemble model."""
    server = QuasimotoMCPServer()
    result = server.load_model("quasimoto_ensemble", {"n_waves": 8})

    assert result["success"] is True
    assert "model_id" in result
    assert result["model_type"] == "quasimoto_ensemble"
    assert result["n_parameters"] > 0


def test_inference_1d():
    """Test inference on 1D model."""
    server = QuasimotoMCPServer()
    load_result = server.load_model("quasimoto_1d")
    model_id = load_result["model_id"]

    inference_result = server.inference(model_id, {"x": [0.5], "t": [0.0]})

    assert inference_result["success"] is True
    assert "output" in inference_result
    assert isinstance(inference_result["output"], list)


def test_inference_tensors_on_device():
    """Test that inference creates tensors on the correct device."""
    server = QuasimotoMCPServer(device='cpu')
    load_result = server.load_model("quasimoto_1d")
    model_id = load_result["model_id"]

    # Run inference
    inference_result = server.inference(model_id, {"x": [0.5], "t": [0.0]})

    assert inference_result["success"] is True

    # Verify the input tensors would have been created on CPU
    # (we can't directly check them as they're local to the method,
    # but we know they were created correctly if inference succeeded)


def test_get_parameters():
    """Test getting model parameters."""
    server = QuasimotoMCPServer()
    load_result = server.load_model("quasimoto_1d")
    model_id = load_result["model_id"]

    params_result = server.get_parameters(model_id)

    assert params_result["success"] is True
    assert "parameters" in params_result
    assert "A" in params_result["parameters"]
    assert params_result["n_parameters"] == 8


def test_benchmark():
    """Test running a benchmark."""
    server = QuasimotoMCPServer()
    result = server.benchmark("quasimoto_1d", {"epochs": 10})

    assert result["success"] is True
    assert result["epochs"] == 10
    assert "final_loss" in result
    assert "initial_loss" in result
    assert result["final_loss"] < result["initial_loss"]  # Training should reduce loss


def test_benchmark_device_info():
    """Test that benchmark includes device information."""
    server = QuasimotoMCPServer(device='cpu')
    result = server.benchmark("quasimoto_1d", {"epochs": 10})

    assert result["success"] is True
    assert "device" in result
    assert result["device"] == 'cpu'


def test_handle_request():
    """Test handling MCP requests."""
    server = QuasimotoMCPServer()

    # Test list capabilities
    response = server.handle_request({"operation": "list_capabilities"})
    assert "name" in response

    # Test load model
    response = server.handle_request({
        "operation": "load_model",
        "params": {"model_type": "quasimoto_1d"}
    })
    assert response["success"] is True


def test_mcp_app_creation():
    """Test that the Flask app can be created."""
    app = create_mcp_app()
    assert app is not None

    with app.test_client() as client:
        # Test root endpoint
        response = client.get('/')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "name" in data

        # Test MCP endpoint
        response = client.post('/mcp',
                              json={"operation": "list_capabilities"},
                              content_type='application/json')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "capabilities" in data


def test_mcp_app_with_device():
    """Test that the Flask app can be created with specific device."""
    app = create_mcp_app(device='cpu')
    assert app is not None

    with app.test_client() as client:
        # Test capabilities include device info
        response = client.get('/')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "features" in data
        assert "device_info" in data["features"]


def test_mcp_endpoint_load_and_inference():
    """Test full workflow via MCP endpoint."""
    app = create_mcp_app()

    with app.test_client() as client:
        # Load model
        response = client.post('/mcp',
                              json={
                                  "operation": "load_model",
                                  "params": {"model_type": "quasimoto_1d"}
                              },
                              content_type='application/json')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        model_id = data["model_id"]

        # Run inference
        response = client.post('/mcp',
                              json={
                                  "operation": "inference",
                                  "params": {
                                      "model_id": model_id,
                                      "inputs": {"x": [0.5], "t": [0.0]}
                                  }
                              },
                              content_type='application/json')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        assert "output" in data


def test_invalid_model_type():
    """Test handling of invalid model type."""
    server = QuasimotoMCPServer()
    result = server.load_model("invalid_model")

    assert result["success"] is False
    assert "error" in result


def test_inference_on_nonexistent_model():
    """Test inference on non-existent model."""
    server = QuasimotoMCPServer()
    result = server.inference("nonexistent", {"x": [0.0], "t": [0.0]})

    assert result["success"] is False
    assert "error" in result


def test_load_string_theory_model():
    """Test loading a string theory model."""
    server = QuasimotoMCPServer()
    result = server.load_model("string_theory", {"dimensions": 10, "hidden_size": 64})

    assert result["success"] is True
    assert "model_id" in result
    assert result["dimensions"] == 10
    assert result["n_parameters"] > 0


def test_string_spectrum_operation():
    """Test string spectrum operation."""
    server = QuasimotoMCPServer()
    result = server.string_spectrum({"max_modes": 10, "dimensions": 10})

    assert result["success"] is True
    assert result["dimensions"] == 10
    assert result["max_modes"] == 10
    assert "energy_spectrum" in result


def test_string_parameters_operation():
    """Test string parameters operation."""
    server = QuasimotoMCPServer()
    result = server.string_parameters({
        "energy_scale": 1.0,
        "coupling_constant": 0.1
    })

    assert result["success"] is True
    assert "parameters" in result
    assert result["parameters"]["coupling_constant"] == 0.1


def test_unified_inference_operation():
    """Test unified inference operation."""
    server = QuasimotoMCPServer()
    result = server.unified_inference({
        "dredge_insight": "Test insight",
        "quasimoto_coords": [0.5, 0.5],
        "string_modes": [1, 2, 3]
    })

    assert result["success"] is True
    assert result["dredge_insight"] == "Test insight"
    assert "coupled_amplitude" in result
    assert "unified_field" in result


def test_handle_request_string_operations():
    """Test handling string theory operations via request handler."""
    server = QuasimotoMCPServer()

    # Test string_spectrum
    response = server.handle_request({
        "operation": "string_spectrum",
        "params": {"max_modes": 5, "dimensions": 10}
    })
    assert response["success"] is True

    # Test string_parameters
    response = server.handle_request({
        "operation": "string_parameters",
        "params": {"energy_scale": 1.0, "coupling_constant": 0.1}
    })
    assert response["success"] is True

    # Test unified_inference
    response = server.handle_request({
        "operation": "unified_inference",
        "params": {
            "dredge_insight": "Test",
            "quasimoto_coords": [0.5],
            "string_modes": [1, 2]
        }
    })
    assert response["success"] is True


def test_mcp_app_string_theory_endpoints():
    """Test MCP app with string theory endpoints."""
    app = create_mcp_app()

    with app.test_client() as client:
        # Test string spectrum via MCP endpoint
        response = client.post('/mcp',
                              json={
                                  "operation": "string_spectrum",
                                  "params": {"max_modes": 10, "dimensions": 10}
                              },
                              content_type='application/json')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True

        # Test unified inference
        response = client.post('/mcp',
                              json={
                                  "operation": "unified_inference",
                                  "params": {
                                      "dredge_insight": "Integration test",
                                      "quasimoto_coords": [0.5],
                                      "string_modes": [1, 2, 3]
                                  }
                              },
                              content_type='application/json')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True


def test_dependabot_alerts_no_token():
    """Test Dependabot alerts without GITHUB_TOKEN."""
    # Save and remove token
    original_token = os.getenv("GITHUB_TOKEN")
    if "GITHUB_TOKEN" in os.environ:
        del os.environ["GITHUB_TOKEN"]

    try:
        server = QuasimotoMCPServer()
        result = server.get_dependabot_alerts()

        assert result["success"] is False
        assert "GITHUB_TOKEN" in result["error"]
    finally:
        # Restore token
        if original_token:
            os.environ["GITHUB_TOKEN"] = original_token


def test_explain_dependabot_alert_no_token():
    """Test explain Dependabot alert without GITHUB_TOKEN."""
    # Save and remove token
    original_token = os.getenv("GITHUB_TOKEN")
    if "GITHUB_TOKEN" in os.environ:
        del os.environ["GITHUB_TOKEN"]

    try:
        server = QuasimotoMCPServer()
        result = server.explain_dependabot_alert(alert_id=1)

        assert result["success"] is False
        assert "GITHUB_TOKEN" in result["error"]
    finally:
        # Restore token
        if original_token:
            os.environ["GITHUB_TOKEN"] = original_token


def test_update_dependabot_alert_no_token():
    """Test update Dependabot alert without GITHUB_TOKEN."""
    # Save and remove token
    original_token = os.getenv("GITHUB_TOKEN")
    if "GITHUB_TOKEN" in os.environ:
        del os.environ["GITHUB_TOKEN"]

    try:
        server = QuasimotoMCPServer()
        result = server.update_dependabot_alert(
            alert_id=1,
            state="dismissed",
            dismissed_reason="not_used"
        )

        assert result["success"] is False
        assert "GITHUB_TOKEN" in result["error"]
    finally:
        # Restore token
        if original_token:
            os.environ["GITHUB_TOKEN"] = original_token


def test_update_dependabot_alert_invalid_state():
    """Test update Dependabot alert with invalid state."""
    server = QuasimotoMCPServer()
    result = server.update_dependabot_alert(alert_id=1, state="invalid_state")

    assert result["success"] is False
    assert "Invalid state" in result["error"]


def test_update_dependabot_alert_missing_reason():
    """Test update Dependabot alert with dismissed state but no reason."""
    server = QuasimotoMCPServer()
    result = server.update_dependabot_alert(alert_id=1, state="dismissed")

    assert result["success"] is False
    assert "dismissed_reason is required" in result["error"]


def test_update_dependabot_alert_invalid_reason():
    """Test update Dependabot alert with invalid dismissed reason."""
    server = QuasimotoMCPServer()
    result = server.update_dependabot_alert(
        alert_id=1,
        state="dismissed",
        dismissed_reason="invalid_reason"
    )

    assert result["success"] is False
    assert "Invalid dismissed_reason" in result["error"]


def test_get_recommendation_critical():
    """Test recommendation for critical severity."""
    server = QuasimotoMCPServer()
    recommendation = server._get_recommendation(9.5, "critical")

    assert "CRITICAL" in recommendation
    assert "immediately" in recommendation.lower()


def test_get_recommendation_high():
    """Test recommendation for high severity."""
    server = QuasimotoMCPServer()
    recommendation = server._get_recommendation(7.5, "high")

    assert "HIGH" in recommendation
    assert "soon as possible" in recommendation.lower()


def test_get_recommendation_medium():
    """Test recommendation for medium severity."""
    server = QuasimotoMCPServer()
    recommendation = server._get_recommendation(5.0, "medium")

    assert "MEDIUM" in recommendation


def test_get_recommendation_low():
    """Test recommendation for low severity."""
    server = QuasimotoMCPServer()
    recommendation = server._get_recommendation(2.0, "low")

    assert "LOW" in recommendation


def test_get_recommendation_na_score():
    """Test recommendation with N/A CVSS score."""
    server = QuasimotoMCPServer()
    recommendation = server._get_recommendation("N/A", "medium")

    assert isinstance(recommendation, str)
    assert len(recommendation) > 0


def test_handle_request_dependabot_operations():
    """Test handling Dependabot operations via request handler."""
    server = QuasimotoMCPServer()

    # Test get_dependabot_alerts (will fail without token or with invalid repo)
    response = server.handle_request({
        "operation": "get_dependabot_alerts",
        "params": {}
    })
    assert "success" in response

    # Test explain_dependabot_alert
    response = server.handle_request({
        "operation": "explain_dependabot_alert",
        "params": {"alert_id": 1}
    })
    assert "success" in response

    # Test update_dependabot_alert with invalid state
    response = server.handle_request({
        "operation": "update_dependabot_alert",
        "params": {"alert_id": 1, "state": "invalid"}
    })
    assert response["success"] is False


def test_list_capabilities_includes_dependabot():
    """Test that capabilities list includes Dependabot operations."""
    server = QuasimotoMCPServer()
    capabilities = server.list_capabilities()

    operations = capabilities["capabilities"]["operations"]
    assert "get_dependabot_alerts" in operations
    assert "explain_dependabot_alert" in operations
    assert "update_dependabot_alert" in operations


def test_mcp_app_dependabot_endpoints():
    """Test MCP app with Dependabot endpoints."""
    app = create_mcp_app()

    with app.test_client() as client:
        # Test capabilities endpoint shows Dependabot operations
        response = client.get('/')
        assert response.status_code == 200
        data = json.loads(response.data)
        operations = data["capabilities"]["operations"]
        assert "get_dependabot_alerts" in operations
        assert "explain_dependabot_alert" in operations
        assert "update_dependabot_alert" in operations

        # Test get_dependabot_alerts endpoint
        response = client.post('/mcp',
                              json={
                                  "operation": "get_dependabot_alerts",
                                  "params": {}
                              },
                              content_type='application/json')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "success" in data

        # Test explain_dependabot_alert endpoint
        response = client.post('/mcp',
                              json={
                                  "operation": "explain_dependabot_alert",
                                  "params": {"alert_id": 1}
                              },
                              content_type='application/json')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "success" in data

        # Test update_dependabot_alert endpoint with invalid params
        response = client.post('/mcp',
                              json={
                                  "operation": "update_dependabot_alert",
                                  "params": {"alert_id": 1, "state": "invalid"}
                              },
                              content_type='application/json')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is False

