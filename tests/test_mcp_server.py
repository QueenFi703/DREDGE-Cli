"""Tests for the DREDGE MCP server."""
import pytest
import json
from dredge.mcp_server import QuasimotoMCPServer, create_mcp_app


def test_mcp_server_creation():
    """Test that MCP server can be created."""
    server = QuasimotoMCPServer()
    assert server is not None
    assert isinstance(server.models, dict)
    assert len(server.models) == 0
    assert server.string_theory_server is not None


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

