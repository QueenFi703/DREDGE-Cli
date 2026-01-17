"""Tests for the DREDGE String Theory module."""
import pytest
import math
from dredge.string_theory import (
    StringVibration,
    StringTheoryNN,
    StringQuasimocoIntegration,
    DREDGEStringTheoryServer,
    calculate_string_parameters
)
import torch


def test_string_vibration_creation():
    """Test that StringVibration can be created."""
    sv = StringVibration(dimensions=10, length=1.0)
    assert sv is not None
    assert sv.dimensions == 10
    assert sv.length == 1.0


def test_vibrational_mode():
    """Test vibrational mode calculation."""
    sv = StringVibration()
    
    # Test mode at x=0 (should be 0 for all modes)
    assert sv.vibrational_mode(1, 0.0) == 0.0
    
    # Test mode at x=0.5 (should be maximum for odd modes)
    mode1_mid = sv.vibrational_mode(1, 0.5)
    assert abs(mode1_mid - 1.0) < 1e-10
    
    # Test mode at x=1.0 (should be 0 for all modes)
    assert abs(sv.vibrational_mode(1, 1.0)) < 1e-10


def test_energy_level():
    """Test energy level calculation."""
    sv = StringVibration(length=1.0)
    
    # Energy should be proportional to mode number
    e1 = sv.energy_level(1)
    e2 = sv.energy_level(2)
    
    assert e2 == 2 * e1


def test_mode_spectrum():
    """Test mode spectrum generation."""
    sv = StringVibration()
    spectrum = sv.mode_spectrum(max_modes=10)
    
    assert len(spectrum) == 10
    # Energy should increase with mode number
    for i in range(len(spectrum) - 1):
        assert spectrum[i + 1] > spectrum[i]


def test_dimensional_compactification():
    """Test dimensional compactification calculation."""
    sv = StringVibration(dimensions=10)
    result = sv.dimensional_compactification(radius=1.0)
    
    assert result["compactification_radius"] == 1.0
    assert result["effective_dimensions"] == 4
    assert result["hidden_dimensions"] == 6
    assert len(result["kaluza_klein_modes"]) == 10


def test_string_theory_nn_creation():
    """Test StringTheoryNN neural network creation."""
    model = StringTheoryNN(dimensions=10, hidden_size=64)
    assert model is not None
    assert model.dimensions == 10
    assert model.hidden_size == 64


def test_string_theory_nn_forward():
    """Test StringTheoryNN forward pass."""
    model = StringTheoryNN(dimensions=10, hidden_size=64)
    
    # Create input tensor
    x = torch.randn(5, 10)  # batch_size=5, dimensions=10
    
    # Forward pass
    output = model(x)
    
    assert output.shape == (5, 1)


def test_string_quasimoco_integration():
    """Test StringQuasimocoIntegration."""
    integration = StringQuasimocoIntegration(dimensions=10)
    
    assert integration is not None
    assert integration.dimensions == 10


def test_coupled_amplitude():
    """Test coupled amplitude calculation."""
    integration = StringQuasimocoIntegration()
    
    amplitude = integration.coupled_amplitude(
        string_modes=[1, 2, 3],
        quasimoto_coords=[0.5, 0.5, 0.5]
    )
    
    assert amplitude > 0
    assert isinstance(amplitude, float)


def test_generate_unified_field():
    """Test unified field generation."""
    integration = StringQuasimocoIntegration()
    
    field = integration.generate_unified_field(
        x_range=(0.0, 1.0),
        num_points=50
    )
    
    assert "x_coordinates" in field
    assert "field_amplitudes" in field
    assert "dimensions" in field
    assert len(field["x_coordinates"]) == 50
    assert len(field["field_amplitudes"]) == 50


def test_calculate_string_parameters():
    """Test calculation of fundamental string parameters."""
    params = calculate_string_parameters(
        energy_scale=1.0,
        coupling_constant=0.1
    )
    
    assert "string_length" in params
    assert "string_tension" in params
    assert "coupling_constant" in params
    assert "energy_scale" in params
    assert "planck_length" in params
    
    assert params["coupling_constant"] == 0.1
    assert params["energy_scale"] == 1.0


def test_dredge_string_theory_server_creation():
    """Test DREDGEStringTheoryServer creation."""
    server = DREDGEStringTheoryServer()
    
    assert server is not None
    assert isinstance(server.models, dict)
    assert len(server.models) == 0


def test_load_string_model():
    """Test loading a string theory model."""
    server = DREDGEStringTheoryServer()
    
    result = server.load_string_model(dimensions=10, hidden_size=64)
    
    assert result["success"] is True
    assert "model_id" in result
    assert result["dimensions"] == 10
    assert result["n_parameters"] > 0


def test_compute_string_spectrum():
    """Test computing string spectrum."""
    server = DREDGEStringTheoryServer()
    
    result = server.compute_string_spectrum(max_modes=10, dimensions=10)
    
    assert result["success"] is True
    assert result["dimensions"] == 10
    assert result["max_modes"] == 10
    assert "energy_spectrum" in result
    assert len(result["energy_spectrum"]) == 10


def test_unified_inference():
    """Test unified inference."""
    server = DREDGEStringTheoryServer()
    
    result = server.unified_inference(
        dredge_insight="Test insight",
        quasimoto_coords=[0.5, 0.5],
        string_modes=[1, 2, 3]
    )
    
    assert result["success"] is True
    assert result["dredge_insight"] == "Test insight"
    assert result["quasimoto_coordinates"] == [0.5, 0.5]
    assert result["string_modes"] == [1, 2, 3]
    assert "coupled_amplitude" in result
    assert "unified_field" in result


def test_vibrational_mode_invalid_inputs():
    """Test vibrational mode with invalid inputs."""
    sv = StringVibration()
    
    # Test invalid mode number
    with pytest.raises(ValueError):
        sv.vibrational_mode(0, 0.5)
    
    # Test invalid position
    with pytest.raises(ValueError):
        sv.vibrational_mode(1, 1.5)


def test_string_theory_parameter_count():
    """Test that string theory models have correct parameter counts."""
    model = StringTheoryNN(dimensions=10, hidden_size=64)
    
    n_params = sum(p.numel() for p in model.parameters())
    
    # Expected parameters:
    # input_layer: 10 * 64 + 64 = 704
    # hidden_layer: 64 * 64 + 64 = 4160
    # output_layer: 64 * 1 + 1 = 65
    # Total: 4929
    assert n_params == 4929
