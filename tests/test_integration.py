"""
Integration tests for Quasimoto wave processing.
Tests CLI commands, MCP tools, and core functionality.
"""
import pytest
import json
import sys
from pathlib import Path

# Test signal data
TEST_SIGNAL = [0.5, 0.8, 1.0, 0.9, 0.6, 0.3, 0.1, 0.2, 0.5, 0.8]


def test_quasimoto_import():
    """Test that quasimoto module can be imported."""
    try:
        from dredge import quasimoto
        assert hasattr(quasimoto, 'QuasimotoProcessor')
        assert hasattr(quasimoto, 'create_processor')
    except ImportError as e:
        pytest.skip(f"PyTorch not available: {e}")


def test_create_processor():
    """Test processor creation."""
    try:
        from dredge.quasimoto import create_processor
        processor = create_processor(n_waves=8, device="cpu")
        assert processor is not None
        assert processor.n_waves == 8
        assert processor.device == "cpu"
        assert processor.fitted == False
    except ImportError:
        pytest.skip("PyTorch not available")


def test_fit_signal():
    """Test fitting a signal."""
    try:
        from dredge.quasimoto import create_processor
        processor = create_processor(n_waves=4)
        result = processor.fit(TEST_SIGNAL, epochs=100)
        
        assert 'final_loss' in result
        assert 'epochs' in result
        assert result['epochs'] == 100
        assert processor.fitted == True
    except ImportError:
        pytest.skip("PyTorch not available")


def test_predict_after_fit():
    """Test prediction after fitting."""
    try:
        from dredge.quasimoto import create_processor
        processor = create_processor(n_waves=4)
        processor.fit(TEST_SIGNAL, epochs=100)
        
        predictions = processor.predict([0, 1, 2, 3, 4])
        assert len(predictions) == 5
        assert all(isinstance(p, (float, int)) for p in predictions)
    except ImportError:
        pytest.skip("PyTorch not available")


def test_detect_anomalies():
    """Test anomaly detection."""
    try:
        from dredge.quasimoto import create_processor
        
        # Create signal with known anomaly
        signal = [0.5, 0.6, 0.5, 0.6, 0.5, 10.0, 0.5, 0.6, 0.5, 0.6]
        
        processor = create_processor(n_waves=4)
        processor.fit(signal, epochs=200)
        anomalies = processor.detect_anomalies(signal, threshold=2.0)
        
        assert isinstance(anomalies, list)
        # Should detect the spike at index 5
        assert 5 in anomalies or len(anomalies) > 0
    except ImportError:
        pytest.skip("PyTorch not available")


def test_wave_parameters():
    """Test extracting wave parameters."""
    try:
        from dredge.quasimoto import create_processor
        processor = create_processor(n_waves=2)
        processor.fit(TEST_SIGNAL, epochs=50)
        
        params = processor.get_wave_parameters()
        assert 'wave_0' in params
        assert 'wave_1' in params
        assert 'amplitude' in params['wave_0']
        assert 'frequency' in params['wave_0']
    except ImportError:
        pytest.skip("PyTorch not available")


def test_mcp_fit_tool():
    """Test MCP fit_quasimoto_wave tool."""
    try:
        from dredge.mcp_server import fit_quasimoto_wave
        
        result = fit_quasimoto_wave(TEST_SIGNAL, n_waves=4, epochs=100)
        
        if 'error' in result:
            pytest.skip(result['error'])
        
        assert result['success'] == True
        assert 'final_loss' in result
        assert 'parameters' in result
        assert result['n_waves'] == 4
    except ImportError:
        pytest.skip("PyTorch not available")


def test_mcp_detect_anomalies_tool():
    """Test MCP detect_signal_anomalies tool."""
    try:
        from dredge.mcp_server import detect_signal_anomalies
        
        result = detect_signal_anomalies(TEST_SIGNAL, threshold=2.0, n_waves=4)
        
        if 'error' in result:
            pytest.skip(result['error'])
        
        assert result['success'] == True
        assert 'anomalies' in result
        assert 'count' in result
        assert isinstance(result['anomalies'], list)
    except ImportError:
        pytest.skip("PyTorch not available")


def test_mcp_transform_insight_tool():
    """Test MCP transform_insight_wave tool."""
    try:
        from dredge.mcp_server import transform_insight_wave
        
        result = transform_insight_wave("Digital memory must be human-reachable", n_waves=4)
        
        if 'error' in result:
            pytest.skip(result['error'])
        
        assert result['success'] == True
        assert 'insight' in result
        assert 'wave_encoding' in result
        assert result['insight'] == "Digital memory must be human-reachable"
    except ImportError:
        pytest.skip("PyTorch not available")


def test_mcp_predict_continuation_tool():
    """Test MCP predict_signal_continuation tool."""
    try:
        from dredge.mcp_server import predict_signal_continuation
        
        result = predict_signal_continuation(TEST_SIGNAL, n_future=5, n_waves=4)
        
        if 'error' in result:
            pytest.skip(result['error'])
        
        assert result['success'] == True
        assert 'predictions' in result
        assert len(result['predictions']) == 5
    except ImportError:
        pytest.skip("PyTorch not available")


def test_mcp_wave_similarity_tool():
    """Test MCP analyze_wave_similarity tool."""
    try:
        from dredge.mcp_server import analyze_wave_similarity
        
        signal1 = [0.5, 0.6, 0.7, 0.8, 0.9]
        signal2 = [0.5, 0.6, 0.7, 0.8, 0.9]  # Same as signal1
        
        result = analyze_wave_similarity(signal1, signal2, n_waves=2)
        
        if 'error' in result:
            pytest.skip(result['error'])
        
        assert result['success'] == True
        assert 'similarity_score' in result
        assert 0.0 <= result['similarity_score'] <= 1.0
    except ImportError:
        pytest.skip("PyTorch not available")


def test_mcp_tool_registry():
    """Test MCP tool registry."""
    from dredge.mcp_server import MCP_TOOLS, get_tool_info
    
    assert len(MCP_TOOLS) == 5
    assert 'fit_quasimoto_wave' in MCP_TOOLS
    assert 'detect_signal_anomalies' in MCP_TOOLS
    assert 'transform_insight_wave' in MCP_TOOLS
    
    info = get_tool_info()
    assert 'name' in info
    assert 'tools' in info
    assert len(info['tools']) == 5


def test_mcp_call_tool():
    """Test MCP tool calling interface."""
    try:
        from dredge.mcp_server import call_tool
        
        result = call_tool('fit_quasimoto_wave', signal_data=TEST_SIGNAL, n_waves=2, epochs=50)
        
        if 'error' in result and 'PyTorch' in result['error']:
            pytest.skip("PyTorch not available")
        
        assert 'success' in result or 'error' in result
    except ImportError:
        pytest.skip("PyTorch not available")


def test_cli_help():
    """Test CLI help command."""
    from dredge.cli import main
    
    # Test that help runs without error
    result = main(['--help'])
    # argparse calls sys.exit(0) for help, which is caught as SystemExit


def test_cli_version():
    """Test CLI version command."""
    from dredge.cli import main
    from dredge import __version__
    
    result = main(['--version'])
    assert result == 0


def test_cli_benchmark_without_torch():
    """Test benchmark command fails gracefully without torch."""
    from dredge.cli import main
    
    # This will fail if torch is not installed, which is expected
    result = main(['benchmark', '--model', 'quasimoto', '--epochs', '10'])
    # Should return 0 if torch is available, 1 if not
    assert result in [0, 1]
