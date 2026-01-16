"""
DREDGE MCP Server
Model Context Protocol server exposing Quasimoto wave processing tools.
"""
from typing import List, Dict, Any, Optional
import json

from . import __version__

# MCP server tools for Quasimoto integration
# These tools can be called by AI agents through the MCP protocol


def fit_quasimoto_wave(signal_data: List[float], dimensions: int = 1, n_waves: int = 16, epochs: int = 2000) -> Dict[str, Any]:
    """
    Fit Quasimoto wave function to signal data using GPU acceleration.
    
    This tool trains an ensemble of Quasimoto waves to represent the input signal,
    capturing both global patterns and localized irregularities.
    
    Args:
        signal_data: List of signal values to fit
        dimensions: Number of dimensions (1D, 4D, or 6D) - currently only 1D supported
        n_waves: Number of waves in the ensemble (default: 16)
        epochs: Number of training epochs (default: 2000)
        
    Returns:
        Dictionary containing:
        - final_loss: Training loss achieved
        - parameters: Learned wave parameters
        - n_waves: Number of waves used
        - epochs: Training epochs completed
    """
    try:
        from .quasimoto import create_processor
    except ImportError:
        return {
            "error": "PyTorch is required for wave fitting",
            "install_command": "pip install torch"
        }
    
    if dimensions != 1:
        return {"error": f"Only 1D signals currently supported, got {dimensions}D"}
    
    try:
        processor = create_processor(n_waves=n_waves, device="cpu")
        result = processor.fit(signal_data, epochs=epochs)
        parameters = processor.get_wave_parameters()
        
        return {
            "success": True,
            "final_loss": result["final_loss"],
            "parameters": parameters,
            "n_waves": n_waves,
            "epochs": epochs,
            "signal_length": len(signal_data)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def detect_signal_anomalies(signal_data: List[float], threshold: float = 2.0, n_waves: int = 16) -> Dict[str, Any]:
    """
    Detect localized irregularities in signal using ensemble waves.
    
    Fits Quasimoto wave ensemble to signal and identifies points where
    the residual exceeds the threshold (in standard deviations).
    
    Args:
        signal_data: List of signal values
        threshold: Number of standard deviations for anomaly detection (default: 2.0)
        n_waves: Number of waves in ensemble (default: 16)
        
    Returns:
        Dictionary containing:
        - anomalies: List of indices where anomalies detected
        - count: Number of anomalies found
        - percentage: Percentage of signal that is anomalous
        - threshold: Threshold used
    """
    try:
        from .quasimoto import create_processor
    except ImportError:
        return {
            "error": "PyTorch is required",
            "install_command": "pip install torch"
        }
    
    try:
        processor = create_processor(n_waves=n_waves)
        processor.fit(signal_data, epochs=1500)
        anomalies = processor.detect_anomalies(signal_data, threshold=threshold)
        
        return {
            "success": True,
            "anomalies": anomalies,
            "count": len(anomalies),
            "percentage": 100 * len(anomalies) / len(signal_data),
            "threshold": threshold,
            "signal_length": len(signal_data)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def transform_insight_wave(insight_text: str, n_waves: int = 16) -> Dict[str, Any]:
    """
    Transform insight text through Quasimoto wave representation.
    
    Converts text insight into a wave-encoded representation that can be
    stored and retrieved based on wave similarity.
    
    Args:
        insight_text: Text insight to transform
        n_waves: Number of waves in ensemble (default: 16)
        
    Returns:
        Dictionary containing:
        - insight: Original insight text
        - wave_encoding: Wave parameters
        - fit_quality: How well the wave represents the insight
        - n_waves: Number of waves used
    """
    try:
        from .quasimoto import create_processor
    except ImportError:
        return {
            "error": "PyTorch is required",
            "install_command": "pip install torch"
        }
    
    try:
        processor = create_processor(n_waves=n_waves)
        result = processor.transform_insight(insight_text)
        
        return {
            "success": True,
            **result
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def predict_signal_continuation(signal_data: List[float], n_future: int = 50, n_waves: int = 16) -> Dict[str, Any]:
    """
    Predict future signal values using fitted Quasimoto waves.
    
    Fits waves to the input signal and extrapolates to predict future values.
    
    Args:
        signal_data: Historical signal data
        n_future: Number of future points to predict (default: 50)
        n_waves: Number of waves in ensemble (default: 16)
        
    Returns:
        Dictionary containing:
        - predictions: List of predicted future values
        - fit_loss: Quality of fit to historical data
        - n_future: Number of predictions made
    """
    try:
        from .quasimoto import create_processor
    except ImportError:
        return {
            "error": "PyTorch is required",
            "install_command": "pip install torch"
        }
    
    try:
        processor = create_processor(n_waves=n_waves)
        result = processor.fit(signal_data, epochs=1500)
        
        # Generate future indices
        future_indices = list(range(len(signal_data), len(signal_data) + n_future))
        predictions = processor.predict(future_indices)
        
        return {
            "success": True,
            "predictions": predictions,
            "fit_loss": result["final_loss"],
            "n_future": n_future,
            "historical_length": len(signal_data)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def analyze_wave_similarity(signal1: List[float], signal2: List[float], n_waves: int = 16) -> Dict[str, Any]:
    """
    Analyze similarity between two signals using wave representations.
    
    Fits Quasimoto waves to both signals and compares their wave parameters
    to determine structural similarity.
    
    Args:
        signal1: First signal
        signal2: Second signal
        n_waves: Number of waves in ensemble (default: 16)
        
    Returns:
        Dictionary containing:
        - similarity_score: Measure of wave parameter similarity
        - signal1_params: Wave parameters for signal 1
        - signal2_params: Wave parameters for signal 2
    """
    try:
        from .quasimoto import create_processor
        import numpy as np
    except ImportError:
        return {
            "error": "PyTorch and NumPy are required",
            "install_command": "pip install torch numpy"
        }
    
    try:
        # Fit to both signals
        proc1 = create_processor(n_waves=n_waves)
        proc2 = create_processor(n_waves=n_waves)
        
        proc1.fit(signal1, epochs=1000)
        proc2.fit(signal2, epochs=1000)
        
        params1 = proc1.get_wave_parameters()
        params2 = proc2.get_wave_parameters()
        
        # Simple similarity based on parameter differences
        # In production, would use more sophisticated metric
        similarities = []
        for i in range(n_waves):
            p1 = params1[f"wave_{i}"]
            p2 = params2[f"wave_{i}"]
            
            # Compare key parameters
            diff = abs(p1["amplitude"] - p2["amplitude"])
            diff += abs(p1["frequency"] - p2["frequency"])
            similarities.append(1.0 / (1.0 + diff))
        
        avg_similarity = np.mean(similarities)
        
        return {
            "success": True,
            "similarity_score": float(avg_similarity),
            "signal1_length": len(signal1),
            "signal2_length": len(signal2),
            "n_waves": n_waves
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# MCP Server tool registry
MCP_TOOLS = {
    "fit_quasimoto_wave": {
        "function": fit_quasimoto_wave,
        "description": "Fit Quasimoto wave function to signal data using GPU acceleration",
        "parameters": {
            "signal_data": {"type": "array", "description": "Signal values to fit"},
            "dimensions": {"type": "integer", "default": 1, "description": "Number of dimensions"},
            "n_waves": {"type": "integer", "default": 16, "description": "Number of waves"},
            "epochs": {"type": "integer", "default": 2000, "description": "Training epochs"}
        }
    },
    "detect_signal_anomalies": {
        "function": detect_signal_anomalies,
        "description": "Detect localized irregularities in signal using ensemble waves",
        "parameters": {
            "signal_data": {"type": "array", "description": "Signal to analyze"},
            "threshold": {"type": "number", "default": 2.0, "description": "Anomaly threshold (std devs)"},
            "n_waves": {"type": "integer", "default": 16, "description": "Number of waves"}
        }
    },
    "transform_insight_wave": {
        "function": transform_insight_wave,
        "description": "Transform insight text through Quasimoto wave representation",
        "parameters": {
            "insight_text": {"type": "string", "description": "Text insight to transform"},
            "n_waves": {"type": "integer", "default": 16, "description": "Number of waves"}
        }
    },
    "predict_signal_continuation": {
        "function": predict_signal_continuation,
        "description": "Predict future signal values using fitted Quasimoto waves",
        "parameters": {
            "signal_data": {"type": "array", "description": "Historical signal data"},
            "n_future": {"type": "integer", "default": 50, "description": "Points to predict"},
            "n_waves": {"type": "integer", "default": 16, "description": "Number of waves"}
        }
    },
    "analyze_wave_similarity": {
        "function": analyze_wave_similarity,
        "description": "Analyze similarity between two signals using wave representations",
        "parameters": {
            "signal1": {"type": "array", "description": "First signal"},
            "signal2": {"type": "array", "description": "Second signal"},
            "n_waves": {"type": "integer", "default": 16, "description": "Number of waves"}
        }
    }
}


def get_tool_info() -> Dict[str, Any]:
    """Get information about available MCP tools."""
    return {
        "name": "DREDGE MCP Server",
        "version": __version__,
        "tools": {
            name: {
                "description": tool["description"],
                "parameters": tool["parameters"]
            }
            for name, tool in MCP_TOOLS.items()
        }
    }


def call_tool(tool_name: str, **kwargs) -> Dict[str, Any]:
    """
    Call an MCP tool by name.
    
    Args:
        tool_name: Name of the tool to call
        **kwargs: Tool-specific parameters
        
    Returns:
        Tool result dictionary
    """
    if tool_name not in MCP_TOOLS:
        return {
            "error": f"Unknown tool: {tool_name}",
            "available_tools": list(MCP_TOOLS.keys())
        }
    
    tool = MCP_TOOLS[tool_name]
    try:
        return tool["function"](**kwargs)
    except Exception as e:
        return {
            "error": f"Tool execution failed: {str(e)}",
            "tool": tool_name
        }
