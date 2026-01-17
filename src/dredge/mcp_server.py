"""
DREDGE MCP Server
Model Context Protocol server for serving Quasimoto wave function models.
Integrates Quasimoto benchmarks with MCP protocol for external applications.
"""
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn

# Import from benchmarks - assumes benchmarks are in Python path or installed
try:
    from quasimoto_extended_benchmark import (
        QuasimotoWave,
        QuasimotoWave4D,
        QuasimotoWave6D,
        QuasimotoEnsemble,
        generate_data
    )
except ImportError:
    # Fallback: add benchmarks to path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'benchmarks'))
    from quasimoto_extended_benchmark import (
        QuasimotoWave,
        QuasimotoWave4D,
        QuasimotoWave6D,
        QuasimotoEnsemble,
        generate_data
    )

from . import __version__
from .string_theory import (
    DREDGEStringTheoryServer,
    StringVibration,
    calculate_string_parameters
)


class QuasimotoMCPServer:
    """
    MCP Server for Quasimoto neural wave function models.
    
    Provides model inference, training, and benchmark capabilities
    via the Model Context Protocol.
    """
    
    def __init__(self):
        self.models: Dict[str, nn.Module] = {}
        self.model_configs: Dict[str, Dict[str, Any]] = {}
        self.string_theory_server = DREDGEStringTheoryServer()
        
    def list_capabilities(self) -> Dict[str, Any]:
        """List available MCP server capabilities."""
        return {
            "name": "DREDGE Quasimoto String Theory MCP Server",
            "version": __version__,
            "protocol": "Model Context Protocol v1.0",
            "capabilities": {
                "models": {
                    "quasimoto_1d": "1D wave function (8 parameters)",
                    "quasimoto_4d": "4D spatiotemporal wave function (13 parameters)",
                    "quasimoto_6d": "6D high-dimensional wave function (17 parameters)",
                    "quasimoto_ensemble": "Ensemble of wave functions (configurable)",
                    "string_theory": "String theory neural network (configurable dimensions)"
                },
                "operations": [
                    "load_model",
                    "inference",
                    "get_parameters",
                    "benchmark",
                    "string_spectrum",
                    "string_parameters",
                    "unified_inference"
                ]
            }
        }
    
    def load_model(self, model_type: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Load a Quasimoto model.
        
        Args:
            model_type: Type of model ('quasimoto_1d', 'quasimoto_4d', 'quasimoto_6d', 'quasimoto_ensemble')
            config: Optional configuration (e.g., ensemble size)
            
        Returns:
            Model information and status
        """
        config = config or {}
        model_id = f"{model_type}_{len(self.models)}"
        
        try:
            if model_type == "quasimoto_1d":
                model = QuasimotoWave()
            elif model_type == "quasimoto_4d":
                model = QuasimotoWave4D()
            elif model_type == "quasimoto_6d":
                model = QuasimotoWave6D()
            elif model_type == "quasimoto_ensemble":
                n_waves = config.get("n_waves", 16)
                model = QuasimotoEnsemble(n=n_waves)
            elif model_type == "string_theory":
                dimensions = config.get("dimensions", 10)
                hidden_size = config.get("hidden_size", 64)
                result = self.string_theory_server.load_string_model(
                    dimensions=dimensions,
                    hidden_size=hidden_size
                )
                if result["success"]:
                    result["config"] = config
                return result
            else:
                return {
                    "success": False,
                    "error": f"Unknown model type: {model_type}"
                }
            
            # Count parameters
            n_params = sum(p.numel() for p in model.parameters())
            
            self.models[model_id] = model
            self.model_configs[model_id] = {
                "type": model_type,
                "config": config,
                "n_parameters": n_params
            }
            
            return {
                "success": True,
                "model_id": model_id,
                "model_type": model_type,
                "n_parameters": n_params,
                "config": config
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _create_input_tensors(self, inputs: Dict[str, Any], keys: List[str]) -> List[torch.Tensor]:
        """
        Helper method to create input tensors from input dictionary.
        
        Args:
            inputs: Dictionary of input values
            keys: List of keys to extract from inputs
            
        Returns:
            List of tensors
        """
        return [torch.tensor(inputs.get(key, [0.0])) for key in keys]
    
    def inference(self, model_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run inference on a loaded model.
        
        Args:
            model_id: ID of the loaded model
            inputs: Input data (coordinates)
            
        Returns:
            Model outputs
        """
        if model_id not in self.models:
            return {
                "success": False,
                "error": f"Model not found: {model_id}"
            }
        
        try:
            model = self.models[model_id]
            model_type = self.model_configs[model_id]["type"]
            
            with torch.no_grad():
                if model_type == "quasimoto_1d":
                    x, t = self._create_input_tensors(inputs, ["x", "t"])
                    output = model(x, t)
                    
                elif model_type == "quasimoto_4d":
                    x, y, z, t = self._create_input_tensors(inputs, ["x", "y", "z", "t"])
                    output = model(x, y, z, t)
                    
                elif model_type == "quasimoto_6d":
                    x1, x2, x3, x4, x5, t = self._create_input_tensors(
                        inputs, ["x1", "x2", "x3", "x4", "x5", "t"]
                    )
                    output = model(x1, x2, x3, x4, x5, t)
                    
                elif model_type == "quasimoto_ensemble":
                    x, t = self._create_input_tensors(inputs, ["x", "t"])
                    output = model(x, t)
                    
                else:
                    return {
                        "success": False,
                        "error": f"Inference not implemented for {model_type}"
                    }
            
            return {
                "success": True,
                "model_id": model_id,
                "output": output.tolist()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_parameters(self, model_id: str) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Args:
            model_id: ID of the loaded model
            
        Returns:
            Model parameters and configuration
        """
        if model_id not in self.models:
            return {
                "success": False,
                "error": f"Model not found: {model_id}"
            }
        
        try:
            model = self.models[model_id]
            config = self.model_configs[model_id]
            
            params = {}
            for name, param in model.named_parameters():
                # Safely convert parameter to Python value
                if param.numel() == 1:
                    value = param.detach().squeeze().item()
                else:
                    value = param.detach().tolist()
                
                params[name] = {
                    "value": value,
                    "shape": list(param.shape),
                    "requires_grad": param.requires_grad
                }
            
            return {
                "success": True,
                "model_id": model_id,
                "model_type": config["type"],
                "n_parameters": config["n_parameters"],
                "parameters": params
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def benchmark(self, model_type: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run a benchmark on a model type.
        
        Args:
            model_type: Type of model to benchmark
            config: Benchmark configuration (epochs, etc.)
            
        Returns:
            Benchmark results
        """
        config = config or {}
        epochs = config.get("epochs", 100)
        
        try:
            if model_type == "quasimoto_1d" or model_type == "quasimoto_ensemble":
                # Generate 1D test data
                x, t, y = generate_data()
                
                if model_type == "quasimoto_1d":
                    model = QuasimotoWave()
                else:
                    n_waves = config.get("n_waves", 16)
                    model = QuasimotoEnsemble(n=n_waves)
                
                # Simple training loop
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
                losses = []
                
                for epoch in range(epochs):
                    optimizer.zero_grad()
                    pred = model(x, t)
                    loss = torch.nn.functional.mse_loss(pred, y)
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
                
                return {
                    "success": True,
                    "model_type": model_type,
                    "epochs": epochs,
                    "final_loss": losses[-1],
                    "initial_loss": losses[0],
                    "losses": losses[::max(1, epochs // 10)]  # Sample 10 points
                }
            else:
                return {
                    "success": False,
                    "error": f"Benchmark not implemented for {model_type}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle an MCP request.
        
        Args:
            request: MCP request with 'operation' and 'params'
            
        Returns:
            Response dict
        """
        operation = request.get("operation")
        params = request.get("params", {})
        
        if operation == "list_capabilities":
            return self.list_capabilities()
        elif operation == "load_model":
            return self.load_model(params.get("model_type"), params.get("config"))
        elif operation == "inference":
            return self.inference(params.get("model_id"), params.get("inputs", {}))
        elif operation == "get_parameters":
            return self.get_parameters(params.get("model_id"))
        elif operation == "benchmark":
            return self.benchmark(params.get("model_type"), params.get("config"))
        elif operation == "string_spectrum":
            return self.string_spectrum(params)
        elif operation == "string_parameters":
            return self.string_parameters(params)
        elif operation == "unified_inference":
            return self.unified_inference(params)
        else:
            return {
                "success": False,
                "error": f"Unknown operation: {operation}"
            }
    
    def string_spectrum(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute string theory vibrational spectrum.
        
        Args:
            params: Parameters including max_modes and dimensions
            
        Returns:
            String spectrum data
        """
        try:
            max_modes = params.get("max_modes", 10)
            dimensions = params.get("dimensions", 10)
            return self.string_theory_server.compute_string_spectrum(
                max_modes=max_modes,
                dimensions=dimensions
            )
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def string_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate fundamental string theory parameters.
        
        Args:
            params: Parameters including energy_scale and coupling_constant
            
        Returns:
            String theory parameters
        """
        try:
            energy_scale = params.get("energy_scale", 1.0)
            coupling_constant = params.get("coupling_constant", 0.1)
            result = calculate_string_parameters(
                energy_scale=energy_scale,
                coupling_constant=coupling_constant
            )
            return {
                "success": True,
                "parameters": result
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def unified_inference(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Unified inference combining DREDGE, Quasimoto, and String Theory.
        
        Args:
            params: Parameters including dredge_insight, quasimoto_coords, string_modes
            
        Returns:
            Combined inference results
        """
        try:
            dredge_insight = params.get("dredge_insight", "")
            quasimoto_coords = params.get("quasimoto_coords", [0.5])
            string_modes = params.get("string_modes", [1, 2, 3])
            
            return self.string_theory_server.unified_inference(
                dredge_insight=dredge_insight,
                quasimoto_coords=quasimoto_coords,
                string_modes=string_modes
            )
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


def create_mcp_app():
    """Create Flask app for MCP server."""
    from flask import Flask, jsonify, request
    
    app = Flask(__name__)
    server = QuasimotoMCPServer()
    
    @app.route('/')
    def index():
        """MCP server information."""
        return jsonify(server.list_capabilities())
    
    @app.route('/mcp', methods=['POST'])
    def mcp_endpoint():
        """MCP protocol endpoint."""
        try:
            request_data = request.get_json()
            response = server.handle_request(request_data)
            return jsonify(response)
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    return app


def run_mcp_server(host='0.0.0.0', port=3002, debug=False):
    """
    Run the DREDGE MCP server.
    
    Args:
        host: Host to bind to (default: 0.0.0.0)
        port: Port to listen on (default: 3002)
        debug: Enable debug mode (default: False)
    """
    app = create_mcp_app()
    print(f"🚀 Starting DREDGE MCP Server on http://{host}:{port}")
    print(f"📡 MCP Version: {__version__}")
    print(f"🌊 Quasimoto models available")
    print(f"🔧 Debug mode: {debug}")
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_mcp_server()
