"""
DREDGE String Theory Module
Implements string theory models for integration with DREDGE and Quasimoto.
Provides string vibration modes, dimensional analysis, and theoretical physics calculations.
Supports GPU acceleration for enhanced performance.
"""
import math
from typing import List, Dict, Any, Tuple
import torch
import torch.nn as nn

# Physical constants
PLANCK_LENGTH = 1.616e-35  # meters

# Computational constants
DEFAULT_KK_MODES = 10  # Number of Kaluza-Klein modes to compute

# Neural network constants
MIN_NN_LAYERS = 1  # Minimum number of hidden layers
MAX_NN_LAYERS = 10  # Maximum number of hidden layers


def get_optimal_device() -> str:
    """
    Detect and return the optimal device for computation.
    
    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    try:
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
    except Exception:
        # If any error occurs during device detection, fall back to CPU
        pass
    return 'cpu'


def get_device_info() -> Dict[str, Any]:
    """
    Get detailed information about available compute devices.
    
    Returns:
        Dictionary with device capabilities
    """
    info = {
        'optimal_device': get_optimal_device(),
        'cpu_available': True,
        'cuda_available': torch.cuda.is_available(),
        'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
    }
    
    if info['cuda_available']:
        info['cuda_device_count'] = torch.cuda.device_count()
        info['cuda_device_name'] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None
        info['cuda_version'] = torch.version.cuda
    
    return info


class StringVibration:
    """
    String vibration model implementing fundamental string theory concepts.
    
    Models the vibration modes of a fundamental string in various dimensions.
    """
    
    def __init__(self, dimensions: int = 10, length: float = 1.0):
        """
        Initialize string vibration model.
        
        Args:
            dimensions: Number of spacetime dimensions (default: 10 for superstring theory)
            length: String length in Planck units (default: 1.0)
        """
        self.dimensions = dimensions
        self.length = length
        self.planck_constant = 1.0  # Normalized units
        
    def vibrational_mode(self, n: int, x: float) -> float:
        """
        Calculate the amplitude of the nth vibrational mode at position x.
        
        Args:
            n: Mode number (n >= 1)
            x: Position along string (0 <= x <= 1)
            
        Returns:
            Amplitude at position x for mode n
        """
        if n < 1:
            raise ValueError("Mode number must be >= 1")
        if not (0 <= x <= 1):
            raise ValueError("Position must be between 0 and 1")
        
        return math.sin(n * math.pi * x)
    
    def energy_level(self, n: int) -> float:
        """
        Calculate energy level for the nth mode.
        
        E_n = n * h / (2L) in natural units
        
        Args:
            n: Mode number
            
        Returns:
            Energy of the mode
        """
        return n * self.planck_constant / (2 * self.length)
    
    def mode_spectrum(self, max_modes: int = 10) -> List[float]:
        """
        Generate energy spectrum for modes up to max_modes.
        
        Args:
            max_modes: Maximum mode number
            
        Returns:
            List of energy levels
        """
        return [self.energy_level(n) for n in range(1, max_modes + 1)]

    def batch_vibrational_modes(
        self, modes: List[int], positions: List[float]
    ) -> List[List[float]]:
        """
        Compute vibrational amplitudes for all (mode, position) pairs in one
        vectorised operation using PyTorch.

        Args:
            modes: List of mode numbers (n >= 1).  If empty, returns ``[]``.
            positions: List of positions (0 <= x <= 1).  If empty, each row
                       in the result will also be an empty list.

        Returns:
            2-D list of shape ``(len(modes), len(positions))`` where
            ``result[i][j]`` is the amplitude for ``modes[i]`` at
            ``positions[j]``.  Returns ``[]`` when ``modes`` is empty.
        """
        n_t = torch.tensor(modes, dtype=torch.float32)
        x_t = torch.tensor(positions, dtype=torch.float32)
        # (M, 1) * (1, P) → (M, P)
        result = torch.sin(n_t.unsqueeze(1) * math.pi * x_t.unsqueeze(0))
        return result.tolist()
    
    def dimensional_compactification(self, radius: float) -> Dict[str, Any]:
        """
        Calculate effects of dimensional compactification.
        
        Models Kaluza-Klein dimensional reduction.
        
        Args:
            radius: Compactification radius
            
        Returns:
            Dictionary with compactification parameters
        """
        # Kaluza-Klein momentum quantization
        kk_modes = [n / radius for n in range(1, DEFAULT_KK_MODES + 1)]
        
        return {
            "compactification_radius": radius,
            "kaluza_klein_modes": kk_modes,
            "effective_dimensions": 4,  # 3 spatial + 1 time
            "hidden_dimensions": self.dimensions - 4
        }


class StringTheoryNN(nn.Module):
    """
    Enhanced neural network model for string theory calculations.
    
    Supports configurable depth, GPU acceleration, and batch normalization.
    Integrates with Quasimoto wave functions to model string dynamics.
    """
    
    def __init__(self, dimensions: int = 10, hidden_size: int = 64, 
                 num_layers: int = 2, use_batch_norm: bool = False,
                 device: str = 'cpu'):
        """
        Initialize string theory neural network.
        
        Args:
            dimensions: Input dimensionality (spacetime dimensions)
            hidden_size: Hidden layer size
            num_layers: Number of hidden layers (default: 2, supports 1-10)
            use_batch_norm: Whether to use batch normalization
            device: Device to run on ('cpu', 'cuda', or 'mps')
        """
        super().__init__()
        self.dimensions = dimensions
        self.hidden_size = hidden_size
        self.num_layers = max(MIN_NN_LAYERS, min(num_layers, MAX_NN_LAYERS))  # Clamp between limits
        self.use_batch_norm = use_batch_norm
        self.device = device
        
        # Build network layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(dimensions, hidden_size))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(nn.Tanh())
        
        # Hidden layers
        for _ in range(self.num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Tanh())
        
        # Output layer
        layers.append(nn.Linear(hidden_size, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Move to device
        self.to(self.device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute string amplitude.
        
        Args:
            x: Input tensor of shape (batch, dimensions)
            
        Returns:
            String amplitude predictions
        """
        # Ensure input is on correct device
        x = x.to(self.device)
        return self.network(x)

    def batch_infer(self, inputs: List[List[float]]) -> List[float]:
        """
        Run inference on a batch of input vectors in a single GPU pass.

        This is the primary high-throughput entry point.  All samples are
        forwarded together, maximising GPU amortisation.

        The model is evaluated with ``torch.no_grad()``; it must be in
        ``eval`` mode for deterministic results (call ``model.eval()`` before
        using this method if batch normalisation layers are present).

        Args:
            inputs: List of coordinate vectors, each of length ``dimensions``.

        Returns:
            List of scalar amplitude predictions (one per input vector).
            Returns ``[]`` when ``inputs`` is empty.
        """
        if not inputs:
            return []
        batch = torch.tensor(inputs, dtype=torch.float32)
        with torch.no_grad():
            output = self.forward(batch)   # shape: (N, 1)
        return output.squeeze(1).tolist()
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get information about the device being used."""
        return {
            'device': self.device,
            'cuda_available': torch.cuda.is_available(),
            'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
            'num_layers': self.num_layers,
            'use_batch_norm': self.use_batch_norm
        }


class StringQuasimotoIntegration:
    """
    Integration layer between String Theory and Quasimoto models.
    
    Combines string vibration modes with quantum wave functions.
    """
    
    def __init__(self, dimensions: int = 10):
        """
        Initialize integration layer.
        
        Args:
            dimensions: Number of dimensions
        """
        self.dimensions = dimensions
        self.string_vibration = StringVibration(dimensions=dimensions)
        self.string_nn = StringTheoryNN(dimensions=dimensions)
        
    def coupled_amplitude(
        self, 
        string_modes: List[int], 
        quasimoto_coords: List[float]
    ) -> float:
        """
        Calculate coupled amplitude between string modes and wave functions.
        
        Args:
            string_modes: List of string vibrational mode numbers
            quasimoto_coords: Quasimoto wave function coordinates
            
        Returns:
            Coupled amplitude value
        """
        # String contribution
        string_energy = sum(
            self.string_vibration.energy_level(n) for n in string_modes
        )
        
        # Position-dependent coupling
        if quasimoto_coords:
            position_factor = sum(abs(c) for c in quasimoto_coords) / len(quasimoto_coords)
        else:
            position_factor = 1.0
        
        return string_energy * position_factor
    
    def generate_unified_field(
        self, 
        x_range: Tuple[float, float] = (0.0, 1.0), 
        num_points: int = 100
    ) -> Dict[str, List[float]]:
        """
        Generate a unified field combining string and quantum effects.
        
        Args:
            x_range: Range of x coordinates
            num_points: Number of sampling points
            
        Returns:
            Dictionary with coordinates and field values
        """
        # Number of modes to average for field calculation
        NUM_MODES = 3
        
        x_min, x_max = x_range
        x_values = [x_min + (x_max - x_min) * i / (num_points - 1) for i in range(num_points)]
        
        # Generate field values using first NUM_MODES modes
        field_values = []
        for x in x_values:
            amplitude = sum(
                self.string_vibration.vibrational_mode(n, x) 
                for n in range(1, NUM_MODES + 1)
            ) / float(NUM_MODES)
            field_values.append(amplitude)
        
        return {
            "x_coordinates": x_values,
            "field_amplitudes": field_values,
            "dimensions": self.dimensions
        }


def calculate_string_parameters(
    energy_scale: float = 1.0,
    coupling_constant: float = 0.1
) -> Dict[str, Any]:
    """
    Calculate fundamental string theory parameters.
    
    Args:
        energy_scale: Energy scale in GeV
        coupling_constant: String coupling constant g_s
        
    Returns:
        Dictionary of calculated parameters
    """
    # String length (Planck scale)
    string_length = PLANCK_LENGTH * math.sqrt(coupling_constant)
    
    # String tension
    tension = 1.0 / (2.0 * math.pi * coupling_constant)
    
    return {
        "string_length": string_length,
        "string_tension": tension,
        "coupling_constant": coupling_constant,
        "energy_scale": energy_scale,
        "planck_length": PLANCK_LENGTH
    }


class DREDGEStringTheoryServer:
    """
    Enhanced server component integrating DREDGE, Quasimoto, and String Theory.
    
    Provides unified interface for all three theoretical frameworks with
    GPU acceleration, caching, and monitoring support.
    """
    
    def __init__(self, use_cache: bool = True, device: str = 'auto'):
        """
        Initialize DREDGE String Theory server.
        
        Args:
            use_cache: Whether to enable result caching
            device: Device to use ('auto', 'cpu', 'cuda', or 'mps')
        """
        self.string_vibration = StringVibration()
        self.integration = StringQuasimotoIntegration()
        self.models: Dict[str, nn.Module] = {}
        
        # Determine device
        self.device = get_optimal_device() if device == 'auto' else device
        
        # Initialize cache if enabled
        self.use_cache = use_cache
        if use_cache:
            from .cache import ResultCache
            self.cache = ResultCache()
        else:
            self.cache = None
        
    def load_string_model(
        self, 
        dimensions: int = 10, 
        hidden_size: int = 64,
        num_layers: int = 2,
        use_batch_norm: bool = False
    ) -> Dict[str, Any]:
        """
        Load a string theory neural network model.
        
        Args:
            dimensions: Spacetime dimensions
            hidden_size: Neural network hidden layer size
            num_layers: Number of hidden layers (1-10)
            use_batch_norm: Whether to use batch normalization
            
        Returns:
            Model information
        """
        model_id = f"string_theory_{len(self.models)}"
        model = StringTheoryNN(
            dimensions=dimensions, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            use_batch_norm=use_batch_norm,
            device=self.device
        )
        
        n_params = sum(p.numel() for p in model.parameters())
        
        self.models[model_id] = model
        
        return {
            "success": True,
            "model_id": model_id,
            "dimensions": dimensions,
            "n_parameters": n_params,
            "num_layers": num_layers,
            "device": self.device,
            "device_info": model.get_device_info()
        }
    
    def compute_string_spectrum(
        self, 
        max_modes: int = 10, 
        dimensions: int = 10
    ) -> Dict[str, Any]:
        """
        Compute string vibrational spectrum with caching.
        
        Args:
            max_modes: Maximum number of modes
            dimensions: Number of dimensions
            
        Returns:
            Spectrum data
        """
        # Check cache first
        if self.cache:
            cached = self.cache.get_spectrum(max_modes, dimensions)
            if cached:
                cached['cached'] = True
                return cached
        
        vibration = StringVibration(dimensions=dimensions)
        spectrum = vibration.mode_spectrum(max_modes=max_modes)
        
        result = {
            "success": True,
            "dimensions": dimensions,
            "max_modes": max_modes,
            "energy_spectrum": spectrum,
            "cached": False
        }
        
        # Cache the result
        if self.cache:
            self.cache.set_spectrum(max_modes, dimensions, result)
        
        return result
    
    def unified_inference(
        self,
        dredge_insight: str,
        quasimoto_coords: List[float],
        string_modes: List[int]
    ) -> Dict[str, Any]:
        """
        Unified inference combining DREDGE, Quasimoto, and String Theory with caching.
        
        Args:
            dredge_insight: DREDGE insight text
            quasimoto_coords: Quasimoto wave function coordinates
            string_modes: String vibrational modes
            
        Returns:
            Combined inference results
        """
        # Check cache first
        if self.cache:
            cached = self.cache.get_unified_inference(dredge_insight, quasimoto_coords, string_modes)
            if cached:
                cached['cached'] = True
                return cached
        
        # Compute coupled amplitude
        amplitude = self.integration.coupled_amplitude(
            string_modes=string_modes,
            quasimoto_coords=quasimoto_coords
        )
        
        # Generate unified field
        field = self.integration.generate_unified_field()
        
        result = {
            "success": True,
            "dredge_insight": dredge_insight,
            "quasimoto_coordinates": quasimoto_coords,
            "string_modes": string_modes,
            "coupled_amplitude": amplitude,
            "unified_field": field,
            "device": self.device,
            "cached": False
        }
        
        # Cache the result
        if self.cache:
            self.cache.set_unified_inference(dredge_insight, quasimoto_coords, string_modes, result)
        
        return result
