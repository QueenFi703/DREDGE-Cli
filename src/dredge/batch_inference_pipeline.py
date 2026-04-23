"""
DREDGE Batch Inference Pipeline
Provides batched neural-network inference over StringTheoryNN with GPU
memory pre-allocation for improved throughput.
"""
import time
from typing import Any, Dict, List, Optional

import torch

from .string_theory import StringTheoryNN, StringVibration, get_optimal_device

# Default batch size used when callers do not specify one
DEFAULT_BATCH_SIZE = 32


class BatchInferenceEngine:
    """
    High-throughput batched inference engine for StringTheoryNN.

    Key optimisations:
    * GPU memory pre-allocation for the maximum expected batch size.
    * Single ``forward`` call per batch instead of one per sample.
    * Optional warm-up pass to trigger JIT compilation / kernel caching.
    """

    def __init__(
        self,
        dimensions: int = 10,
        hidden_size: int = 64,
        num_layers: int = 2,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: str = "auto",
        preallocate: bool = True,
    ):
        """
        Initialise the engine.

        Args:
            dimensions: Spacetime dimensions passed to StringTheoryNN.
            hidden_size: Hidden-layer width.
            num_layers: Number of hidden layers.
            batch_size: Maximum batch size (used for pre-allocation).
            device: ``'auto'``, ``'cpu'``, ``'cuda'``, or ``'mps'``.
            preallocate: Pre-allocate GPU memory for ``batch_size`` inputs.
        """
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.device = get_optimal_device() if device == "auto" else device

        self.model = StringTheoryNN(
            dimensions=dimensions,
            hidden_size=hidden_size,
            num_layers=num_layers,
            device=self.device,
        )
        self.model.eval()

        # Pre-allocate input buffer on the target device
        self._input_buffer: Optional[torch.Tensor] = None
        if preallocate:
            self._input_buffer = torch.zeros(
                batch_size, dimensions, device=self.device
            )

    # ------------------------------------------------------------------

    def _warm_up(self) -> None:
        """Run a single dummy forward pass to trigger JIT / kernel caching."""
        dummy = torch.zeros(1, self.dimensions, device=self.device)
        with torch.no_grad():
            self.model(dummy)

    # ------------------------------------------------------------------

    def infer_batch(
        self, inputs: List[List[float]]
    ) -> List[float]:
        """
        Run inference on a list of input vectors in a single GPU pass.

        Args:
            inputs: List of coordinate vectors, each of length ``dimensions``.

        Returns:
            List of scalar amplitude predictions.
        """
        if not inputs:
            return []

        n = len(inputs)
        # Build tensor – reuse pre-allocated buffer when possible
        if self._input_buffer is not None and n <= self.batch_size:
            self._input_buffer[:n] = torch.tensor(
                inputs, dtype=torch.float32, device=self.device
            )
            batch_tensor = self._input_buffer[:n]
        else:
            batch_tensor = torch.tensor(
                inputs, dtype=torch.float32, device=self.device
            )

        with torch.no_grad():
            output = self.model(batch_tensor)  # shape: (n, 1)

        return output.squeeze(1).tolist()

    def infer_batch_chunked(
        self,
        inputs: List[List[float]],
        chunk_size: Optional[int] = None,
    ) -> List[float]:
        """
        Run inference on an arbitrarily large list, processing in chunks.

        Args:
            inputs: Input vectors.
            chunk_size: Chunk size (defaults to ``self.batch_size``).

        Returns:
            Flat list of scalar predictions.
        """
        chunk_size = chunk_size or self.batch_size
        results: List[float] = []
        for start in range(0, len(inputs), chunk_size):
            chunk = inputs[start : start + chunk_size]
            results.extend(self.infer_batch(chunk))
        return results

    # ------------------------------------------------------------------

    def benchmark(
        self,
        num_samples: int = 1000,
        chunk_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Measure throughput for ``num_samples`` random inputs.

        Args:
            num_samples: Number of random samples to process.
            chunk_size: Processing chunk size (defaults to ``self.batch_size``).

        Returns:
            Dictionary with throughput and latency statistics.
        """
        chunk_size = chunk_size or self.batch_size
        inputs = torch.randn(num_samples, self.dimensions).tolist()

        # Warm up
        self._warm_up()

        start = time.perf_counter()
        results = self.infer_batch_chunked(inputs, chunk_size=chunk_size)
        elapsed = time.perf_counter() - start

        return {
            "num_samples": num_samples,
            "chunk_size": chunk_size,
            "device": self.device,
            "total_seconds": elapsed,
            "throughput_per_sec": num_samples / elapsed if elapsed > 0 else 0,
            "latency_ms_per_sample": (elapsed / num_samples * 1000)
            if num_samples > 0
            else 0,
            "num_results": len(results),
        }


class BatchVibrationalModes:
    """
    Batch calculator for StringVibration mode spectra.

    Computes mode spectra for many (n, x) pairs in vectorised form
    using PyTorch rather than Python loops.
    """

    def __init__(self, dimensions: int = 10, length: float = 1.0):
        self.vibration = StringVibration(dimensions=dimensions, length=length)

    def compute_batch(
        self,
        modes: List[int],
        positions: List[float],
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Compute vibrational amplitudes for all (mode, position) combinations.

        Args:
            modes: List of mode numbers (n >= 1).
            positions: List of positions (0 <= x <= 1).
            device: Torch device to use.

        Returns:
            Tensor of shape ``(len(modes), len(positions))``.
        """
        import math

        n_t = torch.tensor(modes, dtype=torch.float32, device=device)
        x_t = torch.tensor(positions, dtype=torch.float32, device=device)
        # Broadcasting: (M, 1) * (1, P) → (M, P)
        return torch.sin(n_t.unsqueeze(1) * math.pi * x_t.unsqueeze(0))
