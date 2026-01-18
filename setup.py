"""
Setup script for DREDGE-Cli with optional CUDA/C++ extension support.

This setup.py allows the package to build custom CUDA extensions when:
1. PyTorch is installed
2. CUDA and nvcc are available
3. The build is explicitly requested (e.g., pip install -e .)

If CUDA is not available, the package will install without extensions
and fall back to pure Python implementations at runtime.
"""

import os
import sys
import warnings
from setuptools import setup

# Try to build CUDA extensions if available
try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    # Check if CUDA is available
    if torch.cuda.is_available():
        # Define extension module
        ext_modules = [
            CUDAExtension(
                name="dredge_custom_ops",
                sources=[
                    "csrc/fused_wave.cpp",
                    "csrc/fused_wave_kernel.cu",
                ],
                extra_compile_args={
                    "cxx": ["-O3"],
                    "nvcc": ["-O3", "-lineinfo"],
                },
            )
        ]
        cmdclass = {"build_ext": BuildExtension}
        print("✓ CUDA available - building custom extensions")
    else:
        ext_modules = []
        cmdclass = {}
        print("ℹ CUDA not available - skipping custom extensions")
except ImportError:
    ext_modules = []
    cmdclass = {}
    print("ℹ PyTorch not installed - skipping custom extensions")
except Exception as e:
    ext_modules = []
    cmdclass = {}
    warnings.warn(f"Failed to setup CUDA extensions: {e}. Continuing without them.")

# Run setup
setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
