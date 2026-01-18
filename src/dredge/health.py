"""
Health check and system information utilities for DREDGE.
"""
import sys
import os
import platform
import shutil
from typing import Dict, Any


def get_system_info() -> Dict[str, Any]:
    """Get system information for diagnostics."""
    try:
        import torch
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        # Safely check for MPS availability
        mps_available = False
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps'):
            mps_available = torch.backends.mps.is_available()
    except ImportError:
        torch_version = "not installed"
        cuda_available = False
        mps_available = False
    
    try:
        import flask
        flask_version = flask.__version__
    except ImportError:
        flask_version = "not installed"
    
    uname = platform.uname()
    release_lower = uname.release.lower()
    is_termux = "TERMUX_VERSION" in os.environ
    is_ish = "alpine" in release_lower or "ish" in release_lower
    
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "mps_available": mps_available,
        "flask_version": flask_version,
        "term_width": shutil.get_terminal_size(fallback=(80, 24)).columns,
        "is_termux": is_termux,
        "is_ish": is_ish,
    }


def check_dependencies() -> Dict[str, bool]:
    """Check if required dependencies are installed."""
    checks = {}
    
    # Check Flask
    try:
        import flask
        checks["flask"] = True
    except ImportError:
        checks["flask"] = False
    
    # Check PyTorch
    try:
        import torch
        checks["torch"] = True
    except ImportError:
        checks["torch"] = False
    
    # Check NumPy
    try:
        import numpy
        checks["numpy"] = True
    except ImportError:
        checks["numpy"] = False
    
    # Check matplotlib
    try:
        import matplotlib
        checks["matplotlib"] = True
    except ImportError:
        checks["matplotlib"] = False
    
    return checks


def format_system_info(info: Dict[str, Any]) -> str:
    """Format system information as a readable string."""
    lines = [
        "System Information:",
        f"  Python: {info['python_version']}",
        f"  Platform: {info['platform']}",
        f"  Machine: {info['machine']}",
        f"  Terminal: {info['term_width']} columns",
        "",
        "Dependencies:",
        f"  Flask: {info['flask_version']}",
        f"  PyTorch: {info['torch_version']}",
        f"  CUDA available: {info['cuda_available']}",
        f"  MPS available: {info['mps_available']}",
        "",
        "Environment:",
        f"  Termux: {info['is_termux']}",
        f"  iSH: {info['is_ish']}",
    ]
    return "\n".join(lines)


def validate_server_config(host: str, port: int, debug: bool) -> None:
    """Validate server configuration and provide helpful error messages."""
    # Check port range
    if not (1 <= port <= 65535):
        raise ValueError(f"Port must be between 1 and 65535, got {port}")
    
    # Check if port is privileged
    if port < 1024 and hasattr(os, 'geteuid') and os.geteuid() != 0:
        print(f"Warning: Port {port} is privileged and may require sudo/admin rights", file=sys.stderr)
    
    # Check common port conflicts
    common_ports = {
        80: "HTTP",
        443: "HTTPS",
        3000: "Node.js/React dev server",
        5000: "Flask default",
        8080: "HTTP alternate",
    }
    if port in common_ports:
        print(f"Note: Port {port} is commonly used by {common_ports[port]}", file=sys.stderr)
    
    # Validate host
    if host not in ["0.0.0.0", "localhost", "127.0.0.1"] and not host.startswith("192.168."):
        print(f"Warning: Binding to {host} may expose the server publicly", file=sys.stderr)


def check_health() -> Dict[str, Any]:
    """Perform a comprehensive health check."""
    health = {
        "status": "healthy",
        "checks": {},
        "system": get_system_info(),
    }
    
    # Check dependencies
    deps = check_dependencies()
    health["checks"]["dependencies"] = deps
    
    # Check if all required dependencies are present
    required = ["flask", "torch", "numpy"]
    missing = [dep for dep in required if not deps.get(dep, False)]
    
    if missing:
        health["status"] = "unhealthy"
        health["missing_dependencies"] = missing
    
    return health
