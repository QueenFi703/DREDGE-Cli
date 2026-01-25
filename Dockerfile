# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[...]
# DREDGE-Cli Multi-Stage Dockerfile
# Optimized for CPU (Flask) and GPU (PyTorch/Dolly) workloads
# 
# Platform Support:
# - Linux with NVIDIA GPU: Use gpu-build stage (CUDA 11.8)
# - Linux CPU-only: Use cpu-build stage
# - macOS with Apple Silicon (M1/M2/M3): MPS support not available in Docker
#   For macOS/MPS: Install locally with `pip install -e .` (PyTorch >=2.0 includes MPS)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[...]

# ────────────────────────────────────────────────────────────────[...]
# Stage 1: Base Python Environment
# ────────────────────────────────────────────────────────────────[...]
FROM python:3.14-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Layer 1: System dependencies with immediate cleanup
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Layer 2: Python dependencies
COPY requirements.txt pyproject.toml ./

# ────────────────────────────────────────────────────────────────[...]
# Stage 2: CPU-Only Build (Flask server)
# ────────────────────────────────────────────────────────────────[...]
FROM base AS cpu-build

# Layer 2: Python dependencies with cache cleanup
RUN pip install --no-cache-dir \
    flask>=3.0.0 \
    numpy>=1.24.0 \
    matplotlib>=3.5.0 \
    && rm -rf /root/.cache/pip \
    && rm -rf /tmp/*

# Layer 3: Application code
COPY src/ ./src/
COPY README.md LICENSE ./

RUN pip install -e . \
    && rm -rf /root/.cache/pip \
    && rm -rf /tmp/*

EXPOSE 3001

CMD ["dredge-cli", "serve", "--host", "0.0.0.0", "--port", "3001"]

# ────────────────────────────────────────────────────────────────[...]
# Stage 3: GPU Build (Full Dolly + Quasimoto)
# ────────────────────────────────────────────────────────────────[...]
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 AS gpu-build

# Layer 1: System dependencies with immediate cleanup
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    gcc \
    g++ \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app

# Layer 2: Python dependencies with pip cache purging
COPY requirements.txt pyproject.toml ./

# Upgrade pip and install build dependencies for PEP 660 editable install support
RUN pip3 install --no-cache-dir --upgrade pip setuptools>=64 wheel \
    && rm -rf /root/.cache/pip \
    && rm -rf /tmp/*

# Install PyTorch with CUDA 11.8 support (latest compatible versions)
# Note: Overrides requirements.txt torch>=2.0.0 with specific CUDA build
RUN pip3 install --no-cache-dir \
    torch \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu118 \
    && pip3 install --no-cache-dir \
    flask>=3.0.0 \
    numpy>=1.24.0 \
    matplotlib>=3.5.0 \
    && rm -rf /root/.cache/pip \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Layer 3: Application code (exclude benchmarks from GPU stage)
COPY src/ ./src/
COPY README.md LICENSE ./

RUN pip3 install -e . \
    && rm -rf /root/.cache/pip \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

EXPOSE 3001 3002

CMD ["dredge-cli", "mcp", "--host", "0.0.0.0", "--port", "3002"]

# ────────────────────────────────────────────────────────────────[...]
# Stage 4: Development Build
# ────────────────────────────────────────────────────────────────[...]
FROM gpu-build AS dev

# Layer 4: Development dependencies and dynamic files
RUN pip3 install --no-cache-dir \
    pytest>=7.0.0 \
    black>=23.0.0 \
    mypy>=1.0.0 \
    ruff>=0.1.0 \
    && rm -rf /root/.cache/pip \
    && rm -rf /tmp/*

COPY tests/ ./tests/
COPY benchmarks/ ./benchmarks/

CMD ["/bin/bash"]