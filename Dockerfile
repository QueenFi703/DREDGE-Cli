ARG PYTHON_VERSION=3.12
ARG SWIFT_IMAGE=swift:5.9-jammy

FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# shared setup
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/dredge

# Install the checked-out DREDGE source, not a fresh clone from GitHub.
COPY pyproject.toml setup.py requirements.txt README.md ./
COPY src ./src
COPY csrc ./csrc
RUN python -m pip install --upgrade pip && \
    python -m pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    grep -vE '^[[:space:]]*torch([<>=!~ ].*)?$' requirements.txt > /tmp/requirements-no-torch.txt && \
    python -m pip install -r /tmp/requirements-no-torch.txt && \
    python -m pip install -e .

# CPU image
FROM base AS cpu-build

# GPU image
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS gpu-build
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    git \
    python3-pip \
    python3.10 \
    python3.10-venv \
    && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch for GPU
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install DREDGE CLI for GPU from the checked-out source.
WORKDIR /opt/dredge
COPY pyproject.toml setup.py requirements.txt README.md ./
COPY src ./src
COPY csrc ./csrc
RUN grep -vE '^[[:space:]]*torch([<>=!~ ].*)?$' requirements.txt > /tmp/requirements-no-torch.txt && \
    python3 -m pip install -r /tmp/requirements-no-torch.txt && \
    python3 -m pip install -e .

FROM ${SWIFT_IMAGE} AS swift-toolchain

# Development image (Python + Swift toolchain + DREDGE Swift dependencies)
FROM swift-toolchain AS dev
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/dredge
COPY . /opt/dredge
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    grep -vE '^[[:space:]]*torch([<>=!~ ].*)?$' requirements.txt > /tmp/requirements-no-torch.txt && \
    python3 -m pip install -r /tmp/requirements-no-torch.txt && \
    python3 -m pip install -e . && \
    python3 -m pip install pytest black ruff pytest-cov mypy

# Production image
FROM base AS prod
WORKDIR /opt/dredge
RUN useradd -m -u 1000 dredge && chown -R dredge:dredge /opt/dredge
USER dredge
ENTRYPOINT ["dredge"]
CMD ["--help"]
