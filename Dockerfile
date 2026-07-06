FROM python:3.14-slim AS base

# shared setup
RUN apt-get update && apt-get install -y curl git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e .

# CPU image
FROM base AS cpu-build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
CMD ["dredge-cli", "serve", "--host", "0.0.0.0", "--port", "3001"]

# ────────────────────────────────────────────────────────────────[...]
# Stage 3: GPU Build (Full Dolly + Quasimoto)
# ────────────────────────────────────────────────────────────────[...]
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 AS gpu-build

# Layer 1: System dependencies with immediate cleanup
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.14 \
    python3.14-dev \
    python3-pip \
    gcc \
    g++ \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.14 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.14 1

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e . && pip install --no-cache-dir torch
CMD ["dredge-cli", "mcp", "--host", "0.0.0.0", "--port", "3002"]

# Development image
FROM cpu-build AS dev
RUN pip install --no-cache-dir pytest black ruff
