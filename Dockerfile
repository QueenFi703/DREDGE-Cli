#syntax=docker/dockerfile:1
#Multi-stage build for smaller image size and faster deployment

# ---- Builder stage ----
FROM python:3.14-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# Build-time system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies into a venv first (better layer caching)
COPY requirements.txt ./
RUN python -m venv /opt/venv \
    && . /opt/venv/bin/activate \
    && pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt


# Now copy the actual source and install the dredge package itself
COPY . .
RUN . /opt/venv/bin/activate \
    && pip install --no-cache-dir -e .

# ---- Runtime stage ----
FROM python:3.14-slim AS runtime

ENV PYTHONPATH="/app/src:${PYTHONPATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8001

WORKDIR /app

# Runtime-only system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Bring in the venv (with all deps + dredge installed) and the source tree
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /build /app

EXPOSE 8001

CMD ["dredge-server", "--host", "0.0.0.0", "--port", "8001"]

# ---- Dev stage (optional: docker build --target dev) ----
FROM runtime AS dev
RUN . /opt/venv/bin/activate \
    && pip install --no-cache-dir pytest black ruff mypy pytest-cov
CMD ["dredge-server", "--host", "0.0.0.0", "--port", "8001"]

# ---- GPU stage (optional: docker build --target gpu) ----
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 AS gpu
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH="/app/src:${PYTHONPATH}" \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8001

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    curl \
    ffmpeg \ 
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /build /app

# Swap in the CUDA build of torch for the GPU inference
RUN . /opt/venv/bin/activate \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu118

EXPOSE 8001
CMD ["dredge-server", "--host", "0.0.0.0", "--port", "8001"]

