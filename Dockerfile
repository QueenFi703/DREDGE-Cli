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
    && pip install "torch>=2.8.0" --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt \
    && python -c 'import torch; assert torch.version.cuda is None'


# Now copy the actual source and install the dredge package itself
COPY . .
RUN . /opt/venv/bin/activate \
    && pip install --no-cache-dir . && pip check

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

# Compatibility target used by Compose and the CI build matrix.
FROM runtime AS cpu-build

# ---- Dev stage (optional: docker build --target dev) ----
FROM runtime AS dev
RUN . /opt/venv/bin/activate \
    && pip install --no-cache-dir pytest black ruff mypy pytest-cov
CMD ["dredge-server", "--host", "0.0.0.0", "--port", "8001"]

# ---- GPU stage (optional: docker build --target gpu) ----
# Keep the interpreter/OS identical to the builder. PyTorch's CUDA wheels
# include the CUDA runtime; copying this venv into Ubuntu's Python breaks it.
# CUDA 11.8 has no wheels satisfying the project's torch>=2.8 requirement.
FROM runtime AS gpu
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_REQUIRE_CUDA="cuda>=12.8"

RUN . /opt/venv/bin/activate \
    && pip uninstall -y torch \
    && pip install --no-cache-dir "torch>=2.8.0" --index-url https://download.pytorch.org/whl/cu128 \
    && python -c 'import torch; assert torch.version.cuda == "12.8"; assert torch.ones(1).item() == 1' \
    && pip check

# Compatibility target used by Compose and the CI build matrix.
FROM gpu AS gpu-build
