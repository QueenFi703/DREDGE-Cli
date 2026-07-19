FROM python:3.14-slim AS base
RUN apt-get update && apt-get install -y curl git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python3 -m pip install --no-cache-dir -e .

FROM base AS cpu-build
RUN python3 -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
CMD ["dredge-server", "--host", "0.0.0.0", "--port", "8001"]
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 AS gpu-build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    ca-certificates \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    libsqlite3-dev \
    libbz2-dev \
    liblzma-dev \
    uuid-dev \
    tk-dev \
    && rm -rf /var/lib/apt/lists/*
#Build and install Python 3.14 from source
RUN wget https://www.python.org/ftp/python/3.14.6/Python-3.14.6.tgz \
    && tar xzf Python-3.14.6.tgz \
    && cd Python-3.14.6 \
    && ./configure --enable-optimizations --with-ensurepip=install \
    && make altinstall \
    && cd .. \
    && rm -rf Python-3.14.6 Python-3.14.6.tgz

RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.14 1 && \
    update-alternatives --install /usr/bin/python python /usr/local/bin/python3.14 1
WORKDIR /app
COPY . .
RUN python3 -m pip install --no-cache-dir -r requirements.txt
RUN python3 -m pip install --no-cache-dir -e . && python3 -m pip install --no-cache-dir torch
# Railway.app optimized DREDGE Production Build
# Multi-stage build for smaller image size and faster deployment

FROM python:3.14-slim as builder


ENV PYTHONPATH="/app/src:${PYTHONPATH}"\
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY requirements.txt ./

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Runtime stage
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=3001 \
    FLASK_ENV=production

WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    && apt-get clean \
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "wsgi:app"]
FROM cpu-build AS dev
RUN pip install --no-cache-dir pytest black ruff
CMD ["gunicorn", "wsgi:app"]
