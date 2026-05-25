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

# GPU image
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS gpu-build
RUN apt-get update && apt-get install -y python3.10 python3-pip curl git && rm -rf /var/lib/apt/lists/*
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e . && pip install --no-cache-dir torch
CMD ["dredge-cli", "mcp", "--host", "0.0.0.0", "--port", "3002"]

# Development image
FROM cpu-build AS dev
RUN pip install --no-cache-dir pytest black ruff
