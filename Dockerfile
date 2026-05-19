FROM python:3.12-slim AS base

# shared setup
RUN apt-get update && apt-get install -y curl

# CPU image
FROM base AS cpu-build
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# GPU image
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS gpu-build
RUN pip install torch

# Development image
FROM cpu-build AS dev
RUN pip install pytest black ruff
