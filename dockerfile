FROM python:3.14-slim AS base
WORKDIR /app
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y curl git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -e .
FROM base AS cpu-build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
CMD ["dredge-cli", "serve", "--host", "0.0.0.0", "--port", "3001"]
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 AS gpu-build
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.14 \
   python3.14-dev \
    python3-pip \
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.14 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.14 1
COPY . .
RUN pip install --no cache-dir -r requirements.txt
RUN--no-cache-dir -e . && pip install --no-cache-dir torch
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "wsgi:app"]
FROM cpu-build AS dev
RUN pip install --no-cache-dir pytest black ruff
CMD ["gunicorn", "wsgi:app"]
