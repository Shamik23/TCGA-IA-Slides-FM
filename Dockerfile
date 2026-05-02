# TCGA Survival Model Training Pipeline
# Multi-stage build: slim Python + CPU PyTorch (works on x86_64 and arm64).
# For NVIDIA GPUs, swap the base image to pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface

# System deps required by Pillow / OpenCV-style image stacks
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libjpeg-dev \
        zlib1g-dev \
        libpng-dev \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first for better layer caching
COPY pyproject.toml ./
COPY README.md ./
COPY src/ ./src/

# CPU-only torch wheels keep the image small; override at build time if needed.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --upgrade pip \
    && pip install --index-url ${TORCH_INDEX_URL} torch==2.4.1 torchvision==0.19.1 \
    && pip install -e ".[dev]"

# Application code
COPY scripts/ ./scripts/
COPY tests/ ./tests/

# Pre-create runtime directories
RUN mkdir -p /app/data /app/runs ${HF_HOME}

# Run as a non-root user for safety
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["tcga-survival"]
CMD ["--help"]
