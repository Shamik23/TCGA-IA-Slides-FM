# TCGA Survival Model Training Pipeline
# Multi-stage build: supports both CPU and NVIDIA GPU variants.
# Usage:
#   CPU (default):  docker build -t tcga-survival-fm:latest .
#   GPU:            docker build --build-arg BASE=gpu-base -t tcga-survival-fm:latest .

# Default to CPU; override with --build-arg BASE=gpu-base for CUDA builds.
# Must be declared BEFORE any FROM so it is a global build arg.
ARG BASE=cpu-base

# Named base-image stages so a simple --build-arg can switch between them.
FROM python:3.11-slim AS cpu-base
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime AS gpu-base

# hadolint ignore=DL3006
FROM ${BASE} AS base
ARG BASE

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

# Install PyTorch: GPU image already has it, CPU image needs install
ARG INSTALL_DEV=false
RUN pip install "pip==24.3.1"
RUN if [ "$BASE" = "gpu-base" ]; then \
        echo "Using pre-installed GPU PyTorch from base image"; \
    else \
        pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 torchvision==0.19.1; \
    fi

COPY src/ ./src/
RUN if [ "$INSTALL_DEV" = "true" ]; then pip install -e ".[dev]"; else pip install -e "."; fi

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
