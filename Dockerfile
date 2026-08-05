FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# Keeps HuggingFace model cache outside the project dir
ENV HF_HOME=/tmp/hf_cache
ENV TRANSFORMERS_CACHE=/tmp/hf_cache

# System deps: Python 3.11 + build tools for fasttext / bitsandbytes
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-distutils \
    python3-pip git curl build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Install PyTorch with CUDA 12.1 first so the requirements install below
# doesn't overwrite it with a CPU-only build from PyPI.
RUN pip install --no-cache-dir \
    "torch==2.3.1" "torchvision==0.18.1" "torchaudio==2.3.1" \
    --index-url https://download.pytorch.org/whl/cu121

# Install the rest of requirements, skipping:
#   - torch* (already installed above)
#   - pkld   (internal package, not on PyPI)
COPY requirements.txt /tmp/requirements.txt
RUN grep -vE "^(torch|pkld)" /tmp/requirements.txt \
    | pip install --no-cache-dir -r /dev/stdin

# GCS library for artifact upload in vertex_job_runner.py
RUN pip install --no-cache-dir google-cloud-storage

# Copy the project (respects .dockerignore)
COPY . /app
WORKDIR /app

# Make the src package importable
ENV PYTHONPATH=/app

# Default entrypoint is the Vertex job runner; override for interactive use.
ENTRYPOINT ["python", "scripts/vertex/vertex_job_runner.py"]
