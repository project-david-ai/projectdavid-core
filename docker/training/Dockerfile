# docker/training/Dockerfile.api
#
# Training Service — FastAPI HTTP layer.
# Lightweight, no GPU, no ML frameworks.
# Serves the fine-tuning REST API on port 9001.
#
# Build:
#   docker build -t projectdavid/training-api:latest -f docker/training/Dockerfile.api .
#
# The GPU training runner (worker.py + axolotl/unsloth) lives in Dockerfile.

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip setuptools wheel

# API dependencies only — no torch, no axolotl, no unsloth
COPY training_api_reqs.txt /app/training_api_reqs.txt
RUN pip install -r /app/training_api_reqs.txt

# Application source — training package only
COPY src/api/training /app/src/api/training

# Required so Python resolves src.api.training as a package
COPY src/__init__.py /app/src/__init__.py
COPY src/api/__init__.py /app/src/api/__init__.py

EXPOSE 9001

CMD ["uvicorn", "src.api.training.app:app", "--host", "0.0.0.0", "--port", "9001"]