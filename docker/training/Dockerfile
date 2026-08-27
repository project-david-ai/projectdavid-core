# -------------------------
# MULTI-STAGE DOCKERFILE (ProjectDavid Training Stack)
# -------------------------
#
# This image is shared by training-api and training-worker.
# training-worker runs unsloth fine-tuning jobs as direct subprocesses.
# Ray is NOT used here — inference_worker owns the Ray cluster.
# -------------------------

# === Stage 0: base layer ============================================
FROM python:3.12-slim AS base

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=6000 \
    PIP_RETRIES=30 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /install

RUN pip install --upgrade pip setuptools wheel


# === Stage 1: install ALL packages ==================================
# Heavy ML deps (torch, unsloth, xformers, bitsandbytes) are pinned by
# index URL and cannot carry hashes — they live in training_reqs_unhashed.txt.
# Pure-Python infrastructure deps live in training_reqs_hashed.txt.
FROM base AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel

COPY training_reqs_unhashed.txt .
RUN pip install \
    --prefix=/install/packages \
    --default-timeout=6000 \
    --retries=30 \
    -r training_reqs_unhashed.txt

COPY training_reqs_hashed.txt .
RUN pip install \
    --prefix=/install/packages \
    --require-hashes \
    --default-timeout=600 \
    --retries=10 \
    -r training_reqs_hashed.txt


# === Stage 2: runtime container =====================================
FROM python:3.12-slim AS runtime

WORKDIR /app

# gcc/g++ MUST be present at runtime — Triton compiles CUDA kernels
# on first inference, not at build time.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    g++ \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY --from=builder /install/packages /usr/local

# Resolve known version-collision packages after merge.
RUN pip install --force-reinstall --no-cache-dir charset-normalizer packaging

ENV PATH="/usr/local/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# training-api and training-worker both use this image.
#
# Copy through a fresh staging location rather than directly into /app.
# The explicit integrity checks prevent a corrupt Docker build/cache layer
# from silently publishing an empty training application or worker entrypoint.
COPY src/api/training /tmp/projectdavid-training-src

RUN test -s /tmp/projectdavid-training-src/app.py \
    && test -s /tmp/projectdavid-training-src/worker.py \
    && mkdir -p /app/src/api/training \
    && cp -R /tmp/projectdavid-training-src/. /app/src/api/training/ \
    && test -s /app/src/api/training/app.py \
    && test -s /app/src/api/training/worker.py \
    && rm -rf /tmp/projectdavid-training-src

# training-worker overrides CMD via docker-compose command:
#   command: ["python", "src/api/training/worker.py"]

# training-api port
EXPOSE 9001

# Default: training-api
CMD ["python", "-m", "uvicorn", "src.api.training.app:app", "--host", "0.0.0.0", "--port", "9001"]
