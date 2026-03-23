# -------------------------
# MULTI-STAGE DOCKERFILE (ProjectDavid Training Stack)
# -------------------------

# === Stage 0: base layer ============================================
FROM python:3.11-slim AS base

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=6000 \
    PIP_RETRIES=30 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /install

RUN pip install --upgrade pip setuptools wheel

# === Stage 1: install UNHASHED packages =============================
# Heavy ML deps (torch, unsloth, xformers, bitsandbytes) are pinned by
# index URL and cannot carry hashes — they live here.
FROM base AS unhashed_builder

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

# === Stage 2: install HASHED stable packages ========================
# Pure-Python infrastructure deps (redis, fastapi, projectdavid-*,
# ray, etc.) — fully hash-verified for supply-chain integrity.
FROM base AS hashed_builder

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel

COPY training_reqs_hashed.txt .
RUN pip install \
    --prefix=/install/packages \
    --require-hashes \
    --default-timeout=600 \
    --retries=10 \
    -r training_reqs_hashed.txt

# === Stage 3: runtime container =====================================
FROM python:3.11-slim AS runtime
WORKDIR /app

# Required for Triton JIT compilation and basic networking.
# gcc/g++ MUST be present at runtime — Triton compiles CUDA kernels
# on first inference, not at build time.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    g++ \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# --- Merge both build stages into /usr/local -------------------------
# Unhashed (ML stack) first, then hashed (infra) on top.
# If both stages resolve different versions of the same package the
# compiled C extensions (.so) and pure-Python files can mismatch after
# the sequential COPY — the force-reinstall step below resolves this.
COPY --from=unhashed_builder /install/packages /usr/local
COPY --from=hashed_builder   /install/packages /usr/local

# --- Resolve known version-collision packages ------------------------
# charset-normalizer: unhashed and hashed stages can pull different
# minor versions whose .so and .py files end up mismatched after merge.
# packaging: required by ray.widgets at import time.
RUN pip install --force-reinstall --no-cache-dir charset-normalizer packaging

ENV PATH="/usr/local/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# --- Copy source code ------------------------------------------------
# training-api and training-worker both use this same image.
# The worker overrides CMD via docker-compose command:.
COPY src/api/training /app/src/api/training

# Expose port for training-api (worker ignores this)
EXPOSE 9001

# Ray dashboard port (only meaningful when worker starts Ray head node)
EXPOSE 8265

# --- Default command (training-api) ----------------------------------
# Uses python -m uvicorn instead of the uvicorn script directly.
# The uvicorn entrypoint script carries a shebang pointing to
# /opt/venv/bin/python from the builder stage which does not exist
# in the runtime container — invoking via python -m bypasses this.
# training-worker overrides this in docker-compose.yml:
#   command: ["python", "src/api/training/worker.py"]
CMD ["python", "-m", "uvicorn", "src.api.training.app:app", "--host", "0.0.0.0", "--port", "9001"]