# -------------------------
# MULTI-STAGE DOCKERFILE (TRAINING API)
# -------------------------

# === Stage 0: base layer ============================================
FROM python:3.11-slim AS base
WORKDIR /install

RUN pip install --upgrade pip setuptools wheel

# === Stage 1: install UNHASHED packages ===============================================
FROM base AS builder

COPY training_reqs_unhashed.txt .

RUN pip install \
    --prefix=/install/packages \
    --default-timeout=1800 \
    --retries 30 \
    -r training_reqs_unhashed.txt

# === Stage 2: runtime container =======================================================
FROM python:3.11-slim AS runtime
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# --- Copy installed packages from the builder stage ---
COPY --from=builder /install/packages /usr/local

# --- Copy training app code ---
COPY src/api/training /app/src/api/training

EXPOSE 9001

# --- Default command ---
CMD ["uvicorn", "src.api.training.app:app", "--host", "0.0.0.0", "--port", "9001"]
