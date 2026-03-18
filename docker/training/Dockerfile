# -------------------------
# MULTI-STAGE DOCKERFILE (TRAINING API)
# -------------------------

# === Stage 0: base layer ============================================
FROM python:3.11-slim AS base

# Set environment variables for pip globally to prevent timeouts and OOMs.
# Putting NO_CACHE_DIR here ensures the builder stage actually uses it!
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=6000 \
    PIP_RETRIES=30

WORKDIR /install

RUN pip install --upgrade pip setuptools wheel

# === Stage 1: install packages (Builder Stage) ========================
FROM base AS builder

# Create a virtual environment. This is MUCH safer for heavy ML
# libraries like PyTorch/CUDA than using --prefix
RUN python -m venv /opt/venv

# Activate the venv for subsequent RUN commands
ENV PATH="/opt/venv/bin:$PATH"

COPY training_reqs_unhashed.txt .

# Install requirements into the venv.
# We explicitly use --no-cache-dir again just to be 100% sure.
RUN pip install --no-cache-dir -r training_reqs_unhashed.txt

# === Stage 2: runtime container =======================================
FROM python:3.11-slim AS runtime
WORKDIR /app

# PYTHONUNBUFFERED=1: Forces python to print logs immediately
# PYTHONDONTWRITEBYTECODE=1: Prevents python from writing .pyc files to disk
# PATH: Activates the virtual environment we copy over
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# --- Copy the fully built virtual environment from the builder stage ---
COPY --from=builder /opt/venv /opt/venv

# --- Copy training app code ---
COPY src/api/training /app/src/api/training

EXPOSE 9001

# --- Default command ---
# Because we added /opt/venv/bin to the PATH above, uvicorn will run directly
CMD ["uvicorn", "src.api.training.app:app", "--host", "0.0.0.0", "--port", "9001"]