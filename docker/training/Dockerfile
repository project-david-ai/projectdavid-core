# -------------------------
# MULTI-STAGE DOCKERFILE (ProjectDavid Training Stack)
# -------------------------

# === Stage 0: base layer ============================================
FROM python:3.11-slim AS base

# Set environment variables for pip globally
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=6000 \
    PIP_RETRIES=30 \
    # Prevent python from buffering logs (crucial for worker visibility)
    PYTHONUNBUFFERED=1 \
    # Prevent python from writing .pyc files
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# === Stage 1: install packages (Builder Stage) ========================
FROM base AS builder

# Install system build tools needed to compile heavy ML dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment for isolation
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel

# Copy requirements and install
COPY training_reqs_unhashed.txt .
RUN pip install --no-cache-dir -r training_reqs_unhashed.txt

# === Stage 2: runtime container =======================================
FROM python:3.11-slim AS runtime
WORKDIR /app

# Required for Triton JIT compilation and basic networking
# We MUST have gcc/g++ here because Triton compiles kernels at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    g++ \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy the fully built virtual environment
COPY --from=builder /opt/venv /opt/venv

# Ensure environment uses the venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app

# --- Copy source code ---
# Note: training-api and training-worker both use this same image
COPY src/api/training /app/src/api/training

# Expose port for the training-api (worker doesn't need an exposed port)
EXPOSE 9001

# --- Default command (for training-api) ---
# The training-worker overrides this in docker-compose.yml
CMD ["uvicorn", "src.api.training.app:app", "--host", "0.0.0.0", "--port", "9001"]