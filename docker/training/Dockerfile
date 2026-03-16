# docker/training/Dockerfile
#
# Self-hosted fine-tuning container for projectdavid-core.
#
# Supports two training frameworks selectable at runtime via --framework:
#   axolotl  — full-featured, YAML-driven, supports chatml/alpaca/sharegpt/jsonl
#   unsloth  — fast LoRA, lower VRAM requirement, good for consumer GPUs
#
# Build:
#   docker build -t projectdavid/training:latest -f docker/training/Dockerfile .
#
# Runtime requirements:
#   - NVIDIA GPU with CUDA 12.1+ drivers
#   - nvidia-container-toolkit on the host
#   - ${SHARED_PATH} mounted at /mnt/training_data
#
# Directory contract (all paths relative to /mnt/training_data):
#   datasets/{user_id}/{dataset_id}/     ← input dataset files (read)
#   checkpoints/{job_id}/                ← training output (write)
#   configs/{job_id}/config.yml          ← training config written by API (read)
#   .hf_cache/                           ← HF model cache (read/write, persists between jobs)

FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    curl \
    wget \
    build-essential \
    libssl-dev \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip setuptools wheel

# ─── Axolotl ─────────────────────────────────────────────────────────────────
RUN pip install \
    "axolotl[flash-attn] @ git+https://github.com/OpenAccess-AI-Collective/axolotl.git@v0.4.1" \
    || pip install "axolotl @ git+https://github.com/OpenAccess-AI-Collective/axolotl.git@v0.4.1"

# ─── Unsloth ─────────────────────────────────────────────────────────────────
# Install fails gracefully — axolotl is always available as fallback.
RUN pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" \
    || echo "[training] Unsloth install failed — axolotl-only mode active"

# ─── Python dependencies ─────────────────────────────────────────────────────
COPY training_reqs_unhashed.txt /app/training_reqs_unhashed.txt
RUN pip install -r /app/training_reqs_unhashed.txt

# Copy training scripts
COPY docker/training/entrypoint.sh /app/entrypoint.sh
COPY docker/training/unsloth_train.py /app/unsloth_train.py
COPY docker/training/worker.py /app/worker.py
RUN chmod +x /app/entrypoint.sh

# Mount point — populated at runtime via ${SHARED_PATH} bind mount
RUN mkdir -p /mnt/training_data

ENTRYPOINT ["/app/entrypoint.sh"]