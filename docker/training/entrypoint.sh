#!/bin/bash
# docker/training/entrypoint.sh
#
# Entrypoint for the projectdavid training container.
#
# Called by TrainingContainerManager with:
#   --framework axolotl|unsloth
#   --config    /mnt/training_data/configs/{job_id}/config.yml
#
# Exit codes:
#   0  — training completed successfully
#   1  — configuration or argument error
#   2  — training process failed
#   3  — unsupported framework

set -euo pipefail

FRAMEWORK="axolotl"
CONFIG_PATH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --framework) FRAMEWORK="$2"; shift 2 ;;
        --config)    CONFIG_PATH="$2"; shift 2 ;;
        *) echo "[training] Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ─── Validation ──────────────────────────────────────────────────────────────
if [[ -z "$CONFIG_PATH" ]]; then
    echo "[training] ERROR: --config is required" >&2; exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "[training] ERROR: config file not found: $CONFIG_PATH" >&2; exit 1
fi

if [[ "$FRAMEWORK" != "axolotl" && "$FRAMEWORK" != "unsloth" ]]; then
    echo "[training] ERROR: unsupported framework '$FRAMEWORK'. Use axolotl or unsloth." >&2
    exit 3
fi

# ─── HuggingFace cache ───────────────────────────────────────────────────────
# Persisted at /mnt/training_data/.hf_cache — shared with all jobs so base
# model weights are only downloaded once.
export HF_HOME="${HF_HOME:-/mnt/training_data/.hf_cache}"
mkdir -p "$HF_HOME"

if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "[training] HF_TOKEN present — logging in for gated model access."
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || true
fi

# ─── GPU check ───────────────────────────────────────────────────────────────
echo "[training] Checking GPU availability..."
python -c "
import torch
gpus = torch.cuda.device_count()
print(f'[training] {gpus} GPU(s) available')
assert gpus > 0, 'No GPU detected — training requires NVIDIA GPU'
" || { echo "[training] ERROR: No CUDA GPU available." >&2; exit 1; }

# ─── Summary ─────────────────────────────────────────────────────────────────
echo "[training] ========================================"
echo "[training] Framework : $FRAMEWORK"
echo "[training] Config    : $CONFIG_PATH"
echo "[training] HF cache  : $HF_HOME"
echo "[training] Started   : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[training] ========================================"

# ─── Run ─────────────────────────────────────────────────────────────────────
case "$FRAMEWORK" in
    axolotl)
        python -m axolotl.cli.train "$CONFIG_PATH"
        EXIT_CODE=$?
        ;;
    unsloth)
        python /app/unsloth_train.py "$CONFIG_PATH"
        EXIT_CODE=$?
        ;;
esac

# ─── Result ──────────────────────────────────────────────────────────────────
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "[training] Completed successfully at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    exit 0
else
    echo "[training] FAILED with exit code $EXIT_CODE at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >&2
    exit 2
fi
