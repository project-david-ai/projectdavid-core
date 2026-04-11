#!/bin/bash

# Start SSH daemon
service ssh start

# Start Tailscale in userspace mode — no NET_ADMIN required.
if [[ -n "${TAILSCALE_AUTH_KEY:-}" ]]; then
    tailscaled --tun=userspace-networking --statedir=/var/lib/tailscale &
    sleep 3
    tailscale up \
        --authkey="${TAILSCALE_AUTH_KEY}" \
        --hostname="${NODE_ID:-sovereign-forge-worker}" \
        --accept-routes || true

    # Capture Tailscale IP and override NODE_IP regardless of what was passed in.
    # This makes the worker self-configuring — no hardcoded IP needed in the template.
    TAILSCALE_IP=$(tailscale ip -4 2>/dev/null || true)
    if [[ -n "${TAILSCALE_IP}" ]]; then
        export NODE_IP="${TAILSCALE_IP}"
        echo "[entrypoint] Tailscale connected: ${NODE_IP}"
    else
        echo "[entrypoint] Tailscale connected but could not detect IP — NODE_IP unchanged"
    fi
else
    echo "[entrypoint] TAILSCALE_AUTH_KEY not set — skipping Tailscale."
fi

# HuggingFace login if token provided
if [[ -n "${HF_TOKEN:-}" ]]; then
    huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential || true
    echo "[entrypoint] HuggingFace authenticated."
else
    echo "[entrypoint] HF_TOKEN not set — public models only."
fi

# Ray startup — HEAD or WORKER depending on RAY_ADDRESS
if [[ -z "${RAY_ADDRESS:-}" ]]; then
    echo "[entrypoint] Starting Ray HEAD node on client server port ${RAY_CLIENT_SERVER_PORT:-10001}"
    ray start --head \
        --node-ip-address="${NODE_IP:-0.0.0.0}" \
        --dashboard-host=0.0.0.0 \
        --dashboard-port="${RAY_DASHBOARD_PORT:-8265}" \
        --ray-client-server-port="${RAY_CLIENT_SERVER_PORT:-10001}" \
        --disable-usage-stats
else
    # Strip ray:// prefix and force port 6379 for GCS worker join
    RAY_HEAD_GCS="${RAY_ADDRESS#ray://}"
    RAY_HEAD_GCS="${RAY_HEAD_GCS%:*}:6379"
    echo "[entrypoint] Joining Ray cluster as WORKER node at ${RAY_HEAD_GCS}"
    ray start \
        --address="${RAY_HEAD_GCS}" \
        --node-ip-address="${NODE_IP:-}" \
        --disable-usage-stats
    echo "[entrypoint] Ray worker node started."
fi

exec python3 /app/inference_worker.py
