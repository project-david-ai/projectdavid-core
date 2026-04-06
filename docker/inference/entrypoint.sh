#!/bin/bash
set -e

# Start Tailscale in kernel mode — requires NET_ADMIN and /dev/net/tun.
# Cap and device are set in docker-compose.yml (inference-worker service).
if [[ -n "${TAILSCALE_AUTH_KEY:-}" ]]; then
    tailscaled --statedir=/tmp/tailscale &
    sleep 3
    tailscale up \
        --authkey="${TAILSCALE_AUTH_KEY}" \
        --hostname="${NODE_ID:-sovereign-forge-worker}" \
        --accept-routes
    echo "[entrypoint] Tailscale connected: $(tailscale ip -4)"
else
    echo "[entrypoint] TAILSCALE_AUTH_KEY not set — skipping Tailscale."
fi

exec python3 /app/inference_worker.py
