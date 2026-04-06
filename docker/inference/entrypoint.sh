#!/bin/bash

# Start SSH daemon
service ssh start

# Start Tailscale in userspace mode — no NET_ADMIN required.
if [[ -n "${TAILSCALE_AUTH_KEY:-}" ]]; then
    tailscaled --tun=userspace-networking --statedir=/tmp/tailscale &
    sleep 3
    tailscale up \
        --authkey="${TAILSCALE_AUTH_KEY}" \
        --hostname="${NODE_ID:-sovereign-forge-worker}" \
        --accept-routes || true
    echo "[entrypoint] Tailscale connected: $(tailscale ip -4)"
else
    echo "[entrypoint] TAILSCALE_AUTH_KEY not set — skipping Tailscale."
fi

# Start Ray HEAD node with explicit client server port
# This opens port 10001 for worker nodes to connect via ray://
if [[ -z "${RAY_ADDRESS:-}" ]]; then
    echo "[entrypoint] Starting Ray HEAD node on client server port ${RAY_CLIENT_SERVER_PORT:-10001}"
    ray start --head \
        --dashboard-host=0.0.0.0 \
        --dashboard-port="${RAY_DASHBOARD_PORT:-8265}" \
        --ray-client-server-port="${RAY_CLIENT_SERVER_PORT:-10001}" \
        --disable-usage-stats
fi

exec python3 /app/inference_worker.py
