#!/bin/bash

# Start SSH daemon
service ssh start

# Start Tailscale in userspace mode — no NET_ADMIN required.
# Works on RunPod, AWS, Azure, any unprivileged container.
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

exec python3 /app/inference_worker.py
