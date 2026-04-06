# =============================================================================
# docker/inference/Dockerfile
#
# Sovereign Forge — Inference Worker
# =============================================================================

FROM vllm/vllm-openai:v0.8.5

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install Tailscale + OpenSSH
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    iptables \
    openssh-server \
    && curl -fsSL https://tailscale.com/install.sh | sh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Configure SSH — key-only, no password auth
RUN mkdir -p /var/run/sshd /root/.ssh \
    && chmod 700 /root/.ssh \
    && echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAILOVtSg1Tw4aiWqM9qkrWfxpQwYzXfMptG2+CEgQSWem francis.neequaye@gmail.com" \
       >> /root/.ssh/authorized_keys \
    && chmod 600 /root/.ssh/authorized_keys \
    && echo "PermitRootLogin yes" >> /etc/ssh/sshd_config \
    && echo "PasswordAuthentication no" >> /etc/ssh/sshd_config

RUN pip install --no-cache-dir \
    "ray[serve]==2.40.0" \
    "pymysql>=1.1.0" \
    "sqlalchemy>=2.0.0" \
    "projectdavid-common" \
    "projectdavid-orm" \
    "passlib" \
    "python-dotenv"

COPY src/api/training/inference_worker.py /app/inference_worker.py
COPY src/api/training/models /app/src/api/training/models
COPY src/api/training/db /app/src/api/training/db
COPY docker/inference/entrypoint.sh /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

ENV PYTHONPATH=/app

EXPOSE 22
EXPOSE 8265
EXPOSE 10001
EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
