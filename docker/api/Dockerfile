# -------------------------
# MULTI-STAGE DOCKERFILE
# -------------------------

# ⭐ GLOBAL DEFAULT → CPU TORCH
ARG TORCH_CHANNEL=cpu


# === Stage 0: base layer ===============================================================
FROM python:3.11-slim-bookworm AS base
WORKDIR /install

ARG TORCH_CHANNEL
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/${TORCH_CHANNEL}

RUN pip install --upgrade pip setuptools wheel


# === Stage 1: install UNHASHED packages ===============================================
FROM base AS unhashed_builder
ARG TORCH_CHANNEL

COPY api_unhashed_reqs.txt .

RUN echo "opencv-python-headless" >> api_unhashed_reqs.txt

RUN pip install \
    --prefix=/install/packages \
    --no-cache-dir \
    --default-timeout=100 \
    --retries 10 \
    -r api_unhashed_reqs.txt


# === Stage 2: install HASHED stable trusted packages ===================================
FROM base AS hashed_builder
ARG TORCH_CHANNEL

COPY api_reqs_hashed.txt .

RUN pip install \
    --prefix=/install/packages \
    --require-hashes \
    --no-cache-dir \
    --default-timeout=100 \
    --retries 10 \
    -r api_reqs_hashed.txt


# === Stage 3: runtime container ========================================================
FROM python:3.11-slim-bookworm AS runtime
WORKDIR /app

RUN rm -f /etc/apt/sources.list.d/*.sources && \
    printf "deb http://deb.debian.org/debian bookworm main contrib non-free non-free-firmware\n\
deb http://deb.debian.org/debian bookworm-updates main contrib non-free non-free-firmware\n\
deb http://security.debian.org/debian-security bookworm-security main contrib non-free non-free-firmware\n" \
    > /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libmagic1 \
        firejail \
        supervisor \
        wget \
        vim \
        nano \
        git \
        dos2unix \
        ca-certificates \
        curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


COPY --from=unhashed_builder /install/packages /usr/local
COPY --from=hashed_builder   /install/packages /usr/local

RUN pip install --force-reinstall --no-cache-dir charset-normalizer


RUN pip install --no-cache-dir html2text playwright && \
    playwright install --with-deps chromium


COPY src /app/src
COPY docker/api/init_and_run_api.sh /app/init_and_run_api.sh
COPY docker/api/supervisord.conf    /etc/supervisor/conf.d/supervisord.conf

RUN dos2unix /app/init_and_run_api.sh && chmod +x /app/init_and_run_api.sh


ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src/api

EXPOSE 9000

ENTRYPOINT ["/app/init_and_run_api.sh"]
CMD ["/usr/bin/supervisord", "-n"]