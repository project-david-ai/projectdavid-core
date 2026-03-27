# -------------------------
# MULTI-STAGE DOCKERFILE (SANDBOX)
# -------------------------

# === Stage 0: base_workers layer (common bootstrap) ============================================
FROM python:3.11-slim AS base
WORKDIR /install

RUN pip install --upgrade pip setuptools wheel

# === Stage 1: install UNHASHED packages ===============================================
FROM base AS unhashed_builder

COPY sandbox_reqs_unhashed.txt .

# Add OpenCV headless so cv2 won't look for libGL.so.1
RUN echo "opencv-python-headless" >> sandbox_reqs_unhashed.txt

RUN pip install \
    --prefix=/install/packages \
    --default-timeout=1800 \
    --retries 30 \
    -r sandbox_reqs_unhashed.txt

# === Stage 2: install HASHED stable packages ==========================================
FROM base AS hashed_builder

COPY sandbox_reqs_hashed.txt .

RUN pip install \
    --prefix=/install/packages \
    --require-hashes \
    --default-timeout=600 \
    --retries 10 \
    -r sandbox_reqs_hashed.txt

# === Stage 3: runtime container =======================================================
FROM python:3.11-slim AS sandbox-runtime
WORKDIR /app

ENV PYTHONUNBUFFERED=1

# --- Install minimal OS dependencies ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    firejail \
    libmagic1 \
    supervisor \
    sudo \
    curl \
    wget \
    vim \
    nano \
    less \
    net-tools \
    iproute2 \
    man-db \
    locales \
    htop \
    iputils-ping \
    nmap \
    netcat-openbsd \
    dnsutils \
    tcpdump \
    whois \
    traceroute \
    procps \
    git \
    build-essential \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    libopenblas-dev \
    liblapack-dev \
    libxml2-dev \
    libxslt1-dev \
    libffi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


COPY docker/sandbox/computer_shell_netfilter.conf /app/computer_shell_netfilter.conf

# --- Create a restricted sandbox user ---
RUN useradd -ms /bin/bash sandboxuser && \
    echo 'sandboxuser ALL=(ALL) NOPASSWD: /usr/bin/firejail /bin/bash, /usr/bin/firejail /usr/bin/apt-get *, /usr/bin/firejail /usr/bin/pip' \
      > /etc/sudoers.d/sandboxuser && \
    chmod 440 /etc/sudoers.d/sandboxuser

# --- Prepare app directory ---
RUN mkdir -p /app /var/log/supervisor && \
    chown -R sandboxuser:sandboxuser /app /var/log/supervisor

# --- Copy installed packages from both build stages ---
# NOTE: these two layers are merged into /usr/local sequentially.
# If both stages resolve different versions of the same package, the compiled
# C extensions (.so files) and pure-Python files can end up mismatched.
# The force-reinstall step below fixes any such collisions.
COPY --from=unhashed_builder /install/packages /usr/local
COPY --from=hashed_builder   /install/packages /usr/local

# --- Resolve package version conflicts introduced by the two-stage merge ---
# charset-normalizer is the known offender: the unhashed and hashed stages can
# resolve different versions whose compiled C extensions (.so) and pure-Python
# files end up mismatched after the sequential COPY. Force a clean reinstall
# to get a single coherent wheel.
RUN pip install --force-reinstall --no-cache-dir charset-normalizer

# --- Copy sandbox app code ---
COPY src/api/sandbox /app/sandbox

# --- Utilities ---
COPY docker/common/wait-for-it.sh     /app/wait-for-it.sh
COPY docker/sandbox/supervisord.conf  /etc/supervisor/conf.d/supervisord.conf
RUN chmod +x /app/wait-for-it.sh

# --- Expose sandbox FastAPI server port ---
EXPOSE 8000

# --- Default command ---
CMD ["/usr/bin/supervisord", "-n"]
