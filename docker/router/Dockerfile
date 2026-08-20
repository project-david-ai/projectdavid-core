# ── Stage 1: build ────────────────────────────────────────────────────────────
FROM rust:1.88-slim-bookworm AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        pkg-config \
        libssl-dev \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY rust/request_router/ .

# Expose toolchain version in CI logs and build using the committed lockfile.
RUN rustc --version && cargo --version

RUN cargo build --locked --release --bin pd_router


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
# Minimal image — just the binary, CA certificates, and curl for the healthcheck.
FROM debian:bookworm-slim AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/pd_router /usr/local/bin/pd_router

# Upstream FastAPI URL — overridable at runtime.
ENV UPSTREAM_URL=http://api:9000

# Port the router listens on.
ENV LISTEN_PORT=9100

# Tracing level.
ENV LOG_LEVEL=pd_router=info

EXPOSE 9100

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9100/_router/health || exit 1

ENTRYPOINT ["/usr/local/bin/pd_router"]
