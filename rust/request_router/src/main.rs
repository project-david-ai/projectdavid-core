//! pd_router — Project David inference request router
//!
//! Sits between nginx and FastAPI. Owns the connection pool to the upstream
//! API and handles SSE streaming responses without buffering, allowing
//! concurrent inference requests to stream simultaneously without GIL
//! contention on the FastAPI side.
//!
//! Configuration (environment variables):
//!   UPSTREAM_URL   — FastAPI base URL (default: http://api:9000)
//!   LISTEN_PORT    — Port to bind on (default: 9100)
//!   LOG_LEVEL      — tracing filter string (default: pd_router=info)
//!
//! All requests are proxied transparently. Hop-by-hop headers are stripped.
//! SSE responses (Content-Type: text/event-stream) are streamed without
//! buffering. All other responses are buffered and forwarded.

use axum::{
    body::Body,
    extract::{Request, State},
    http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode},
    response::Response,
    routing::get,
    Router,
};
use bytes::Bytes;
use futures_util::StreamExt;
use reqwest::Client;
use std::{net::SocketAddr, sync::Arc, time::Instant};
use tokio::net::TcpListener;
use tracing::{error, info, warn};

// ── Application state ─────────────────────────────────────────────────────────

#[derive(Clone)]
struct RouterState {
    /// Shared HTTP client with connection pool to the upstream FastAPI.
    client: Arc<Client>,
    /// Base URL of the upstream (e.g. "http://api:9000").
    upstream: Arc<String>,
}

// ── Hop-by-hop header filter ──────────────────────────────────────────────────

/// Returns true if the header should NOT be forwarded to/from upstream.
/// These are connection-scoped headers that must not cross proxy boundaries.
fn is_hop_by_hop(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "connection"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailers"
            | "transfer-encoding"
            | "upgrade"
    )
}

// ── Health check ──────────────────────────────────────────────────────────────

async fn health() -> &'static str {
    "ok"
}

// ── Proxy handler ─────────────────────────────────────────────────────────────

async fn proxy(
    State(state): State<RouterState>,
    req: Request,
) -> Result<Response, StatusCode> {
    let start = Instant::now();
    let method = req.method().clone();
    let path_and_query = req
        .uri()
        .path_and_query()
        .map(|pq| pq.as_str())
        .unwrap_or("/")
        .to_string();

    let upstream_url = format!("{}{}", state.upstream, path_and_query);

    // ── Build upstream request ────────────────────────────────────────────────

    let mut upstream_req = state.client.request(
        reqwest::Method::from_bytes(method.as_str().as_bytes()).unwrap_or(reqwest::Method::GET),
        &upstream_url,
    );

    // Forward all client headers except hop-by-hop.
    for (name, value) in req.headers() {
        if !is_hop_by_hop(name.as_str()) {
            if let Ok(v) = reqwest::header::HeaderValue::from_bytes(value.as_bytes()) {
                upstream_req = upstream_req.header(name.as_str(), v);
            }
        }
    }

    // Collect and forward the request body.
    let body_bytes = match axum::body::to_bytes(req.into_body(), 100 * 1024 * 1024).await {
        Ok(b) => b,
        Err(e) => {
            error!("Failed to read request body: {}", e);
            return Err(StatusCode::BAD_REQUEST);
        }
    };

    if !body_bytes.is_empty() {
        upstream_req = upstream_req.body(body_bytes);
    }

    // ── Send to upstream ──────────────────────────────────────────────────────

    let upstream_resp = match upstream_req.send().await {
        Ok(r) => r,
        Err(e) => {
            error!("Upstream {} {} failed: {}", method, path_and_query, e);
            return Err(StatusCode::BAD_GATEWAY);
        }
    };

    let status = StatusCode::from_u16(upstream_resp.status().as_u16())
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

    let resp_headers = upstream_resp.headers().clone();

    // Detect SSE — check before consuming body.
    let is_sse = resp_headers
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|ct| ct.contains("text/event-stream"))
        .unwrap_or(false);

    // ── Build response headers ────────────────────────────────────────────────

    let mut resp_builder = Response::builder().status(status);

    for (name, value) in &resp_headers {
        if !is_hop_by_hop(name.as_str()) {
            if let (Ok(n), Ok(v)) = (
                HeaderName::from_bytes(name.as_ref()),
                HeaderValue::from_bytes(value.as_bytes()),
            ) {
                resp_builder = resp_builder.header(n, v);
            }
        }
    }

    // Always disable upstream buffering for SSE.
    if is_sse {
        resp_builder = resp_builder
            .header("X-Accel-Buffering", "no")
            .header("Cache-Control", "no-cache")
            .header("Connection", "keep-alive");
    }

    // ── Route response body ───────────────────────────────────────────────────

    let response = if is_sse {
        // Stream mode: pipe bytes to client as they arrive.
        // This is the hot path — no buffering, no GIL, pure I/O.
        let byte_stream = upstream_resp.bytes_stream().map(|result| {
            result.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
        });

        resp_builder
            .body(Body::from_stream(byte_stream))
            .map_err(|e| {
                error!("Failed to build streaming response: {}", e);
                StatusCode::INTERNAL_SERVER_ERROR
            })?
    } else {
        // Buffered mode: read full body then forward.
        let body_bytes = upstream_resp.bytes().await.map_err(|e| {
            error!("Failed to read upstream response body: {}", e);
            StatusCode::BAD_GATEWAY
        })?;

        resp_builder.body(Body::from(body_bytes)).map_err(|e| {
            error!("Failed to build buffered response: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?
    };

    let elapsed = start.elapsed();
    info!(
        "{} {} → {} ({:.1}ms) sse={}",
        method,
        path_and_query,
        status,
        elapsed.as_secs_f64() * 1000.0,
        is_sse,
    );

    Ok(response)
}

// ── Entry point ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    // Initialise structured logging.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("pd_router=info")),
        )
        .with_target(false)
        .compact()
        .init();

    let upstream = std::env::var("UPSTREAM_URL")
        .unwrap_or_else(|_| "http://api:9000".to_string());

    let port: u16 = std::env::var("LISTEN_PORT")
        .unwrap_or_else(|_| "9100".to_string())
        .parse()
        .unwrap_or(9100);

    // Connection pool — tuned for high-concurrency inference workloads.
    // Large pool size accommodates many simultaneous SSE streams.
    let client = match Client::builder()
        .pool_max_idle_per_host(100)
        .pool_idle_timeout(std::time::Duration::from_secs(90))
        .tcp_keepalive(std::time::Duration::from_secs(60))
        .tcp_nodelay(true)
        .timeout(std::time::Duration::from_secs(600)) // Long timeout for streaming
        .build()
    {
        Ok(c) => Arc::new(c),
        Err(e) => {
            error!("Failed to build HTTP client: {}", e);
            std::process::exit(1);
        }
    };

    let state = RouterState {
        client,
        upstream: Arc::new(upstream.clone()),
    };

    let app = Router::new()
        // Dedicated health check — nginx and Docker health checks hit this.
        .route("/_router/health", get(health))
        // Everything else is proxied transparently to FastAPI.
        .fallback(proxy)
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));

    info!(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    );
    info!("  Project David — pd_router");
    info!("  Listening on  : {}", addr);
    info!("  Upstream      : {}", upstream);
    info!("  Connection pool: 100 idle/host, keepalive 60s");
    info!(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    );

    let listener = match TcpListener::bind(addr).await {
        Ok(l) => l,
        Err(e) => {
            error!("Failed to bind to {}: {}", addr, e);
            std::process::exit(1);
        }
    };

    if let Err(e) = axum::serve(listener, app).await {
        error!("Server error: {}", e);
        std::process::exit(1);
    }
}
