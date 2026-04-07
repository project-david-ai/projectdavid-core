# Security Policy

## Supported Versions

Security updates are provided for the latest stable release only. Ensure you are running the latest version before reporting a vulnerability.

| Version | Supported |
|---|---|
| >= 1.27.0 | ✅ |
| < 1.27.0 | ❌ |

---

## Reporting a Vulnerability

**Please do not open a public GitHub Issue for security vulnerabilities.**

If you discover a security-related issue in Project David Core, report it privately. All reports are handled with high priority by the maintainer.

**Email:** [engineering@projectdavid.co.uk](mailto:engineering@projectdavid.co.uk)

Include:
- A summary of the issue
- A proof of concept if available
- The version and component affected (API, Sandbox, Inference Worker, Training Pipeline)

**Acknowledgment:** within 48 hours
**Resolution:** coordinated fix and release before public disclosure

We ask that you do not disclose the issue publicly until a patched version has been released.

---

## Security Architecture

Project David Core is designed for deployment in security-sensitive, air-gapped, and sovereignty-constrained environments. The following documents the current security posture of each major component.

### Reverse Proxy and Rate Limiting

All inbound traffic passes through nginx before reaching any application service. nginx enforces:

- **Rate limiting** — 300 requests per minute per IP address, with a burst allowance of 50 requests (`limit_req_zone` on `$binary_remote_addr`). Requests exceeding the burst are rejected immediately (`nodelay`).
- **Request size limits** — 100 MB maximum body size on the core API. Training endpoints that accept dataset uploads allow up to 500 MB.
- **Upstream retry** — failed upstream connections retry once against the same upstream before returning a 502/503/504, preventing stale DNS entries after container restarts from causing permanent failures.
- **SSE / streaming** — `X-Accel-Buffering: no` is set on all proxied responses, disabling buffering at upstream CDN and load balancer layers for correct streaming behaviour.
- **HTTPS** — a TLS server block is included in the nginx configuration as a documented placeholder. Operators should enable it with their own certificates before exposing the platform to untrusted networks. The provided cipher configuration enforces TLSv1.2 and TLSv1.3 with strong ciphers.

The Ray dashboard (port 8265) and Ray client server (port 10001) are not exposed through nginx and are not accessible from outside the Docker network by default.

### API Key Authentication

Every API endpoint requires a valid API key passed in the request. Keys are validated via a FastAPI dependency (`get_api_key`) injected at the router level — there is no unauthenticated surface on the core API. The validated key object is passed directly into every route handler, making the authenticated user identity available throughout the request lifecycle.

API keys are stored hashed in the database. The plain key is returned exactly once at creation time and is never stored or logged. Keys can be scoped, named, and revoked independently.

**Admin vs user scoping** — a subset of endpoints (user provisioning, admin operations, model registry management) require the authenticated key's owner to hold admin status. This is checked via `_is_admin()` against the database on each request — there is no ambient admin session. Regular user keys are rejected with HTTP 403 on admin-scoped routes.

**Self-service vs cross-user access** — API key management endpoints enforce that the authenticated key belongs to the requested user, or that the key owner is an admin. A user cannot read, create, or revoke API keys belonging to another user.

**Key rotation** — keys can be revoked individually by prefix without affecting other keys for the same user. Revoked keys are soft-deleted (`is_active=False`) and rejected on all subsequent requests.

### WebSocket Authentication

WebSocket endpoints require a short-lived signed JWT issued by the main API. The JWT is validated before the WebSocket connection is accepted — unauthenticated connections are closed with `WS_1008_POLICY_VIOLATION` before any data exchange occurs. Room access is enforced at the token level: a JWT issued for room A cannot be used to join room B. User identity is taken exclusively from the verified JWT payload — any `user_id` supplied in the message body is ignored.

### Sandbox — Code Interpreter

The code interpreter executes user-submitted Python in a firejail sandbox with the following controls active by default:

- `--private=<session_dir>` — process HOME is the session working directory
- `--caps.drop=all` — all Linux capabilities dropped
- `--seccomp` — system call filtering
- `--nogroups` — supplementary group memberships stripped
- `--nosound`, `--notv` — device access blocked

A static blocklist rejects submissions containing `__import__`, `exec`, `eval`, `subprocess`, `os.system`, `shutil.rmtree`, and related patterns before execution. Syntax is validated with `ast.parse` before the process is spawned. Temporary files are written to an isolated directory and cleaned up after each execution.

`DISABLE_FIREJAIL=true` disables sandboxing for local development. This must never be set in production.

### Sandbox — Computer Shell

The persistent shell uses firejail with per-process network namespace isolation:

- A new network namespace is created per shell session (`--net=eth0`)
- iptables netfilter rules are applied inside that namespace:
  - Loopback allowed
  - Outbound DNS allowed
  - RFC-1918 ranges blocked (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16) — Docker-internal services unreachable from the shell
  - Public internet allowed (pip, curl, wget, external APIs all work)
- At most one PTY process is alive per room at any time — the `RoomManager` tears down stale sessions before registering new ones, preventing double-broadcast races and PTY descriptor leaks
- Sessions auto-destruct after 5 minutes of inactivity
- Files generated during a session are harvested and uploaded to the file server on session end, then the session directory is wiped

`COMPUTER_SHELL_ALLOW_NET=true` bypasses the netfilter rules entirely. This must never be set in production.

### Inference Worker

The inference worker runs vLLM through Ray Serve. Each model deployment is isolated as a separate Ray Serve application. The worker container runs an OpenSSH daemon to support SSH tunnel connectivity from the HEAD node for multi-node cluster deployments. Key-based authentication is enforced; password authentication is disabled.

`PermitRootLogin yes` is required for RunPod and similar cloud GPU providers. Operators running inference workers on controlled infrastructure should create a dedicated non-root user and set `PermitRootLogin no`.

### Training Pipeline

Training jobs are dispatched via Redis queue. The training worker consumes jobs and executes them as subprocess calls within the container. `HF_HUB_OFFLINE=1` can be set to prevent any outbound HuggingFace requests during training — required for fully air-gapped deployments.

### Secret Management

All platform secrets are generated locally at first run using Python's `secrets` module. No secrets are hardcoded in source code or container images. Secrets are stored in a `.env` file on the operator's machine and explicitly excluded from Docker build contexts via `.dockerignore`. `HF_TOKEN`, when set, is passed to containers as an environment variable — operators in classified environments should apply appropriate host-level access controls on the Docker socket.

### Dependency Scanning

All repos run Bandit, Ruff, and mypy in CI. Known suppressions are documented inline with `# nosec` annotations and justification. `shell=False` is enforced on all subprocess calls except where Windows compatibility requires `shell=True`, in which case all arguments are internally constructed with no user input.

---

## Known Limitations

| Item | Status | Mitigation |
|---|---|---|
| HTTP only by default | Addressable | HTTPS server block documented in nginx config — enable with operator certificates |
| `PermitRootLogin yes` in inference worker | Addressable | Use dedicated non-root user on controlled infrastructure |
| HF_TOKEN visible via `docker inspect` | By design | Apply host-level access controls on the Docker socket |
| No inter-container network policy | By design | Implement Docker network policies for segmented deployments |
| `shell=True` on Windows subprocess paths | By design | Windows-only; all arguments are internally constructed |
| firejail `--private` allows read access to system paths | Roadmap | Full filesystem overlay isolation planned |
| Rate limiting at nginx layer only | By design | No application-layer rate limiting — nginx is the enforcement point |

---

## Responsible Disclosure

Project David Core is maintained by a solo engineer. We appreciate your patience and your help in keeping the ecosystem safe for the operators and organisations depending on it across more than 100 countries.

*Project David is created and maintained by Francis Neequaye Armah.*
*All intellectual property is solely owned by the author.*
