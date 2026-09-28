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

### Model Context Protocol (MCP) Security

Project David supports remote Model Context Protocol (MCP) servers as first-class tool providers. MCP registration, credential ownership, discovery, attachment, and execution are handled by Project David Core rather than delegated to the end-user SDK.

#### Tenant Isolation

Remote MCP servers are represented by tenant-owned server registrations.

A registration belongs to a specific Project David user or tenant and is resolved within that ownership boundary. Two tenants may register the same remote MCP endpoint while using entirely different credentials.

MCP credentials are not globally attached to a URL and are not shared implicitly between users.

Credential lookup during execution is constrained by both the credential identifier and authenticated owner. A credential belonging to one tenant cannot be resolved through another tenant's MCP registration.

#### Credential Storage

Authenticated MCP registrations reference a Project David credential record rather than storing authentication material directly in the server registration, assistant configuration, tool definition, or inference state.

Credential values are encrypted before persistence using Fernet authenticated encryption.

The encryption key is supplied independently through:

`PROJECT_DAVID_CREDENTIAL_KEY`

The encryption key is not stored alongside encrypted credential material.

If the required encryption key is unavailable, credential encryption and decryption fail closed rather than falling back to plaintext storage.

Credential records retain an encryption version so that future encryption migrations can be introduced without silently changing the interpretation of existing ciphertext.

#### Secret Resolution

MCP credentials are resolved only when Core is preparing an authenticated outbound request to the registered MCP server.

The plaintext credential:

- is not returned through the MCP registration API
- is not embedded in assistant tool configuration
- is not stored in Redis inference state
- is not included in normal object representations
- is not included in MCP tool metadata
- is not returned to the SDK during tool discovery
- is not required to be resupplied by the client during inference
- is not intentionally written to application logs or error responses

Decryption occurs at the outbound execution boundary and the resolved secret is applied to the remote MCP request immediately before transport.

The client therefore interacts with an MCP server registration and its exposed tools, not with the underlying authentication secret.

#### Authenticated MCP Execution

Authenticated MCP tool execution is owned by Project David Core.

The SDK does not execute registered remote MCP tools locally and does not need access to the remote provider credential.

The execution path is:

1. An authenticated user registers a remote MCP server and supplies its credential.
2. Core encrypts the credential and persists an indirect credential reference on the MCP registration.
3. Core performs MCP tool discovery through the registered server.
4. Selected MCP tools may be attached to an assistant.
5. During inference, the model selects an attached MCP tool.
6. Core resolves the tenant-owned credential server-side.
7. Core opens the authenticated MCP transport and executes the remote tool.
8. The remote result is returned into the inference loop as a tool result.
9. The model continues the conversation using that result.

The original remote credential is not required to cross the SDK boundary again after registration.

#### Tool-Call Correlation

Structured tool calls emitted by model providers are normalized into Project David's internal tool-call representation before dispatch.

Every promoted structured call receives a stable tool-call identifier if the provider did not supply one.

The same identifier is preserved across:

- assistant tool-call persistence
- action creation
- MCP dispatch
- remote tool execution
- tool-result persistence
- subsequent model turns

This prevents tool results from becoming detached from the model call that initiated them and provides a consistent correlation boundary across providers.

Provider-native structured tool calls are authoritative when available. Project David's legacy textual function-call parser remains available as a compatibility fallback for providers that do not emit structured tool-call events.

#### Tool Discovery and Attachment

Remote MCP tools are discovered by Core using the registered server configuration.

Assistant attachment stores the exposed tool definition required for model selection but does not copy the underlying MCP authentication credential into the assistant.

A model therefore receives the callable tool schema necessary to decide whether to use a tool without receiving the secret required to authenticate to the remote service.

#### Authentication Schemes

The currently implemented authenticated remote MCP transport supports bearer-token credentials.

The authentication model is deliberately separated from MCP server registration and tool execution so that additional authentication schemes can be introduced without requiring provider-specific execution paths.

OAuth, interactive authorization flows, certificate authentication, and other credential mechanisms should not be assumed to be supported unless explicitly documented by the release in use.

#### Failure Behaviour

Authenticated MCP execution fails closed when required authentication material cannot be resolved or decrypted.

Authentication failures from the remote MCP server are treated as execution failures and are not converted into anonymous requests.

Project David does not intentionally fall back from an authenticated MCP registration to unauthenticated transport if credential resolution fails.

Malformed structured tool calls, invalid argument payloads, and tool calls that cannot be correlated safely are rejected rather than executed with ambiguous state.

#### Public MCP Servers

MCP servers that require no authentication may be registered and executed without a credential.

Authenticated and unauthenticated MCP servers use the same Core-owned discovery, assistant attachment, routing, execution, and result-handling architecture. Authentication is an execution concern attached to the server registration rather than a separate tool system.

This separation is intended to prevent authentication-specific behaviour from leaking into model prompting, SDK execution, or individual MCP tool implementations.

#### Operator Responsibilities

Operators enabling remote MCP servers should:

- use dedicated provider credentials with the minimum required permissions
- rotate MCP credentials according to the remote provider's security policy
- protect `PROJECT_DAVID_CREDENTIAL_KEY` independently from the database containing encrypted credentials
- restrict access to Docker, container environments, database administration, and host process inspection
- use HTTPS MCP endpoints when credentials are transmitted over remote networks
- revoke remote provider credentials immediately if the Project David host or credential-encryption key is suspected to be compromised
- avoid placing live MCP credentials in source-controlled `.env`, test fixtures, assistant definitions, or integration scripts

Authenticated MCP integration tests should obtain credentials from local secret-bearing test configuration and must not commit live provider credentials to the repository.

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
| Authenticated MCP transport currently bearer-token based | Current scope | Authentication is isolated behind tenant-owned credential references so additional schemes can be added without provider-specific tool execution paths |

---

## Responsible Disclosure

Project David Core is maintained by a solo engineer. We appreciate your patience and your help in keeping the ecosystem safe for the operators and organisations depending on it across more than 100 countries.

*Project David is created and maintained by Francis Neequaye Armah.*
*All intellectual property is solely owned by the author.*
