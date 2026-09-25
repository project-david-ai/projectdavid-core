# Project David

[![License: PolyForm Noncommercial](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-blue.svg)](https://polyformproject.org/licenses/noncommercial/1.0.0/)
[![Docker Pulls](https://img.shields.io/docker/pulls/thanosprime/entities-api-api?label=API%20Pulls&logo=docker&style=flat-square)](https://hub.docker.com/r/thanosprime/entities-api-api)
[![Docker Image Version](https://img.shields.io/docker/v/thanosprime/entities-api-api?sort=semver&label=API%20Version&style=flat-square)](https://hub.docker.com/r/thanosprime/entities-api-api/tags)
[![CI](https://github.com/frankie336/entities_api/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/frankie336/entities_api/actions/workflows/ci.yml)

Project David is a self-hosted runtime for building stateful LLM and agent systems.

It provides the application infrastructure around model inference: assistants, threads, messages, runs, tool execution, retrieval, persistent state, provider routing, streaming, observability and deployment.

The API follows the resource model introduced by the OpenAI Assistants API, while keeping the runtime independent of any single model provider.

Project David can use hosted model APIs, OpenAI-compatible endpoints and local inference while keeping the surrounding application stack under your control.

![Project David](https://raw.githubusercontent.com/frankie336/entities_api/master/assets/projectdavid_logo.png)

---

## What Project David provides

- Stateful **Assistants, Threads, Messages and Runs**
- Multi-provider model routing
- Tool and function execution
- Multi-agent delegation
- OpenAI-style vector store and retrieval APIs
- File ingestion, chunking, embedding, indexing and search
- Qdrant-backed vector retrieval
- Persistent application state
- Sandboxed code execution using FireJail and PTY isolation
- Web tooling
- Real-time streaming
- Signed file delivery
- API-key based user and service access
- OpenTelemetry and Jaeger tracing
- Redis, MySQL and Qdrant persistence
- Containerized deployment
- Local and private inference support

Project David is intended for systems where the model is one component of the application rather than the application itself.

---

## Architecture

Project David separates model inference from application execution.

Applications interact with a stable runtime API. The runtime owns application state, retrieval, tool execution, provider selection and execution lifecycle independently of the model-serving layer.

A typical execution path is:

```text
Client
  |
  v
Project David API
  |
  v
Assistant / Thread / Run lifecycle
  |
  +--> Persistent application state
  |
  +--> Retrieval / Vector Store
  |
  +--> Tool execution
  |
  +--> Code / Web execution
  |
  v
Provider Router
  |
  +--> Hosted model API
  |
  +--> OpenAI-compatible endpoint
  |
  +--> Local inference
  |
  v
Streaming result
  |
  +--> Persisted execution state
  |
  +--> OpenTelemetry / Jaeger tracing
```

The model provider is therefore replaceable without requiring the application to be rewritten around a different provider-specific execution model.

### Platform stack

![Project David Stack](https://raw.githubusercontent.com/project-david-ai/projectdavid-platform/master/assets/svg/projectdavid-stack.svg)

---

## Runtime model

Project David uses explicit resources rather than treating a model call as the application boundary.

### Assistants

Assistants define model configuration, instructions, tools and execution behaviour.

### Threads

Threads hold durable conversational and application context.

### Messages

Messages are persisted independently of the inference provider.

### Runs

Runs represent execution. A run can involve model inference, retrieval, tool calls, code execution and additional agent activity before reaching a terminal state.

### Tools

Tools are exposed to the runtime as executable capabilities rather than being embedded directly into provider-specific application code.

This separation allows application behaviour, state and execution policy to remain stable while models and providers change underneath it.

---

## Retrieval and vector stores

Project David includes an API-driven retrieval pipeline built around OpenAI-style vector store resources.

The runtime supports the path from file ingestion through retrieval and inference:

```text
File
  |
  v
Ingestion
  |
  v
Parsing / Chunking
  |
  v
Embedding
  |
  v
Indexing
  |
  v
Qdrant
  |
  v
Search / Retrieval
  |
  v
Context assembly
  |
  v
Inference
```

Files, application state, vector data and retrieval infrastructure can remain inside the deployment boundary.

The retrieval layer is part of the runtime rather than a separate application-specific integration.

---

## Provider-independent inference

Project David isolates provider-specific behaviour behind the runtime.

Current deployments can route inference to:

- Hyperbolic
- Together AI
- Ollama
- OpenAI-compatible endpoints
- Other providers implemented through the provider interface

This allows the same application resources and execution model to be used across different inference backends.

Provider selection can therefore be treated as an infrastructure concern rather than being hard-coded into application logic.

---

## Private and multi-provider deployment

Project David does not require application data to be owned or persisted by a model provider.

The API, application state, files, retrieval infrastructure, vector database, tool execution and observability stack can all run inside infrastructure controlled by the operator.

Deployments can combine local and external inference providers according to security, residency, performance and cost requirements.

For example, a deployment can keep:

- file ingestion
- parsing and chunking
- embeddings
- vector storage
- retrieval
- application state
- tool execution
- tracing

inside its own infrastructure while selectively routing permitted inference workloads to an external provider.

A deployment using local embeddings and local inference can keep the complete retrieval-to-inference path inside the deployment boundary.

---

## Data protection

Project David was designed for privacy-conscious and multi-provider deployments.

Data storage, retrieval, inference and tool execution are separate architectural boundaries. This allows operators to control where application data is stored, which services process it and which inference providers may receive it.

The platform can therefore be used as part of systems designed to meet GDPR, data-residency and organisational security requirements.

Compliance is deployment-specific. It depends on infrastructure, configuration, model providers, retention policies, operating procedures and the wider system in which Project David is used.

---

## Security

Project David has undergone independent security review.

The platform is designed around explicit user credentials, isolated execution, locally generated secrets and deployment-controlled infrastructure.

Relevant controls include:

- separate administrator and user API keys
- one-time display of generated API keys
- locally generated deployment secrets
- secrets excluded from version control
- sandboxed code execution
- signed file URLs
- deployment-controlled persistence
- provider separation
- auditable execution through OpenTelemetry tracing

Details of the security review are available on request.

---

## Observability

Execution is instrumented with OpenTelemetry and can be inspected through Jaeger.

Tracing is intended to make the runtime observable across model calls, tool execution and the surrounding application lifecycle.

This is particularly useful when diagnosing failures that do not originate in the model itself, including:

- provider errors
- failed tool calls
- retrieval failures
- execution state transitions
- network and service failures
- latency across runtime components

---

## Quick start

### 1. Install the local package

```bash
pip install -e .
```

### 2. Build and start the Docker stack

```bash
platform-api docker-manager --mode both
```

On first run, the CLI generates the local deployment configuration.

| File | Contents |
|---|---|
| `.env` | Locally generated secrets including database passwords, `DEFAULT_SECRET_KEY`, `SEARXNG_SECRET_KEY` and related deployment configuration |
| `docker-compose.yml` | A generated Compose definition wired to the local configuration |

Both files are created once and left untouched on subsequent runs.

Verify the CLI:

```bash
platform-api --help
```

Example:

```text
Usage: platform-api [OPTIONS] COMMAND [ARGS]...

Entities API management CLI.

╭─ Commands ───────────────────────────────────────────────────────────────╮
│ configure        Update variables in an existing .env without          │
│                  regenerating secrets.                                 │
│ bootstrap-admin  Provision the default admin user inside the running   │
│                  API container.                                        │
╰─────────────────────────────────────────────────────────────────────────╯
```

For the complete command reference, see:

[Docker orchestration commands](https://github.com/project-david-ai/projectdavid_docs/blob/master/src/pages/api-infra/docker_commands.md)

---

### 3. Provision an administrator

Set `SPECIAL_DB_URL` before running the bootstrap command.

Linux / macOS:

```bash
export SPECIAL_DB_URL=mysql+pymysql://user:password@localhost:3307/entities_db
```

Windows PowerShell:

```powershell
Get-Content .env | ForEach-Object {
    if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
        [System.Environment]::SetEnvironmentVariable(
            $matches[1].Trim(),
            $matches[2].Trim()
        )
    }
}
```

Provision the default administrator:

```bash
platform-api bootstrap-admin bootstrap-admin
```

Or provide the values explicitly:

```bash
platform-api bootstrap-admin \
  --db-url "mysql+pymysql://user:password@localhost:3307/entities_db" \
  --email "admin@example.com" \
  --name "Default Admin"
```

Example output:

```text
================================================================
 ✓  Admin API Key Generated
================================================================
 Email   : admin@example.com
 User ID : user_abc123...
 Prefix  : ad_abc12
----------------------------------------------------------------
 API KEY : ad_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
----------------------------------------------------------------
 This key will NOT be shown again.
================================================================
```

Store the key immediately. It is shown once and cannot be recovered.

---

### 4. Provision a user

Install the SDK:

```bash
pip install projectdavid
```

Create a user with the administrator credential:

```python
import os

from dotenv import load_dotenv
from projectdavid import Entity

load_dotenv()

client = Entity(api_key=os.getenv("ADMIN_API_KEY"))

new_user = client.users.create_user(
    full_name="Kevin Flynn",
    email="flynn@encom.com",
    is_admin=False,
)

print(new_user)
```

Issue an API key for the user:

```python
api_key = client.keys.create_key_for_user(
    target_user_id=new_user.id,
    key_name="The Grid",
)

print(api_key.plain_key)
# ea_z_5YV4zGly50UHKlenc9BgTCQXtE....
```

Do not use the administrator key for normal application requests.

Connect using the user credential:

```python
client = Entity(api_key=os.getenv("USER_API_KEY"))
```

---

## Deployment model

Project David is containerized and intended to run as infrastructure rather than as an embedded library inside an application process.

The runtime is composed of separate services for application execution, persistence, retrieval, observability and supporting infrastructure.

This allows deployments to replace or scale individual components without changing the external application API.

The generated Docker configuration provides the default local deployment path. Production deployments can use the same service boundaries as the basis for wider infrastructure integration.

---

## Repository map

Project David is split across several repositories with deliberately separate responsibilities.

| Repository | Responsibility |
|---|---|
| [projectdavid](https://github.com/project-david-ai/projectdavid) | Python SDK |
| [entities-common](https://github.com/project-david-ai/entities-common) | Shared utilities, schemas and validation |
| [david-core](https://github.com/project-david-ai/david-core) | Docker orchestration and infrastructure |
| [reference-frontend](https://github.com/project-david-ai/reference-frontend) | Reference streaming client |
| [entities_cook_book](https://github.com/project-david-ai/entities_cook_book) | Minimal tested examples |
| [projectdavid_docs](https://github.com/project-david-ai/projectdavid_docs) | Documentation source |

---

## Documentation

| Topic | Link |
|---|---|
| Full documentation | [docs.projectdavid.co.uk](https://docs.projectdavid.co.uk/docs) |
| SDK quick start | [SDK Quick Start](https://docs.projectdavid.co.uk/docs/sdk-quick-start) |
| Docker commands | [Docker Commands](https://docs.projectdavid.co.uk/docs/docker_commands) |
| Providers | [Providers](https://docs.projectdavid.co.uk/docs/providers) |

---

## Design principles

Project David is built around several boundaries that are intended to remain stable as the model ecosystem changes.

### Application state should not belong to the model provider

Threads, messages, files, retrieval state and execution state are runtime resources.

### Model providers should be replaceable

Applications should not need to be rewritten because inference moves from one provider to another.

### Retrieval is application infrastructure

Ingestion, vector storage, search and context assembly are first-class runtime capabilities.

### Tools are part of the execution system

Tool calls require lifecycle management, validation, state and failure handling beyond the model response that requested them.

### Local and hosted inference are deployment choices

The application model should remain stable whether inference is provided by a remote API or infrastructure controlled by the operator.

### Observability belongs in the runtime

Agent and LLM systems fail across multiple layers. Model calls, tools, retrieval, storage, networking and execution state need to be traceable as one system.

---

## Source and licensing

The Project David source is publicly available under the
[PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/).

Non-commercial use is permitted under the terms of that licence.

Commercial licensing is available separately.
