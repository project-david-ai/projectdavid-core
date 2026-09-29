# Scratchpad Architecture

**Project David: first-class shared working-memory resource**

**Scope:** Core resource model, Redis data plane, REST/SDK surface, model-facing platform tools, Deep Research integration, tenancy, lifecycle, and migration state.

---

## Table of Contents

1. [Purpose](#1-purpose)
2. [Canonical External Composition](#2-canonical-external-composition)
3. [Core Resource Model](#3-core-resource-model)
4. [SQL / Redis Responsibility Split](#4-sql--redis-responsibility-split)
5. [Redis Data Plane](#5-redis-data-plane)
6. [Content vs Entries](#6-content-vs-entries)
7. [Tenancy and Ownership](#7-tenancy-and-ownership)
8. [First-Class Service Layer](#8-first-class-service-layer)
9. [Public REST Surface](#9-public-rest-surface)
10. [SDK Surface](#10-sdk-surface)
11. [Two Exposure Tracks](#11-two-exposure-tracks)
12. [Platform Tool Hydration](#12-platform-tool-hydration)
13. [Model-Facing Tool Definitions](#13-model-facing-tool-definitions)
14. [Internal Tool Compatibility Surface](#14-internal-tool-compatibility-surface)
15. [ScratchpadMixin Execution Boundary](#15-scratchpadmixin-execution-boundary)
16. [Deep Research Topology](#16-deep-research-topology)
17. [Deep Research ID Propagation: Migration State](#17-deep-research-id-propagation-migration-state)
18. [Run Metadata](#18-run-metadata)
19. [Tool Routing](#19-tool-routing)
20. [Conversation Correlation vs Shared State](#20-conversation-correlation-vs-shared-state)
21. [Common Schemas](#21-common-schemas)
22. [Concurrency](#22-concurrency)
23. [Clear vs Delete](#23-clear-vs-delete)
24. [Lifecycle and Parent Deletion](#24-lifecycle-and-parent-deletion)
25. [Legacy Migration](#25-legacy-migration)
26. [Security Invariants](#26-security-invariants)
27. [Architectural Invariants](#27-architectural-invariants)
28. [Source Map](#28-source-map)
29. [Testing Map](#29-testing-map)
30. [Current Migration Status](#30-current-migration-status)
31. [Design Direction](#31-design-direction)

---

## 1. Purpose

Scratchpad is Project David's shared working-memory resource. It is designed for workflows where one or more agents need to collaborate around a durable, compact body of working state without sharing their full conversation transcripts.

The canonical resource relationship is:

```text
User
 └── Thread
      ├── Message...
      └── Scratchpad (0..1)
```

A Scratchpad:

- has its own stable `scratchpad_id`;
- belongs to exactly one authenticated user;
- is associated with exactly one parent Thread;
- stores resource identity and lifecycle metadata in SQL;
- stores mutable working state in Redis;
- can be used directly through REST/SDK APIs;
- can also be exposed to assistants as a high-level platform capability;
- is shared by Deep Research Supervisor/Worker orchestration while worker conversation Threads remain private.

The central architectural rule:

> **conversation state != shared scratchpad state**

---

## 2. Canonical External Composition

The application-facing composition contract:

```python
thread = client.threads.create_thread()

scratchpad = client.scratchpads.create_scratchpad(
    thread_id=thread.id,
)
```

The caller supplies **only** the parent `thread_id`.

The caller must **not** supply:

- `owner_id`
- `user_id`
- `tenant_id`
- `scratchpad_id`

Ownership is derived by Core from authentication:

```text
authenticated API key/session
        ↓
auth_key.user_id
        ↓
load Thread
        ↓
verify thread.owner_id == auth_key.user_id
        ↓
create Scratchpad
```

A Thread may have **at most one** Scratchpad.

---

## 3. Core Resource Model

The Scratchpad is a first-class resource rather than an alias for a Thread.

```text
Scratchpad
├── id
├── owner_id
├── thread_id
├── created_at
├── updated_at
└── meta_data
```

### Relational constraints

```text
Scratchpad.id
    PRIMARY KEY

Scratchpad.owner_id
    NOT NULL
    FK users.id
    ON DELETE CASCADE

Scratchpad.thread_id
    NOT NULL
    UNIQUE
    FK threads.id
    ON DELETE CASCADE
```

The unique `thread_id` constraint enforces:

```text
Thread 1 ───── 0..1 Scratchpad
```

### Why Scratchpad has its own ID

Historically, Scratchpad storage was keyed by `thread_id`. That made the Thread identifier do two jobs:

- conversation identity
- scratchpad storage identity

Tenantization separates those concerns:

| Identifier      | Role                                                |
| --------------- | --------------------------------------------------- |
| Thread ID       | Conversation / resource parent                      |
| Scratchpad ID   | Shared working-memory resource identity             |

This gives Scratchpad an independent lifecycle and allows future agent systems to address shared memory without pretending that the memory itself is a conversation Thread.

---

## 4. SQL / Redis Responsibility Split

The architecture deliberately separates the **resource plane** from the **mutable data plane**.

| Store     | Owns                                                                                                                                     |
| --------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **SQL**   | Resource identity, tenant ownership, parent Thread association, `created_at` / `updated_at`, `meta_data`, resource existence, relational lifecycle |
| **Redis** | Working content, ordered append-only entries                                                                                             |

Mutable Scratchpad body/history must **not** be duplicated into SQL.

```text
SQL   = identity / ownership / metadata / lifecycle
Redis = active working state
```

This matters because Scratchpad content is expected to change frequently during agent execution, while ownership and parentage are relational facts.

---

## 5. Redis Data Plane

Canonical Redis keys are tenant- and resource-scoped:

```text
scratchpad:{owner_id}:{scratchpad_id}:content
scratchpad:{owner_id}:{scratchpad_id}:entries
```

The old form is **legacy compatibility state** and is not the target storage identity:

```text
scratchpad:{thread_id}:notebook
```

### Content

`content` represents the current editable working body.

- **Storage type:** Redis `STRING`
- **Typical payload:**

```json
{
  "content": "Current plan / synthesis / working state",
  "updated_at": "..."
}
```

### Entries

`entries` is the append-only finding ledger.

- **Storage type:** Redis `LIST`
- **Append semantics:** `RPUSH`, rather than `GET` → modify client-side → `SET`. This matters because multiple workers may append concurrently.
- **Typical entry:**

```json
{
  "content": "Verified finding, URL, number, progress note...",
  "created_at": "..."
}
```

### Current data-plane operations

The first-class cache/service surface is conceptually:

- `get_content`
- `set_content`
- `append_entry`
- `list_entries`
- `clear_content`
- `clear_entries`
- `delete_all_scratchpad_data`

The current Redis cache uses a TTL policy; the resource identity remains SQL-authoritative even if mutable Redis working state expires.

---

## 6. Content vs Entries

Scratchpad intentionally separates two forms of shared state.

| | `content` | `entries` |
| --- | --- | --- |
| **Use for** | Master plan, current synthesis, structured working state, checklists, rewritten/compacted context | Worker findings, URLs, verified facts, measurements, progress notes, incremental discoveries |
| **Semantics** | Replaceable, supervisor-friendly, current-state oriented | Append-only, ordered, worker-friendly, concurrency-safe |

This avoids forcing every worker to rewrite one shared document.

```text
Supervisor
    └── updates content

Worker A
    └── RPUSH entry A

Worker B
    └── RPUSH entry B

Worker C
    └── RPUSH entry C
```

The formatted model view can combine both planes for consumption while preserving different mutation semantics underneath.

---

## 7. Tenancy and Ownership

Every Scratchpad operation is tenant-scoped. Conceptually, resource lookup behaves like:

```sql
WHERE scratchpad.id = :scratchpad_id
  AND scratchpad.owner_id = :authenticated_user_id
```

This rule applies to:

- create
- retrieve
- list
- metadata update
- content read/write
- entry append/list
- clear
- delete
- resolve by Thread

Cross-tenant addressing should normally appear as `404 Not Found` rather than revealing that another tenant's resource exists.

### Important boundary

The model **never** chooses the tenant. The public assistant tool schema must not contain:

- `owner_id`
- `user_id`
- `tenant_id`
- `thread_id`
- `scratchpad_id`

Those are runtime infrastructure concerns owned by Core.

---

## 8. First-Class Service Layer

The canonical Core facade is the **Scratchpad service**.

### Resource-oriented operations

- `create_scratchpad`
- `retrieve_scratchpad`
- `retrieve_scratchpad_by_thread`
- `ensure_scratchpad_for_thread`
- `list_scratchpads`
- `update_metadata`
- `delete_scratchpad`

### Data-oriented operations

- `get_content`
- `set_content`
- `append_entry`
- `list_entries`
- `clear_content`
- `clear_entries`
- `clear_scratchpad`
- `get_formatted_view`

The service is the convergence point between:

```text
SQL resource plane
        +
Redis data plane
```

### Compatibility resolver

During migration, Core also maintains a compatibility bridge:

```python
resolve_scratchpad_for_thread(
    thread_id,
    *,
    user_id,
)
```

Its role:

1. Find or ensure the canonical Scratchpad resource for the Thread.
2. Provide a canonical `scratchpad_id`.
3. Migrate legacy thread-keyed Redis state when necessary.
4. Allow older orchestration paths to continue functioning while callers are converted to direct resource-ID execution.

> The bridge is **transitional**. New internal paths should prefer a known `scratchpad_id`.

---

## 9. Public REST Surface

The first-class REST resource is exposed under `/v1`.

### Resource family

```text
POST   /v1/scratchpads
GET    /v1/scratchpads
GET    /v1/scratchpads/{scratchpad_id}
GET    /v1/threads/{thread_id}/scratchpad
PATCH  /v1/scratchpads/{scratchpad_id}
DELETE /v1/scratchpads/{scratchpad_id}
```

### Mutable state endpoints

```text
GET    /v1/scratchpads/{scratchpad_id}/content
PUT    /v1/scratchpads/{scratchpad_id}/content
DELETE /v1/scratchpads/{scratchpad_id}/content

POST   /v1/scratchpads/{scratchpad_id}/entries
GET    /v1/scratchpads/{scratchpad_id}/entries
DELETE /v1/scratchpads/{scratchpad_id}/entries

POST   /v1/scratchpads/{scratchpad_id}/clear
```

There is intentionally **no public "ensure" endpoint**. `ensure_scratchpad_for_thread()` is an internal orchestration/service concern.

---

## 10. SDK Surface

The first-class SDK surface is `client.scratchpads`.

### Canonical usage

```python
thread = client.threads.create_thread()

scratchpad = client.scratchpads.create_scratchpad(
    thread_id=thread.id,
)

scratchpad = client.scratchpads.retrieve_scratchpad(
    scratchpad.id
)

client.scratchpads.set_content(
    scratchpad.id,
    "Working plan...",
)

client.scratchpads.append_entry(
    scratchpad.id,
    "Verified finding...",
)

entries = client.scratchpads.list_entries(
    scratchpad.id
)
```

The Scratchpads client follows the same resource-client architecture as the rest of the Project David SDK and uses Common Pydantic request/response models.

### Typical methods

| Group      | Methods                                                                                                                        |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Resource   | `create_scratchpad`, `retrieve_scratchpad`, `retrieve_scratchpad_by_thread`, `list_scratchpads`, `update_scratchpad`, `delete_scratchpad` |
| Content    | `get_content`, `set_content`, `clear_content`                                                                                  |
| Entries    | `append_entry`, `list_entries`, `clear_entries`                                                                                |
| State      | `clear_scratchpad`                                                                                                             |

---

## 11. Two Exposure Tracks

Scratchpad has two intentionally different public-facing usage models.

### Track A: Developer / Application Resource API

Used when an application explicitly manages Scratchpad resources:

```text
Application
    ↓
client.scratchpads.*
    ↓
/v1/scratchpads/*
    ↓
ScratchpadService
    ↓
SQL + Redis
```

This track exposes resource lifecycle and state-management operations.

### Track B: Assistant Platform Capability

Most agents do not need the resource-management surface in their prompt. They need the ability to:

- read shared state
- replace shared working content
- append a finding

The assistant therefore subscribes using a placeholder:

```python
tools=[
    {"type": "scratchpad"},
]
```

Core expands that placeholder at context-build time.

---

## 12. Platform Tool Hydration

Platform capabilities are registered through `PLATFORM_TOOL_MAP`.

```python
SCRATCHPAD_TOOLS = [
    read_scratchpad,
    update_scratchpad,
    append_scratchpad,
]

PLATFORM_TOOL_MAP = {
    "code_interpreter": ...,
    "computer": ...,
    "file_search": ...,
    "web_search": ...,
    "scratchpad": SCRATCHPAD_TOOLS,
}
```

The assistant can therefore persist `{"type": "scratchpad"}` rather than carrying three expanded function definitions in its stored configuration.

### Runtime expansion

```text
{"type": "scratchpad"}
        ↓
ContextMixin._resolve_and_prioritize_platform_tools(...)
        ↓
read_scratchpad
update_scratchpad
append_scratchpad
```

### Why this matters

Persisted assistant configuration remains:

- small
- stable
- capability-oriented

while Core remains free to evolve the concrete function definitions. The same architecture is used for other platform capabilities such as `web_search`.

---

## 13. Model-Facing Tool Definitions

The model-facing Scratchpad capability consists of exactly **three** operational functions.

| Function | Signature | Purpose |
| --- | --- | --- |
| **Read** | `read_scratchpad()` | Inspect the current shared working state |
| **Update** | `update_scratchpad(content="...")` | Replace/restructure the current working content |
| **Append** | `append_scratchpad(note="...")` | Record an incremental finding without replacing the working body |

The model should **never** provide a `scratchpad_id`.

**Correct:**

```python
append_scratchpad(
    note="Revenue reported as ..."
)
```

**Incorrect:**

```python
append_scratchpad(
    scratchpad_id="sp_...",
    note="..."
)
```

Resource identity is injected/resolved by the orchestration runtime.

---

## 14. Internal Tool Compatibility Surface

Project David still retains the legacy/model-facing tool router:

```text
POST /tools/scratchpad/read
POST /tools/scratchpad/update
POST /tools/scratchpad/append
```

and the SDK `ToolsClient` compatibility methods. These are distinct from the first-class developer resource API.

### Model / orchestrator path

```text
MODEL / ORCHESTRATOR
read_scratchpad()
update_scratchpad()
append_scratchpad()
        ↓
internal Native execution
        ↓
ScratchpadService
```

### Application / SDK path

```text
APPLICATION / SDK USER
client.scratchpads.*
        ↓
/v1/scratchpads/*
        ↓
ScratchpadService
```

> **Do not** route internal orchestration through the external SDK. Internal Core execution should use the native service layer directly.

---

## 15. ScratchpadMixin Execution Boundary

`ScratchpadMixin` is the model-tool execution boundary. Its important invariant:

> **conversation_thread_id != scratchpad identity**

- Tool protocol correlation belongs to the active conversation Thread.
- Shared memory belongs to the Scratchpad resource.

### ID-first path

```text
tool call
    ↓
scratchpad_id known
    ↓
ScratchpadMixin
    ↓
ScratchpadService
    ↓
Redis resource keys
```

### Migration fallback path

```text
tool call
    ↓
scratchpad_id absent
    ↓
scratch_pad_thread / thread locator
    ↓
resolve_scratchpad_for_thread(...)
    ↓
canonical scratchpad_id
    ↓
ScratchpadService
```

Once the canonical ID is resolved, execution occurs against the first-class Scratchpad API by resource ID. This prevents new code from continuing to treat Thread identity as Scratchpad identity.

---

## 16. Deep Research Topology

Deep Research is the primary multi-agent consumer of Scratchpad. Its intended topology:

```text
                    ┌───────────────────────┐
                    │ Parent/User Thread    │
                    └──────────┬────────────┘
                               │
                               │ owns / composes
                               ▼
                    ┌───────────────────────┐
                    │ Shared Scratchpad     │
                    │ scratchpad_id = sp_*  │
                    └─────┬────────┬────────┘
                          ▲        ▲
                          │        │
              ┌───────────┘        └───────────┐
              │                                │
      ┌───────┴────────┐               ┌───────┴────────┐
      │ Worker A       │               │ Worker B       │
      │ private Thread │               │ private Thread │
      └────────────────┘               └────────────────┘
```

Worker Threads exist for provider conversation/tool protocol. The Scratchpad exists for promoted shared research state. Therefore:

- **worker transcript** is private/local
- **promoted finding** is shared

### Typical flow

```text
Supervisor
    ↓
seed/update shared Scratchpad
    ↓
delegate Worker A
    ↓
Worker A reads shared Scratchpad
    ↓
Worker A searches / verifies
    ↓
Worker A appends promoted finding
    ↓
Supervisor re-reads shared state
    ↓
delegate Worker B
    ↓
Worker B sees seed + Worker A finding
    ↓
Worker B appends new finding
    ↓
Supervisor synthesizes
```

This avoids copying complete worker transcripts into the Supervisor context.

---

## 17. Deep Research ID Propagation: Migration State

The target execution path:

```text
Parent Thread
    ↓
resolve/create Scratchpad ONCE
    ↓
scratchpad_id
    ↓
Supervisor
    ↓
delegated Run metadata
    ↓
Worker A / Worker B / Worker N
    ↓
process_tool_calls(scratchpad_id=...)
    ↓
read / update / append
```

### Implemented

The execution boundary supports direct `scratchpad_id`. When the ID is supplied:

- no thread-resolution bridge is required
- all operations address the canonical resource

The compatibility fallback remains available.

### Transitional compatibility

Current orchestration still retains `scratch_pad_thread` for legacy/in-flight paths.

**Migration rule:**

```text
prefer scratchpad_id
fallback to scratch_pad_thread
```

The legacy thread locator should be removed **only after** every active Supervisor/Worker/provider path propagates `scratchpad_id` directly. Do not remove it merely because the first-class REST resource exists.

---

## 18. Run Metadata

Deep Research delegated workers need enough metadata to recover shared resource identity independently of their private Thread.

The migration target for ephemeral Run metadata:

```python
meta_data={
    "batfish_owner_user_id": origin_user_id,
    "scratchpad_id": scratchpad_id,

    # temporary compatibility locator
    "scratch_pad_thread": parent_thread_id,
}
```

Workers hydrate the canonical ID from Run metadata and pass it through tool routing.

> The **resource ID**, not the worker Thread ID, is the long-term shared-state contract.

---

## 19. Tool Routing

`ToolRoutingMixin.process_tool_calls()` is the orchestration handoff between provider tool calls and concrete Core handlers.

Scratchpad routing must be **symmetrical** for all three operations:

- `read_scratchpad`
- `update_scratchpad`
- `append_scratchpad`

Each operation should receive the same canonical `scratchpad_id`, while retaining the legacy Thread locator only as migration fallback.

**Key rule:**

```text
model arguments        do NOT contain infrastructure identity
orchestrator arguments MAY contain infrastructure identity
```

---

## 20. Conversation Correlation vs Shared State

This distinction is easy to break and is one of the most important Scratchpad invariants.

Suppose Worker A is executing on:

```text
conversation_thread_id = thread_worker_a
```

while sharing:

```text
scratchpad_id = sp_parent
```

Then:

```text
Scratchpad read/write
    → sp_parent

Tool-result protocol message
    → thread_worker_a
```

Writing the provider tool result into the shared parent Thread would corrupt the worker's tool-call sequence. So:

> **shared state destination != tool protocol correlation destination**

`ScratchpadMixin` must preserve this separation.

---

## 21. Common Schemas

The first-class Scratchpad API uses typed Common models rather than loose dictionaries/strings.

### Schema family

- `ScratchpadCreate`
- `ScratchpadRead`
- `ScratchpadUpdate`
- `ScratchpadList`
- `ScratchpadDeleted`
- `ScratchpadContentUpdate`
- `ScratchpadContentRead`
- `ScratchpadEntryCreate`
- `ScratchpadEntryRead`
- `ScratchpadEntryList`
- `ScratchpadStateCleared`

### Important schema rules

| Schema | Rule |
| --- | --- |
| `ScratchpadCreate` | Accepts `thread_id`; does **not** accept `owner_id` / `user_id` |
| `ScratchpadEntryCreate` | Rejects empty content |
| `ScratchpadContentUpdate` | Permits empty content |

The public REST/SDK surface should remain typed end-to-end.

---

## 22. Concurrency

Scratchpad is explicitly designed for multi-worker access.

**Unsafe append:**

```text
Worker A GET
Worker B GET
Worker A modifies
Worker B modifies
Worker A SET
Worker B SET
        ↓
lost write
```

**Target:**

```text
Worker A RPUSH
Worker B RPUSH
Worker C RPUSH
        ↓
three retained entries
```

**Required concurrency invariant:**

```text
N concurrent appends
        ↓
N retained entries
        ↓
zero lost writes
```

Content replacement and entries append are deliberately separate so that a Supervisor rewriting the plan does not overwrite worker findings.

---

## 23. Clear vs Delete

These are different operations.

| | **Clear** | **Delete** |
| --- | --- | --- |
| Redis content | Cleared | Purged |
| Redis entries | Cleared | Purged |
| SQL Scratchpad resource | Retained | Deleted |
| `scratchpad_id` | Retained | Ceases to exist |
| Thread association | Retained | Ceases to exist |
| **Use when** | The resource should remain but its working state should be reset | The Scratchpad resource itself should cease to exist |

---

## 24. Lifecycle and Parent Deletion

SQL foreign keys provide relational cascade behavior:

```text
delete Thread
    ↓
delete Scratchpad SQL row

delete User
    ↓
delete owned Scratchpad SQL rows
```

That is only half of the lifecycle contract. Redis is external to SQL and does not participate in relational cascades. Therefore Core must also ensure:

```text
delete Scratchpad
    → Redis keys removed

delete Thread
    → child Scratchpad Redis keys removed

delete User
    → tenant Scratchpad Redis keys removed
```

### Remaining lifecycle hardening

> **This is an important completion gate.**

A SQL cascade that leaves the following keys behind is an **orphaned data-plane leak**:

```text
scratchpad:{owner_id}:{scratchpad_id}:content
scratchpad:{owner_id}:{scratchpad_id}:entries
```

Tenantization should not be considered fully closed until Thread/User deletion paths prove Redis cleanup.

---

## 25. Legacy Migration

The historical cache identity was:

```text
scratchpad:{thread_id}:notebook
```

The authoritative identity is now:

```text
scratchpad:{owner_id}:{scratchpad_id}:content
scratchpad:{owner_id}:{scratchpad_id}:entries
```

### Migration flow

```text
legacy Thread locator
        ↓
resolve_scratchpad_for_thread(...)
        ↓
ensure canonical SQL resource
        ↓
if canonical Redis state absent:
    migrate legacy data once
        ↓
continue using canonical resource ID
```

The system should **not** maintain two independently writable Scratchpad stores. Compatibility exists to converge old state onto the new data plane, not to create permanent dual-write semantics.

---

## 26. Security Invariants

The following are **non-negotiable**.

| Area | Invariant |
| --- | --- |
| **Ownership** | Owner comes from authentication, never from model/client input |
| **Resource lookup** | `scratchpad_id` + authenticated owner |
| **Cross-tenant behavior** | Do not reveal another tenant's resource existence |
| **Tool schema** | Do not expose `owner_id`, `user_id`, `tenant_id`, `scratchpad_id`, or `thread_id` to the model-facing Scratchpad functions |
| **Internal execution** | Use Core's native service layer rather than an HTTP round-trip through the public SDK |

---

## 27. Architectural Invariants

The shortest statements of the design:

1. Scratchpad is a resource, not a Thread alias.
2. Thread is the compositional parent.
3. One Thread has at most one Scratchpad.
4. Ownership is derived from authentication.
5. SQL owns identity.
6. Redis owns mutable state.
7. `content != entries`
8. worker conversation != shared Scratchpad
9. model tool arguments != infrastructure identity
10. Assistant subscribes to capability; Core hydrates implementation.
11. Prefer `scratchpad_id`; thread locator is compatibility only.
12. Redis lifecycle must follow SQL lifecycle.

---

## 28. Source Map

The following files form the primary implementation surface.

### Core: resource / data plane

| File | Role |
| --- | --- |
| `src/api/entities_api/services/scratchpad_service.py` | Primary first-class Scratchpad facade: resource operations, data operations, formatted view, compatibility resolution/migration |
| `src/api/entities_api/cache/scratchpad_cache.py` | Redis data plane and canonical resource-scoped key operations |
| `src/api/entities_api/routers/scratchpads_router.py` | First-class `/v1/scratchpads/*` REST API |
| `src/api/entities_api/routers/tools_router.py` | Legacy/model-facing `/tools/scratchpad/*` compatibility routes |

### Core: model-facing platform capability

| File | Role |
| --- | --- |
| `src/api/entities_api/constants/tools.py` | `PLATFORM_TOOL_MAP`, including the `scratchpad` capability → `SCRATCHPAD_TOOLS` expansion |
| `src/api/entities_api/platform_tools/definitions/scratch_pad/read_scratchpad.py` | Concrete model-facing function definition (read) |
| `src/api/entities_api/platform_tools/definitions/scratch_pad/update_scratchpad.py` | Concrete model-facing function definition (update) |
| `src/api/entities_api/platform_tools/definitions/scratch_pad/append_scratchpad.py` | Concrete model-facing function definition (append) |
| `src/api/entities_api/orchestration/mixins/context_mixin.py` | Hydrates assistant capability placeholders into concrete platform tool definitions when the input context is built |

### Core: orchestration

| File | Role |
| --- | --- |
| `src/api/entities_api/orchestration/mixins/scratchpad_mixin.py` | Model-tool execution boundary; direct resource-ID execution with compatibility fallback |
| `src/api/entities_api/orchestration/mixins/tool_routing_mixin.py` | Routes model tool calls to Scratchpad handlers |
| `src/api/entities_api/orchestration/mixins/delegation_mixin.py` | Deep Research worker lifecycle and shared parent-Scratchpad handoff |
| `src/api/entities_api/orchestration/engine/orchestrator_core.py` | Top-level orchestration state and tool-routing handoff |
| `src/api/entities_api/orchestration/workers/*` | Provider workers hydrate Run/orchestration state and invoke tool routing |

### Deep Research capability configuration

| File | Role |
| --- | --- |
| `src/api/entities_api/utilities/assistant_manager.py` | Creates ephemeral research workers. Persisted workers subscribe to logical platform capabilities: `[{"type": "web_search"}, {"type": "scratchpad"}]` |
| `src/api/entities_api/platform_tools/tool_reigistry/research_worker.py` | Research-worker execution configuration and bounded turn budget |
| `DEEP_RESEARCH.md` | Deep Research architecture, worker/Supervisor topology, shared-state boundary, provider correlation, failure semantics, and integration model |

### Common

Project David Common contains the typed Scratchpad request/response models and canonical identifier support.

Relevant schema family:

`ScratchpadCreate`, `ScratchpadRead`, `ScratchpadUpdate`, `ScratchpadList`, `ScratchpadDeleted`, `ScratchpadContentUpdate`, `ScratchpadContentRead`, `ScratchpadEntryCreate`, `ScratchpadEntryRead`, `ScratchpadEntryList`, `ScratchpadStateCleared`

### ORM

Project David ORM contains the SQL Scratchpad resource model and User/Thread relationships.

Relevant concepts:

- `Scratchpad` model
- `User.scratchpads`
- `Thread.scratchpad`
- `UNIQUE(thread_id)`
- `owner_id` FK
- `thread_id` FK

### SDK

| File | Role |
| --- | --- |
| `src/projectdavid/clients/scratchpads_client.py` | First-class application client exposed as `client.scratchpads` |

The existing Tools client remains a compatibility surface for model-style thread-locator Scratchpad calls.

---

## 29. Testing Map

Important test categories:

### Resource / service tests

- create / retrieve / list / update / delete
- one Scratchpad per Thread
- typed responses
- duplicate creation conflict

### Tenant isolation

- User A can access User A Scratchpad
- User B cannot access User A Scratchpad
- cross-tenant addressing does not leak existence

### Redis data plane

- tenant-scoped keys
- set/get content
- atomic append
- list ordering
- clear
- full purge

### Execution boundary

- direct `scratchpad_id` skips thread resolution
- read/update/append address the same resource
- legacy thread fallback still resolves correctly
- tool output remains correlated to worker conversation Thread

### Platform capability

```text
{"type": "scratchpad"}
    ↓
read_scratchpad
update_scratchpad
append_scratchpad
```

and no model-facing schema exposes resource identity.

### Deep Research

- parent shared Scratchpad
- private worker Threads
- worker findings promoted into shared state
- no consumer tool-call escape
- bounded worker execution
- failure lifecycle remains correct

### Lifecycle

**Still required as a hard completion gate:**

- Scratchpad delete → Redis purge
- Thread delete → child Redis purge
- User delete → tenant Redis purge

---

## 30. Current Migration Status

The architecture has already crossed the major boundary from a Deep Research helper to a first-class resource.

### Established

- first-class SQL resource
- typed Common schemas
- canonical Scratchpad IDs
- tenant ownership
- one Scratchpad per Thread
- first-class Core service
- first-class REST surface
- tenant/resource-scoped Redis data plane
- content + entries split
- atomic entry append
- first-class SDK surface
- model-facing read/update/append tools
- platform placeholder `{"type": "scratchpad"}`
- runtime tool hydration
- ID-first `ScratchpadMixin` execution
- legacy compatibility resolver

### Transitional

- `scratch_pad_thread` compatibility locator
- legacy thread-keyed state migration
- Deep Research propagation still being converted to direct `scratchpad_id`

### Remaining hardening

- complete `scratchpad_id` propagation through all active worker/provider paths
- prove Thread deletion purges Redis
- prove User deletion purges Redis
- remove compatibility locator only after migration horizon
- remove legacy thread-keyed APIs only when no active consumer requires them

---

## 31. Design Direction

Scratchpad began as a Deep Research implementation detail. Its intended role is broader:

- Deep Research
- Career Agent
- multi-agent planning
- long-running investigations
- engineering agents
- workflow supervisors
- future collaborative agent systems

The reusable abstraction is not "a research notebook". It is:

> **A tenant-owned, Thread-associated, independently addressable shared working-memory resource with a relational identity plane and a high-frequency Redis state plane.**

That is the architectural boundary future implementations should preserve.
