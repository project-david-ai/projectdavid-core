from pathlib import Path

content = """# Scratchpad Tenantization — Outstanding Work

## Objective

Promote the existing Deep Research scratchpad from an internal thread-keyed Redis helper into a **first-class, tenant-owned Project David resource** with a stable external identity, explicit lifecycle, strict per-user access control, safe concurrent writes, and clean deletion semantics.

The canonical external composition contract is:

```python
# Create a thread
thread = client.threads.create_thread()

# Associate one scratchpad with that thread
scratchpad = client.scratchpads.create_scratchpad(
    thread_id=thread.id,
)
```

The caller supplies only the parent `thread_id`.

The caller **must never supply `owner_id` or `user_id`**. Core derives ownership from the authenticated API key/session and validates that the caller owns the parent thread.

Canonical resource graph:

```text
User
 └── Thread
      ├── Message...
      └── Scratchpad (0..1)
```

A thread may have at most one scratchpad.

---

## Current State

### ORM

`projectdavid-orm 1.13.0` has been released with the Scratchpad resource model.

The SQL model establishes:

```text
Thread 1 ───── 0..1 Scratchpad
```

Expected structural rules:

```text
Scratchpad.id
    primary key

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

The ORM relationship model is intentionally asymmetric:

- `Thread.scratchpad`
  - one-to-one
  - compositional parent
  - `delete-orphan`
- `User.scratchpads`
  - ownership/navigation relationship
  - not the orphan parent

No mutable scratchpad body or entry history belongs in SQL.

---

## Locked Design Principles

### 1. Thread is the compositional parent

A Scratchpad is a separate resource, but it exists in association with one Thread.

```text
Thread
   └── Scratchpad
```

The Scratchpad must have its own stable `scratchpad_id`.

`thread_id` must no longer double as the scratchpad storage identity.

---

### 2. One Scratchpad per Thread

The database enforces this with a unique constraint on `scratchpads.thread_id`.

Creation must be idempotent or explicitly reject a second Scratchpad for the same Thread, depending on the final API contract.

There must never be multiple active Scratchpad resources associated with one Thread.

---

### 3. Ownership comes only from authentication

The API contract must never accept:

```text
owner_id
user_id
tenant_id
```

from the client when creating or mutating a Scratchpad.

Core must derive:

```text
authenticated API key/session
        ↓
auth_key.user_id
        ↓
Scratchpad.owner_id
```

At creation time:

```text
thread.owner_id == authenticated user
```

must be verified before the Scratchpad row is created.

---

### 4. Every operation is tenant-scoped

All resource operations must scope by both resource identity and authenticated owner.

Conceptually:

```text
WHERE scratchpad.id = :scratchpad_id
  AND scratchpad.owner_id = :authenticated_user_id
```

Cross-tenant access must fail for:

```text
GET
LIST
UPDATE
APPEND
CLEAR
DELETE
```

Tenant B must never be able to address Tenant A's Scratchpad successfully, even if Tenant B knows the Scratchpad ID or Thread ID.

Where appropriate, cross-tenant addressing should be surfaced as resource-not-found rather than leaking that another tenant's resource exists.

---

### 5. SQL owns metadata; Redis owns working state

SQL owns:

```text
identity
tenant ownership
thread association
created/updated lifecycle metadata
resource existence
```

Redis owns:

```text
working content
ordered worker entries
```

Do **not** duplicate mutable Scratchpad body or entry history into SQL.

---

### 6. Redis keys must be tenant and resource scoped

The existing shape:

```text
scratchpad:{thread_id}:notebook
```

must be retired.

Target shape:

```text
scratchpad:{owner_id}:{scratchpad_id}:content
scratchpad:{owner_id}:{scratchpad_id}:entries
```

This prevents bare thread identifiers from acting as the storage security boundary.

---

### 7. Separate supervisor content from worker append history

The current Scratchpad is one concatenated JSON content blob.

That is insufficient for concurrent workers.

Target model:

```text
content
    supervisor-editable current plan / working body

entries
    ordered append-only worker ledger
```

This gives the Supervisor a mutable working representation while preserving atomic worker contributions.

---

### 8. Appends must be atomic

The current implementation performs:

```text
GET
modify locally
SET
```

which can lose concurrent worker writes.

The new `entries` plane should use an existing atomic Redis list primitive, following Project David's existing Redis patterns where possible.

Preferred direction:

```text
RPUSH
```

Do not introduce Redis Streams, Lua, or another concurrency mechanism unless existing primitives prove insufficient.

---

### 9. Deep Research must share the Scratchpad resource ID

Current Deep Research already has the correct conceptual ownership topology:

```text
parent/user-facing thread
    ↓
shared scratchpad

worker private threads
    ↓
private tool/result conversation
```

Today it propagates the parent thread identifier as `scratch_pad_thread`.

After tenantization, Deep Research should resolve the parent's Scratchpad once and propagate:

```text
scratchpad_id
```

to Supervisor and workers.

Worker conversation threads remain private and distinct.

The shared Scratchpad resource does not become the workers' conversation Thread.

---

### 10. Deletion must cover SQL and Redis

Deletion semantics are a hard lifecycle requirement.

Deleting:

```text
Scratchpad
Thread
User
```

must not leave mutable Scratchpad data stranded in Redis.

Required acceptance gates:

```text
SQL ownership isolation          PASS
cross-tenant access denial       PASS
user deletion cascade            PASS
thread deletion cascade          PASS
Redis scratchpad purge           PASS
no orphaned scratchpad data      PASS
```

The SQL cascade alone is not sufficient; Core must also clean the Redis data plane.

This is required for clean tenant lifecycle handling and GDPR-aligned deletion behaviour.

---

## Outstanding Implementation Work

### Phase 1 — Verify Released ORM 1.13.0

Before Core migration work, verify that the released package is authoritative.

Confirm the published wheel exposes:

```text
Scratchpad model
scratchpads table definition
owner_id FK -> users.id ON DELETE CASCADE
thread_id FK -> threads.id ON DELETE CASCADE
UNIQUE(thread_id)
Thread.scratchpad relationship
User.scratchpads relationship
```

Core Alembic reconciliation must use the **released ORM package**, not a local checkout.

---

### Phase 2 — Common Schemas

Add first-class Pydantic models for the Scratchpad API.

Likely schema family:

```text
ScratchpadCreate
ScratchpadRead
ScratchpadUpdate
ScratchpadEntryCreate
ScratchpadEntryRead
ScratchpadList / response envelope if needed
```

Rules:

- `ScratchpadCreate` accepts `thread_id`.
- It does not accept `owner_id`.
- All user-facing API responses remain typed Pydantic models.
- Do not return loose strings where a structured response belongs.
- Preserve the clean resource-oriented API conventions used elsewhere in Project David.

The exact response schema should be established before Core router implementation.

---

### Phase 3 — Scratchpad ID Generation

Add canonical Scratchpad identifier generation through the existing Project David identifier service.

Target form should follow existing resource-ID conventions.

Do not generate IDs ad hoc in the router or service.

---

### Phase 4 — Core Alembic Migration

After installing/verifying released ORM 1.13.0 in Core:

1. reconcile ORM metadata against the real MySQL schema;
2. generate/review the Scratchpad migration;
3. ensure the migration creates only the intended `scratchpads` table/constraints/indexes;
4. test upgrade against MySQL;
5. test downgrade if project migration policy requires it.

Do not use SQLite as proof of the production migration.

---

### Phase 5 — Core Scratchpad Service

Create a first-class service layer responsible for:

```text
create
retrieve
list
update metadata if supported
delete
resolve by thread
ownership enforcement
```

Creation flow:

```text
authenticated user
    ↓
load thread
    ↓
verify thread.owner_id == authenticated user
    ↓
verify no Scratchpad already exists
    ↓
create Scratchpad(
        id=generated scratchpad_id,
        owner_id=authenticated user,
        thread_id=thread.id,
    )
```

All reads and mutations must be owner-scoped.

Do not trust a Thread ID merely because it exists.

---

### Phase 6 — Core REST API

Add a first-class Scratchpad router/resource surface.

Expected external flow:

```python
thread = client.threads.create_thread()

scratchpad = client.scratchpads.create_scratchpad(
    thread_id=thread.id,
)
```

Likely REST resource family:

```text
POST   /v1/scratchpads
GET    /v1/scratchpads/{scratchpad_id}
GET    /v1/scratchpads
DELETE /v1/scratchpads/{scratchpad_id}
```

Additional content/entry endpoints should be chosen consistently rather than bolted onto the old internal tool names.

All routes must derive the authenticated user from the existing auth dependency.

---

### Phase 7 — Redis Data Plane Refactor

Replace the old thread-keyed cache contract.

Current:

```text
scratchpad:{thread_id}:notebook
```

Target:

```text
scratchpad:{owner_id}:{scratchpad_id}:content
scratchpad:{owner_id}:{scratchpad_id}:entries
```

Required operations:

```text
get_content
set_content
append_entry
list_entries
clear_content
clear_entries
delete_all_scratchpad_data
```

Atomic append should use Redis-native list append semantics.

Do not use read-modify-write append.

---

### Phase 8 — Existing Scratchpad Tool Migration

Existing internal tools:

```text
read_scratchpad
update_scratchpad
append_scratchpad
```

currently operate through `thread_id`.

Refactor them to resolve/use the first-class Scratchpad identity.

During the transition, avoid retaining two independent scratchpad storage paths.

There should be one authoritative data plane.

---

### Phase 9 — Deep Research Backfill

Once the Scratchpad resource and data plane are stable:

1. resolve/create the Scratchpad associated with the parent Deep Research Thread;
2. propagate `scratchpad_id` to the Supervisor;
3. propagate the same `scratchpad_id` to all delegated workers;
4. keep each worker's private conversation Thread separate;
5. ensure tool outputs still write to worker conversation threads where appropriate;
6. ensure Scratchpad reads/writes hit the shared resource.

Target:

```text
Parent Thread
    │
    └── Scratchpad ID sp_...
            ↑
            ├── Supervisor
            ├── Worker A
            ├── Worker B
            └── Worker N
```

---

### Phase 10 — SDK

Add a first-class:

```text
ScratchpadsClient
```

and expose it from:

```python
client.scratchpads
```

Canonical call:

```python
scratchpad = client.scratchpads.create_scratchpad(
    thread_id=thread.id,
)
```

SDK methods should use Common Pydantic request/response models, following the existing Threads/Messages/MCP client conventions.

No caller-owned tenant fields.

---

### Phase 11 — Tenant Isolation Tests

Add explicit Core-level tenant tests.

Minimum cases:

```text
User A can create Scratchpad on User A Thread
User B cannot create Scratchpad on User A Thread

User A can retrieve User A Scratchpad
User B cannot retrieve User A Scratchpad

User A can update User A Scratchpad
User B cannot update User A Scratchpad

User A can append to User A Scratchpad
User B cannot append to User A Scratchpad

User A can delete User A Scratchpad
User B cannot delete User A Scratchpad

User B cannot infer existence of User A Scratchpad
```

These tests are required before calling tenantization complete.

---

### Phase 12 — Concurrency Tests

Prove that multiple workers cannot overwrite one another.

Test concurrent append behaviour against Redis.

Acceptance condition:

```text
N concurrent worker appends
    ↓
N retained entries
    ↓
stable ordering according to Redis append order
    ↓
zero lost writes
```

Supervisor content updates and worker entry appends must not corrupt each other.

---

### Phase 13 — Lifecycle / Deletion Tests

Prove:

```text
delete Scratchpad
    → SQL row removed
    → content key removed
    → entries key removed

delete Thread
    → Scratchpad SQL row removed
    → Redis keys removed

delete User
    → tenant Scratchpad SQL rows removed
    → tenant Scratchpad Redis keys removed
```

No orphaned Redis state is acceptable.

---

## Explicit Non-Goals

For this tenantization work, do not:

- store Scratchpad content/history in SQL;
- expose `owner_id` in create requests;
- keep `thread_id` as the Redis storage identity;
- allow multiple Scratchpads per Thread;
- preserve the unsafe GET-modify-SET append path;
- mix worker conversation Threads with Scratchpad identity;
- introduce model substitution or unrelated Deep Research behavioural changes;
- introduce Redis Streams/Lua unless ordinary Redis list semantics prove inadequate;
- broaden this work into unrelated auth, packaging, or dependency refactors.

---

## Completion Definition

Scratchpad tenantization is complete only when the following are all true:

```text
ORM resource identity             PASS
released ORM verified             PASS
Common schemas                    PASS
canonical scratchpad ID           PASS
Core MySQL migration              PASS
authenticated ownership checks    PASS
cross-tenant isolation            PASS
first-class REST API              PASS
first-class SDK client            PASS
tenant-scoped Redis keys          PASS
atomic append ledger              PASS
Deep Research backfill            PASS
Scratchpad deletion cleanup       PASS
Thread deletion cleanup           PASS
User deletion cleanup             PASS
no orphaned Redis data            PASS
concurrency tests                 PASS
tenant isolation tests            PASS
```

At that point, Scratchpad is no longer a Deep Research implementation detail. It is a reusable, tenant-safe Project David working-memory resource suitable for Deep Research, Career Agent, and future multi-agent workflows.
"""

path = Path("/mnt/data/SCRATCHPAD_TENANTIZATION_OUTSTANDING.md")
path.write_text(content, encoding="utf-8")
print(path)
