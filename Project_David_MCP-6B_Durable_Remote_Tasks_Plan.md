# Project David MCP-6B — Durable Remote Tasks Plan

**Status:** Future / Blocked
**Milestone:** MCP-6B
**Depends on:** MCP-5 registration/trust management plane, MCP-6A long-running synchronous MCP execution
**Primary blocker:** Official Python MCP SDK support for the `io.modelcontextprotocol/tasks` extension
**Scope:** Project David Core only
**Out of scope:** Q desktop changes, custom/hand-rolled Tasks JSON-RPC, replacing the existing Run/Action lifecycle

---

## 1. Purpose

MCP-6B adds durable support for remote MCP Tasks once the official Python MCP SDK exposes the Tasks extension.

The goal is to let Project David start an MCP operation that outlives the initiating `tools/call` request, persist the relationship between the Project David Action and the remote MCP task, recover that relationship after process restarts, and continue driving the existing Project David Run/Action lifecycle until the remote task reaches a terminal state.

MCP-6B must extend the existing orchestration model rather than introduce a competing task system.

The intended hierarchy is:

```text
Project David Run
    └── Project David Action
            └── Remote MCP Task
```

Project David remains the authoritative local lifecycle.

The remote MCP server remains authoritative only for the state of its own remote task.

---

## 2. Why MCP-6B Exists

MCP-6A solves long-running synchronous MCP calls:

```text
tools/call
    ├── progress notifications
    ├── explicit execution timeout
    ├── cancellation propagation
    └── final ToolResultEnvelope
```

That model still assumes the active connection remains responsible for the operation.

MCP Tasks solve a different problem:

- the remote operation can survive the original request,
- the remote server can return a durable task identifier,
- Project David can poll or resume later,
- cancellation can target the durable remote task,
- process restarts do not necessarily lose execution state,
- completion is retrieved separately from task creation.

MCP-6B therefore introduces durable **task linkage and recovery**, not merely a longer timeout.

---

## 3. Protocol Boundary

MCP-6B must use the official MCP Tasks extension exposed by the official Python SDK.

Expected extension namespace:

```text
io.modelcontextprotocol/tasks
```

Expected protocol concepts include:

```text
tasks/get
tasks/update
tasks/cancel
```

with task state similar to:

```text
working
input_required
completed
failed
cancelled
```

and support for:

- durable task identifiers,
- eventual result retrieval,
- task TTL / expiry,
- polling,
- cancellation,
- resumable status inspection.

### Hard rule

**Do not hand-roll the Tasks extension.**

If the official SDK does not expose a stable client API for the extension, MCP-6B remains blocked.

Wire-level compatibility must remain owned by the official MCP SDK.

---

## 4. Preconditions for Unblocking MCP-6B

MCP-6B may begin only when all of the following are true:

1. The installed official Python MCP SDK exposes the Tasks extension.
2. The SDK supports creation or receipt of a durable remote task identifier.
3. The SDK supports task inspection.
4. The SDK supports task cancellation.
5. The SDK exposes task result retrieval.
6. The extension API is documented sufficiently to test against a real or reference MCP server.
7. The extension is no longer dependent on the older removed core Tasks model.

The first implementation step must be a read-only SDK capability audit.

No database migration should be written before that audit determines the actual stable SDK types and method signatures.

---

## 5. Existing Project David Invariants

MCP-6B must not replace or bypass:

```text
OrchestratorCore
Run model
Action model
recursive Level-3 loop
ConsumerToolHandlersMixin
NativeExecutionService
provider-native tools
tool_call_id semantics
assistant/thread context
delegation architecture
code interpreter architecture
```

The existing Run and Action lifecycle remains authoritative.

MCP task state is subordinate execution metadata.

---

## 6. Proposed Persistence Model

If the SDK audit confirms durable remote Tasks, introduce a narrow table dedicated to the Action-to-MCP-task relationship.

Suggested name:

```text
mcp_action_tasks
```

Suggested logical shape:

```text
mcp_action_tasks
----------------
id
action_id
registration_id
remote_task_id
remote_task_status
remote_result_type
expires_at
last_polled_at
created_at
updated_at
```

Recommended constraints:

```text
UNIQUE(action_id)
UNIQUE(registration_id, remote_task_id)
```

Possible additional fields, only if justified by the SDK:

```text
request_state
task_ttl_seconds
last_error
input_required_payload
remote_metadata
```

Do not add speculative columns merely because the draft protocol contains them.

The migration should reflect the SDK and real server behaviour actually observed at implementation time.

---

## 7. Why `action_id` Should Be Unique

A Project David Action represents one logical tool execution.

Therefore:

```text
1 Project David Action
    =
0 or 1 durable remote MCP Task
```

This keeps retries and recovery deterministic.

If an Action already has an associated durable remote task, Core must resume or inspect that task rather than blindly creating another remote task.

That is the foundation for idempotent crash recovery.

---

## 8. Lifecycle Mapping

The remote MCP task state must **not** become the Project David Run state directly.

Suggested mapping:

| Remote MCP task | Project David Action | Notes |
|---|---|---|
| `working` | `pending` / active existing state | Continue polling |
| `input_required` | remains active | Special handling required; do not invent a global Run status |
| `completed` | depends on retrieved result | Completion alone is insufficient |
| `failed` | `failed` | Persist tool failure using existing path |
| `cancelled` | `cancelled` | Project David Run may already be cancelled |
| expired / missing | `failed` or explicit recovery failure | Must be deterministic |

### Critical result rule

A remote task reaching:

```text
completed
```

does **not** automatically imply:

```text
Action = completed
```

The eventual MCP tool result must still be inspected.

For example, a completed remote task may yield:

```text
CallToolResult(isError=True)
```

In that case:

```text
Remote task = completed
Project David Action = failed
```

MCP-4 `ToolResultEnvelope` semantics remain authoritative for the final tool result.

---

## 9. `input_required`

`input_required` is the largest lifecycle complication in MCP-6B.

Do **not** add:

```text
StatusEnum.input_required
```

globally merely to mirror MCP.

Project David should instead represent remote input requirements as MCP-specific execution state attached to the Action/task relationship.

Potential future behaviour:

```text
remote task → input_required
    ↓
record MCP-specific request payload
    ↓
surface requirement through an existing Project David interaction mechanism
    ↓
collect user/agent response
    ↓
submit continuation through official MCP Tasks API
```

The exact mechanism should be designed only after the official SDK exposes the real continuation contract.

For the first MCP-6B implementation, it may be acceptable to detect `input_required` and fail safely with an explicit unsupported-state error if continuation is not yet implemented.

---

## 10. Task Creation Flow

Target flow:

```text
LLM emits tool call
    ↓
Project David creates Action
    ↓
Run → pending_action
    ↓
MCP executor invokes remote tool using official Tasks-enabled SDK
    ↓
server returns durable remote task id
    ↓
Core persists mcp_action_tasks row
    ↓
poll / observe remote task
    ↓
terminal remote state
    ↓
retrieve eventual result
    ↓
adapt via existing MCP-4 ToolResultEnvelope
    ↓
existing Project David submit_tool_result()
    ↓
Action completed / failed
```

No second Action should be created for polling, result retrieval, or cancellation.

---

## 11. Crash Recovery

Crash recovery is a primary reason for MCP-6B.

On restart, Project David must be able to discover Actions with non-terminal remote MCP tasks.

Recovery procedure:

```text
find mcp_action_tasks rows
where associated Action is non-terminal
    ↓
load registration
    ↓
verify registration still exists and remains authorised
    ↓
connect using current registration transport/auth policy
    ↓
tasks/get remote_task_id
    ↓
reconcile remote state
    ↓
resume polling / retrieve result / mark terminal
```

Recovery must be idempotent.

The same remote task must never create duplicate tool messages or duplicate Action completion.

---

## 12. Idempotency Rules

MCP-6B must guarantee:

1. One Action cannot accidentally start multiple remote tasks because of a process retry.
2. The same remote result cannot be submitted twice.
3. Repeated polling is safe.
4. Repeated cancellation is safe.
5. Restart recovery does not duplicate tool output.
6. Remote task disappearance is handled deterministically.
7. A stale worker cannot overwrite a newer terminal local state.

Where possible, use database constraints rather than relying purely on in-memory guards.

---

## 13. Cancellation

Project David remains cancellation authority for the local Run.

Target cancellation flow:

```text
POST /runs/{run_id}/cancel
    ↓
Run = cancelled
    ↓
active MCP-6B supervisor observes cancellation
    ↓
official SDK tasks/cancel(remote_task_id)
    ↓
best-effort remote acknowledgement
    ↓
Action = cancelled
```

Important distinction:

Project David cancellation must not depend on the remote server acknowledging cancellation.

Once the local Run is authoritatively cancelled:

```text
Project David Run = cancelled
```

even if:

```text
remote tasks/cancel fails
```

The failure should be logged/audited and may require cleanup, but it must not resurrect the Run.

---

## 14. Completion / Cancellation Race

MCP-6A established the desired race rule:

> If Project David cancellation is already committed when remote completion is observed, cancellation wins.

MCP-6B must preserve the same rule.

Target reconciliation:

```text
remote task terminal result observed
    ↓
re-read authoritative Project David Run
    ↓
Run cancelled?
    YES → suppress tool result
           Action = cancelled
    NO  → process eventual ToolResultEnvelope
```

A remote completion that occurred physically earlier is not sufficient if Project David has already committed cancellation before the local completion transaction is accepted.

---

## 15. Polling

Polling policy must be bounded and configurable.

Suggested initial behaviour:

```text
fast initial polling
    ↓
bounded backoff
    ↓
maximum interval cap
```

Example concept only:

```text
1s
2s
3s
5s
5s
5s ...
```

Do not hard-code a sophisticated adaptive scheduler in the first implementation.

The SDK's own recommended polling hints or retry metadata should take precedence if exposed.

---

## 16. TTL and Expiry

Remote task TTL must be treated as protocol state, not as a Project David Run timeout.

If the remote task expires before result retrieval:

```text
Action → failed
```

with an explicit MCP task-expiry error.

The corresponding tool output should clearly distinguish:

```text
remote execution failure
```

from:

```text
remote durable task expired before result retrieval
```

Expiry must never be silently treated as successful completion.

---

## 17. Timeout Policy

MCP-6A already distinguishes ordinary synchronous request deadlines from generic transport failures.

MCP-6B introduces a separate concept:

```text
request timeout
≠
remote task TTL
≠
Project David execution deadline
```

These must remain distinct.

Recommended terminology:

```text
request_timeout_seconds
remote_task_ttl
execution_deadline
```

Do not overload the existing registration `timeout_seconds` to mean remote task lifetime.

---

## 18. Progress

MCP-6B should consume progress exposed by the official Tasks API if available.

Progress remains primarily ephemeral.

Possible destinations:

```text
logs
SSE run events
Redis execution stream
future Action progress surface
```

Do not add progress columns to the Action table unless a broader Project David progress model is deliberately adopted.

The durable state required by MCP-6B is the remote task relationship, not every progress update.

---

## 19. Registration and Trust

Remote tasks must remain bound to the MCP registration that created them.

The persisted row should reference:

```text
registration_id
```

rather than only storing a URL.

This preserves:

- stable identity,
- ownership,
- future credential policy,
- endpoint updates,
- auditability.

At recovery time, Core must re-authorise access through current Project David registration rules.

Persisting a remote task id must never imply permanent authorisation to use the registration.

---

## 20. Credentials

MCP-6B must not introduce raw credential persistence.

Existing rule remains:

```text
credential-bearing URLs are rejected
```

and caller-owned authenticated HTTP clients or the future approved credential mechanism remain responsible for secrets.

A remote task identifier should be treated as potentially sensitive execution metadata and must not be exposed outside authorised users.

---

## 21. Suggested Core Components

Possible implementation split:

```text
mcp_remote_client.py
    Tasks-extension SDK wrappers only

mcp_task_service.py
    durable Action ↔ remote-task persistence
    idempotent state reconciliation

mcp_tool_executor.py
    detects durable-task response
    returns control to task supervisor

consumer_tool_handlers_mixin.py
    existing Action/Run integration
    no competing lifecycle

mcp_task_supervisor.py
    poll
    cancellation
    result retrieval
    crash recovery
```

Names are provisional.

Prefer the smallest number of new abstractions that keep protocol, persistence, and orchestration responsibilities separate.

---

## 22. Proposed Service Responsibilities

### `RemoteMcpClient`

Own only official SDK protocol calls:

```text
start / receive durable task
get task
cancel task
update task
retrieve result
```

No database access.

### `McpTaskService`

Own:

```text
create linkage
retrieve linkage
update observed task state
find recoverable tasks
terminal reconciliation guards
```

No transport implementation.

### Task Supervisor

Own:

```text
poll loop
Run cancellation observation
remote cancellation call
eventual result retrieval
recovery coordination
```

No direct SQL outside service boundaries.

---

## 23. Startup Recovery

A future application-start recovery hook may:

```text
query recoverable MCP task rows
    ↓
spawn bounded supervisors
```

Important constraints:

- recovery concurrency must be bounded,
- one remote task must have one active local supervisor,
- startup must not block indefinitely,
- unreachable MCP servers should enter retry/failure policy rather than freeze application startup.

A distributed deployment may eventually require a lease/claim mechanism so multiple API instances do not supervise the same task simultaneously.

That should be added only if deployment topology requires it.

---

## 24. Distributed Execution Consideration

If Project David Core runs with multiple API workers or replicas, in-memory ownership is insufficient.

Potential future mechanisms:

```text
database lease
Redis lease
SELECT ... FOR UPDATE SKIP LOCKED
worker ownership token
```

MCP-6B should first audit current deployment semantics before choosing one.

Do not add distributed coordination machinery merely for theoretical scale.

---

## 25. Observability

Each durable MCP task should be traceable using:

```text
run_id
action_id
registration_id
remote_task_id
tool_call_id
provider_name
```

Logs should make state transitions explicit:

```text
MCP_TASK_CREATED
MCP_TASK_WORKING
MCP_TASK_INPUT_REQUIRED
MCP_TASK_COMPLETED
MCP_TASK_FAILED
MCP_TASK_CANCEL_REQUESTED
MCP_TASK_CANCELLED
MCP_TASK_EXPIRED
MCP_TASK_RECOVERED
```

Avoid logging secrets or raw credentials.

---

## 26. Failure Cases to Test

At minimum:

### Creation

- remote server immediately returns ordinary result instead of Task
- remote server returns durable Task
- connection fails before task id is received
- task id received but local persistence fails

### Polling

- working → completed
- working → failed
- working → cancelled
- working → input_required
- task expires
- server temporarily unavailable
- malformed task state
- unknown task id

### Result

- completed task → successful `CallToolResult`
- completed task → `isError=true`
- completed task → rich structured result
- result retrieval fails after completed status
- duplicate result retrieval

### Cancellation

- local Run cancelled while remote working
- remote cancel succeeds
- remote cancel fails
- remote completes concurrently with local cancellation
- repeated cancel request

### Recovery

- process restart while working
- restart after remote completion but before local result persistence
- restart after result persistence but before local Action finalisation
- registration disabled/deleted during recovery
- remote task missing after restart
- two recovery workers race

---

## 27. Test Strategy

MCP-6B should have four test layers.

### Unit

```text
state mapping
idempotency
persistence
timeout/TTL distinction
cancellation reconciliation
```

### Fake SDK / protocol boundary

Use SDK-compatible fakes that model:

```text
task creation
task polling
cancel
eventual result
input_required
```

### Integration

Run against a local MCP server using the official SDK.

Exercise at least:

```text
create → working → completed
create → cancel
create → process restart → recover
```

### Regression

All existing:

```text
Tool ABI
MCP-1 transport
MCP-2 discovery
MCP-3 execution
MCP-4 rich results
MCP-5 management plane
MCP-6A long-running synchronous execution
```

must remain green.

---

## 28. Migration Gate

Do not create the migration until the official SDK audit is complete.

Before migration:

```text
SDK_TASKS_EXTENSION=PASS
TASK_ID_CONTRACT=KNOWN
TASK_STATUS_ENUM=KNOWN
RESULT_RETRIEVAL_CONTRACT=KNOWN
CANCEL_CONTRACT=KNOWN
TTL_CONTRACT=KNOWN
```

Only then freeze the persistence schema.

---

## 29. Suggested Implementation Sequence

### MCP-6B.0 — SDK capability audit

Read-only.

Confirm official API signatures and extension types.

### MCP-6B.1 — Task persistence foundation

Add the minimum durable Action ↔ Task model and service.

No orchestration integration yet.

### MCP-6B.2 — Remote Tasks client adapter

Expose official SDK Task operations through `RemoteMcpClient`.

### MCP-6B.3 — Durable task creation

Allow an MCP execution to persist its remote task id.

### MCP-6B.4 — Polling and eventual result

Drive remote task to terminal state and route result through MCP-4.

### MCP-6B.5 — Cancellation

Propagate Project David Run cancellation to remote durable Tasks.

### MCP-6B.6 — Crash recovery

Recover non-terminal tasks after process restart.

### MCP-6B.7 — `input_required`

Implement only after the real SDK continuation contract is known.

### MCP-6B.8 — hardening

Race tests, expiry, retries, observability, distributed ownership if required.

---

## 30. Explicit Non-Goals

MCP-6B does **not**:

- replace Project David Actions,
- replace Project David Runs,
- create a generic distributed job framework,
- persist arbitrary MCP credentials,
- add a global `input_required` Run status solely for MCP,
- change provider-native tools,
- change `tool_call_id` semantics,
- redesign delegation,
- redesign code interpreter execution,
- implement MCP Tasks without official SDK support,
- modify Q desktop.

---

## 31. Exit Criteria

MCP-6B is complete when:

```text
[ ] official SDK Tasks extension is used
[ ] no hand-rolled Tasks protocol exists
[ ] remote task id is durably linked to one Action
[ ] task creation is idempotent
[ ] working tasks survive Core restart
[ ] remote state can be polled/recovered
[ ] eventual results route through ToolResultEnvelope
[ ] completed + isError=true maps to failed Action
[ ] local Run cancellation cancels remote task best-effort
[ ] local cancellation remains authoritative
[ ] completion/cancellation race is deterministic
[ ] remote task expiry is explicit
[ ] duplicate result submission is prevented
[ ] registration ownership is enforced during recovery
[ ] no raw credentials are persisted
[ ] existing MCP regression suite remains green
[ ] no unrelated orchestration architecture is changed
```

---

## 32. Decision Record

The deliberate split between MCP-6A and MCP-6B is architectural:

```text
MCP-6A
    long-running active call
    progress
    deadline
    cancellation
    existing Action lifecycle

MCP-6B
    durable remote task identity
    polling
    crash recovery
    eventual result
    durable remote cancellation
```

MCP-6A should not be expanded into an imitation of durable Tasks.

MCP-6B should begin only when the official SDK makes the Tasks extension a stable protocol surface.

Until then:

```text
MCP-6B = BLOCKED BY DESIGN
```

That is preferable to creating a proprietary compatibility layer that Project David would later have to remove.
