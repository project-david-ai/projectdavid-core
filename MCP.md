# Managed MCP Architecture

## Purpose

Project David treats MCP as a managed extension of the existing assistant tool system.

The design does not expose a remote MCP server directly to an assistant. Instead, Project David owns the registration, discovery, naming, attachment, routing, execution lifecycle, and persistence boundaries around the remote server.

The result is a layered model:

```text
Remote MCP server
        |
        | MCP protocol
        v
Project David transport
        |
        | discovery adaptation
        v
Project David tool identity
        |
        | explicit assistant attachment
        v
Assistant tool configuration
        |
        | runtime binding
        v
Project David Tool ABI
        |
        | tools/call
        v
Remote MCP server
```

The central architectural rule is:

> MCP supplies remote capabilities. Project David remains the authority for assistant ownership, capability attachment, tool naming, Run and Action lifecycle, cancellation, persistence, and model-facing tool configuration.

---

## Architecture at a glance

```text
                         MANAGEMENT PLANE
                         ================

 Client / SDK
      |
      | authenticated API
      v
+---------------------------+
| routers/mcp_router.py     |
+-------------+-------------+
              |
              v
+-------------------------------+
| McpRegistrationService        |
| services/                     |
| mcp_registration_service.py   |
+----+----------------------+---+
     |                      |
     |                      |
     v                      v
McpServerRegistration   AssistantMcpTool
     |                      |
     |                      |
     +----------+-----------+
                |
                v
        Assistant.tool_configs
                |
                | cache invalidation
                v
        assistant runtime state


                          DISCOVERY PLANE
                          ===============

 McpRegistrationService
          |
          v
+----------------------------+
| RemoteMcpClient            |
| mcp_remote_client.py       |
+-------------+--------------+
              |
              | MCP tools/list
              v
       Remote MCP server
              |
              | official MCP SDK types
              v
+----------------------------+
| mcp_tool_discovery.py      |
|                            |
| remote_name                |
| canonical_id               |
| provider_name              |
| ToolDefinition             |
+-------------+--------------+
              |
              v
     Project David tool model


                           EXECUTION PLANE
                           ===============

 Model emits provider_name
          |
          v
+---------------------------------------+
| ConsumerToolHandlersMixin             |
| consumer_tool_handlers_mixin.py       |
+-------------------+-------------------+
                    |
                    | ephemeral lookup
                    v
             McpToolExecutor
                    |
                    | ToolCallEnvelope
                    v
+--------------------------------+
| mcp_tool_executor.py           |
+---------------+----------------+
                |
                | remote_name + arguments
                v
+--------------------------------+
| RemoteMcpClient                |
+---------------+----------------+
                |
                | MCP tools/call
                v
         Remote MCP server
                |
                | CallToolResult
                v
         ToolResultEnvelope
                |
                v
       Project David lifecycle
```

---

# 1. Managed server registration

A remote MCP endpoint first becomes a Project David managed registration.

The registration service owns:

- URL validation and normalization
- user ownership
- stable registration identity
- transport identity
- timeout configuration
- enabled/disabled state
- CRUD lifecycle

The canonical service is:

```text
src/api/entities_api/services/mcp_registration_service.py
```

Registration identity is not based on the mutable display name.

The service normalizes the endpoint URL, then derives an identity key from:

```text
transport + normalized_url
```

Conceptually:

```text
user
 |
 +-- MCP registration
       |
       +-- id                stable Project David registration id
       +-- name              mutable display / namespace name
       +-- normalized_url
       +-- identity_key
       +-- transport
       +-- timeout_seconds
       +-- enabled
```

The persistent registration ID is deliberately used as the stable server identity during discovery.

This matters because a display name can change without changing the identity of the remote server registration.

---

# 2. Registration ownership boundary

MCP registrations are user-owned resources.

The management service resolves registrations with both:

```text
server_id
owner_id
```

An assistant is also ownership checked before MCP tools can be attached or detached.

This means an MCP endpoint is not a global capability merely because it exists in the database. The authenticated user must own the registration and must be allowed to modify the target assistant.

Disabled registrations remain persisted, but discovery and attachment paths can require the registration to be enabled.

```text
Authenticated user
       |
       +---- owns ----> McpServerRegistration
       |
       +---- may modify ----> Assistant
                                  |
                                  +---- MCP attachments
```

---

# 3. Discovery is not attachment

Discovery and attachment are separate operations.

Discovery asks:

```text
What tools does this registered remote MCP server currently advertise?
```

Attachment asks:

```text
Which of those tools should this assistant actually receive?
```

They are intentionally not the same operation.

A single MCP server may expose many unrelated capabilities. Project David therefore does not assume that every tool advertised by one server should be attached to an assistant.

```text
Remote server
   |
   | tools/list
   v
+----------------------+
| discovered tools     |
|                      |
| search.issues        |
| get.issue            |
| create.issue         |
| delete.repository    |
+----------+-----------+
           |
           | explicit selection
           v
+----------------------+
| assistant attachment |
|                      |
| search.issues        |
| get.issue            |
+----------------------+
```

The Core discovery endpoint returns discovery data. The SDK adds ergonomic collection and selection behavior on top of this boundary.

---

# 4. Remote transport boundary

The remote protocol boundary lives in:

```text
src/api/entities_api/orchestration/mcp_remote_client.py
```

`RemoteMcpClient` is built on the official MCP Python SDK and Streamable HTTP transport.

Its responsibilities are intentionally narrow:

```text
connect
initialize MCP session
expose negotiated server metadata
tools/list
tools/call
transport lifetime
```

It is not responsible for:

```text
assistant policy
tool persistence
provider alias generation
assistant attachment
Run lifecycle
Action lifecycle
routing policy
```

The client validates remote URLs as absolute HTTP or HTTPS endpoints.

Authentication headers, TLS customization, and similar transport configuration can be supplied through a caller-owned HTTP client. The remote client itself does not need to become the persistence layer for raw credentials.

This separation keeps protocol transport independent from Project David capability policy.

---

# 5. Tool discovery adaptation

Remote MCP tools use MCP-native names and schemas.

Project David adapts them before they enter the internal tool system.

The adapter lives in:

```text
src/api/entities_api/orchestration/mcp_tool_discovery.py
```

The important internal representation is `McpDiscoveredTool`.

Conceptually:

```text
MCP Tool
   |
   |  name
   |  description
   |  inputSchema
   v
+--------------------------------+
| McpDiscoveredTool              |
|                                |
| server_id                      |
| remote_name                    |
| canonical_id                   |
| provider_name                  |
| definition                     |
+--------------------------------+
```

This creates four distinct identity concepts.

## `server_id`

The stable Project David registration ID for the remote MCP server.

## `remote_name`

The name advertised by the remote MCP server.

Example:

```text
search.issues
```

This is the name sent back to the remote server during `tools/call`.

## `canonical_id`

A stable Project David identity for the logical tool.

It separates tool identity from the provider-facing alias used by the model.

## `provider_name`

The name exposed inside Project David's model-facing function tool namespace.

Example shape:

```text
github__search_issues
```

The discovery adapter keeps readable aliases where possible, but can add deterministic disambiguation when normalization would collide, a name is already reserved, or the provider-facing name would exceed its constraints.

## `definition`

The Project David tool definition derived from the MCP schema.

This is what eventually becomes part of the assistant's model-facing tool configuration.

---

# 6. Why `remote_name` and `provider_name` are different

The model does not need to see the remote MCP name directly.

The remote server does not need to know Project David's provider alias.

```text
Model-facing namespace                  Remote MCP namespace

github__search_issues   ----------->    search.issues
        ^
        |
   provider_name                    remote_name
```

This gives Project David a controlled namespace where MCP tools can coexist with:

- built-in tools
- developer-defined function tools
- other MCP servers
- existing orchestration tools

It also allows Project David to reject alias collisions before execution.

---

# 7. Assistant attachment is durable provenance

Attaching a discovered MCP tool does two related things.

First, it creates or updates durable MCP provenance:

```text
AssistantMcpTool
```

Second, it injects the corresponding function definition into:

```text
Assistant.tool_configs
```

The service coordinating both sides is:

```text
src/api/entities_api/services/mcp_registration_service.py
```

The helper functions that keep `tool_configs` coherent live in:

```text
src/api/entities_api/services/mcp_tool_config.py
```

The relationship is:

```text
McpServerRegistration
          |
          | registration_id
          v
   AssistantMcpTool
          |
          | assistant_id
          v
       Assistant
          |
          +---- tool_configs
```

An `AssistantMcpTool` row carries durable provenance such as:

```text
assistant_id
registration_id
remote_name
canonical_id
provider_name
enabled
```

The assistant's `tool_configs` holds the model-facing function definition.

These two representations serve different purposes:

```text
AssistantMcpTool
    = provenance and durable MCP attachment identity

Assistant.tool_configs
    = model-facing capability schema
```

They must remain coherent.

---

# 8. Provider aliases are durable once attached

Discovery may calculate a provider alias for a tool.

Once an MCP tool has been attached to an assistant, the persisted provider alias is treated as durable for that attachment.

If discovery metadata is later refreshed, Core updates the tool definition while retaining the existing attachment alias.

This prevents a tool from unexpectedly changing its model-facing identity during normal metadata refresh.

```text
First attachment

remote_name:     search.issues
provider_name:   github__search_issues

Later discovery refresh

remote_name:     search.issues
new metadata:    ...
persisted alias: github__search_issues
```

This stability is important because tool names participate in model output, routing, persisted actions, and debugging.

---

# 9. Attachment validates current discovery

Attachment does not blindly persist arbitrary remote tool names.

Before attachment, Core discovers the server's current tools and verifies that every requested remote name is actually advertised.

```text
requested names
      |
      v
discover current server tools
      |
      v
+---------------------------+
| requested subset exists?  |
+-------------+-------------+
              |
        +-----+-----+
        |           |
       yes          no
        |           |
        v           v
     attach       reject
```

This keeps the management plane tied to current remote capability state.

The service also handles concurrent attachment conflicts and protects provider alias uniqueness.

---

# 10. Detachment removes both capability and provenance

Detachment reverses the assistant attachment.

Core removes:

```text
AssistantMcpTool rows
```

and removes the associated provider-facing function definitions from:

```text
Assistant.tool_configs
```

Detaching an already detached tool is treated as a no-op.

The operation is assistant-specific. It does not delete the underlying server registration.

```text
delete registration
    !=
detach tool from assistant
```

Deleting a registration is broader because all assistant attachments associated with that registration must be removed from the affected assistants.

---

# 11. Assistant cache coherence

MCP attachment state affects the assistant's effective tool configuration.

After MCP attachment, detachment, registration deletion, or relevant registration changes, Core invalidates the assistant cache.

The MCP registration service obtains the synchronous invalidator through:

```text
src/api/entities_api/utilities/cache_utils.py
```

Conceptually:

```text
MCP state mutation
       |
       v
database commit
       |
       v
assistant cache invalidation
       |
       v
next assistant load sees current tool state
```

Cache invalidation failure is logged, while the durable database mutation remains the source of truth.

---

# 12. Assistant service interaction

The assistant service is MCP-aware.

Relevant Core file:

```text
src/api/entities_api/services/assistants_service.py
```

It imports `AssistantMcpTool` and uses enabled MCP attachment provider names when working with assistant tool configuration.

This is one of the integration points between the generic assistant system and the MCP management plane.

MCP therefore extends the assistant capability model rather than replacing it.

---

# 13. Internal Tool ABI

MCP execution does not bypass Project David's existing orchestration model.

The common execution boundary lives in:

```text
src/api/entities_api/orchestration/tool_abi.py
```

Important abstractions include:

```text
ToolCallEnvelope
ToolResultEnvelope
ToolExecutor
```

The call envelope carries orchestration identity alongside tool arguments:

```text
name
arguments
run_id
thread_id
assistant_id
tool_call_id
```

Conceptually:

```text
provider-specific call
        |
        v
+------------------+
| ToolCallEnvelope |
+--------+---------+
         |
         v
   ToolExecutor
         |
         v
+--------------------+
| ToolResultEnvelope |
+--------------------+
```

The ABI prevents MCP protocol details from leaking through the whole orchestration stack.

---

# 14. MCP execution adapter

Remote MCP execution is owned by:

```text
src/api/entities_api/orchestration/mcp_tool_executor.py
```

`McpToolExecutor` binds:

```text
one discovered tool
+
one RemoteMcpClient factory
```

The executor accepts a `ToolCallEnvelope`.

It verifies that the incoming model-facing name matches the discovered tool's `provider_name`, then translates execution back to the remote MCP namespace:

```text
ToolCallEnvelope.name
        |
        | provider_name
        v
McpToolExecutor
        |
        | remote_name
        v
RemoteMcpClient.call_tool(...)
```

Example:

```text
model emits:
    github__search_issues

executor maps to:
    search.issues

remote call:
    tools/call("search.issues", arguments)
```

Transport or protocol failures are converted into tool failures rather than escaping uncontrolled into the orchestration loop.

---

# 15. Runtime executor bindings are ephemeral

Durable MCP attachment and runtime execution binding are intentionally different concepts.

The persisted attachment lives in the database:

```text
AssistantMcpTool
```

The runtime routing registry lives on an orchestrator instance:

```text
_mcp_tool_executors
```

The registry is managed in:

```text
src/api/entities_api/orchestration/mixins/consumer_tool_handlers_mixin.py
```

The mixin exposes:

```text
bind_mcp_tool_executor(...)
unbind_mcp_tool_executor(...)
_get_mcp_tool_executor(...)
```

and keys the registry by `provider_name`.

```text
Durable state                         Runtime state

AssistantMcpTool                     orchestrator instance
      |                                     |
      | provider_name                       |
      +------------------------------> _mcp_tool_executors
                                             |
                                             +-- alias -> McpToolExecutor
```

The runtime registry also rejects collisions with reserved internal routed tool names.

The source material reviewed for this document shows the binding registry and its execution behavior. It does not include the higher-level composition site that reconstructs those ephemeral bindings from durable attachment records, so that composition step is deliberately not assigned to a specific file here.

---

# 16. Tool routing

MCP tools participate in the existing consumer tool handling path.

Relevant file:

```text
src/api/entities_api/orchestration/mixins/consumer_tool_handlers_mixin.py
```

Routing checks whether a model-emitted tool name has a bound MCP executor.

```text
model tool call
      |
      v
provider_name
      |
      v
+----------------------------+
| MCP executor bound?        |
+-------------+--------------+
              |
       +------+------+
       |             |
      yes            no
       |             |
       v             v
 MCP executor     existing
 execution       consumer/tool path
```

This is important because MCP is additive.

Unbound tool names continue through the pre-existing consumer handoff behavior.

---

# 17. Reserved internal names

MCP provider aliases may not shadow Project David internal routed tools.

The consumer tool handler maintains a reserved-name set containing platform capabilities such as internal execution, web, delegation, and scratchpad tools.

When an MCP executor is bound:

```text
provider_name collides with internal routed name
                    |
                    v
                  reject
```

This protects routing determinism.

The discovery layer also attempts to avoid provider-facing collisions earlier in the lifecycle.

---

# 18. Rich MCP results

MCP results can contain more than text.

Current Core preserves rich result data in `ToolResultEnvelope` while also maintaining the legacy string projection required by existing durable tool-output paths.

Relevant files:

```text
src/api/entities_api/orchestration/mcp_tool_executor.py
src/api/entities_api/orchestration/tool_abi.py
```

The result model can preserve:

```text
content
structured_content
content_blocks
metadata
is_error
```

MCP-specific metadata can include fields derived from the remote result, such as:

```text
mcp_meta
mcp_result_type
```

Conceptually:

```text
MCP CallToolResult
        |
        v
+-----------------------------+
| McpToolExecutor._adapt_result|
+--------------+--------------+
               |
               v
+-----------------------------+
| ToolResultEnvelope          |
|                             |
| content                     |
| structured_content          |
| content_blocks              |
| metadata                    |
| is_error                    |
+--------------+--------------+
               |
               +---- rich representation
               |
               +---- legacy string projection
```

This keeps MCP fidelity without forcing the entire older persistence pipeline to change at once.

---

# 19. Long-running synchronous MCP calls

Project David remains lifecycle authority while an MCP `tools/call` is running.

Long-running execution support spans:

```text
src/api/entities_api/orchestration/mcp_remote_client.py
src/api/entities_api/orchestration/mcp_tool_executor.py
src/api/entities_api/orchestration/mixins/consumer_tool_handlers_mixin.py
```

The remote client can forward MCP progress notifications.

The executor forwards the progress callback.

The consumer tool handler runs the remote call in its own asyncio task and continues checking the authoritative Project David Run state.

```text
                Project David Run
                       |
                       | authoritative status
                       v
model call --> asyncio MCP task --> remote tools/call
                    |
                    +---- progress callback
                    |
                    +---- periodic Run cancellation check
```

If the Project David Run becomes cancelled:

```text
cancel remote execution task
        |
        v
mark Action cancelled
        |
        v
do not persist synthetic tool output
```

This preserves the rule that the remote MCP server does not own the Project David Run lifecycle.

---

# 20. MCP progress is observational

MCP progress notifications are passed through the transport and executor boundary.

Core currently treats progress as execution telemetry rather than as a separate durable MCP task resource.

This distinction matters:

```text
long-running tools/call
    !=
durable MCP Tasks protocol resource
```

The managed architecture documented here is based on normal MCP discovery and synchronous `tools/call`, with Project David Run cancellation layered around it.

---

# 21. Management API surface

The MCP router is:

```text
src/api/entities_api/routers/mcp_router.py
```

It exposes the managed registration and assistant attachment surfaces.

Conceptual route groups:

```text
MCP registrations

POST    /mcp/servers
GET     /mcp/servers
GET     /mcp/servers/{server_id}
PATCH   /mcp/servers/{server_id}
DELETE  /mcp/servers/{server_id}


Discovery

GET     /mcp/servers/{server_id}/tools


Assistant attachments

GET     /assistants/{assistant_id}/mcp-tools
POST    /assistants/{assistant_id}/mcp-tools
DELETE  /assistants/{assistant_id}/mcp-tools
```

The router remains thin.

It authenticates the request, accepts validated contracts, and delegates policy and persistence to `McpRegistrationService`.

---

# 22. State model

The architecture has three categories of state.

## Registration state

```text
McpServerRegistration
```

Durable description of a user-owned remote server endpoint.

## Attachment state

```text
AssistantMcpTool
Assistant.tool_configs
```

Durable description of which remote tools an assistant may expose, plus the model-facing definitions.

## Runtime state

```text
McpToolExecutor
_mcp_tool_executors registry
active RemoteMcpClient session
async execution task
progress callback
```

Ephemeral state used while an orchestrator instance is running.

```text
+----------------------+        +-----------------------+
| Durable DB state     |        | Ephemeral runtime     |
|                      |        |                       |
| McpServerRegistration|------->| RemoteMcpClient       |
| AssistantMcpTool     |        | McpToolExecutor       |
| Assistant.tool_configs|------>| executor registry     |
+----------------------+        | asyncio task          |
                                +-----------------------+
```

---

# 23. Identity model

The managed design uses multiple identity layers on purpose.

```text
registration.id
    |
    | stable remote server registration
    v
server_id

remote MCP server
    |
    | protocol-advertised name
    v
remote_name

Project David
    |
    | stable logical identity
    v
canonical_id

model-facing namespace
    |
    | collision-safe alias
    v
provider_name
```

These should not be collapsed into one string.

They answer different questions:

| Field | Meaning |
|---|---|
| `server_id` | Which managed server registration owns this tool? |
| `remote_name` | What name must be sent to the remote MCP server? |
| `canonical_id` | What is Project David's stable logical identity for the tool? |
| `provider_name` | What function name does the model and router see? |

---

# 24. Capability model

The architecture distinguishes availability from authorization.

```text
remote server advertises tool
             |
             v
        discoverable
             |
             | explicit attachment
             v
   assistant capability
             |
             | runtime binding
             v
         executable
```

A tool being discoverable does not automatically make it part of any assistant.

A tool being attached durably does not mean the remote endpoint is always reachable.

A tool being model-visible still executes under Project David's Run, Action, and error handling rules.

---

# 25. Source ownership map

## MCP-specific Core files

### `src/api/entities_api/routers/mcp_router.py`

Owns:

- authenticated MCP management API
- registration endpoints
- discovery endpoint
- assistant MCP attachment endpoints

Depends on:

- `McpRegistrationService`
- auth and DB dependencies
- shared validation contracts

---

### `src/api/entities_api/services/mcp_registration_service.py`

Owns:

- registration identity
- URL normalization
- ownership checks
- registration CRUD
- discovery orchestration
- discovery pagination for attachment
- attachment validation
- durable `AssistantMcpTool` persistence
- `Assistant.tool_configs` synchronization
- detachment
- registration deletion cleanup
- assistant cache invalidation

This is the center of the managed MCP control plane.

---

### `src/api/entities_api/services/mcp_tool_config.py`

Owns helper logic for:

- reading function tool names
- upserting model-facing function tools
- removing function tools

Used to keep assistant tool configuration coherent with MCP provenance.

---

### `src/api/entities_api/orchestration/mcp_remote_client.py`

Owns:

- official MCP SDK client integration
- Streamable HTTP session lifecycle
- endpoint validation
- server handshake metadata
- `tools/list`
- `tools/call`
- progress callback forwarding

This is the protocol transport boundary.

---

### `src/api/entities_api/orchestration/mcp_tool_discovery.py`

Owns:

- MCP Tool to Project David tool adaptation
- `McpDiscoveredTool`
- discovery page representation
- canonical IDs
- provider-safe aliases
- collision handling
- Project David `ToolDefinition` creation

This is the identity and schema adaptation boundary.

---

### `src/api/entities_api/orchestration/mcp_tool_executor.py`

Owns:

- execution of one discovered MCP tool
- provider-name validation
- mapping provider alias back to remote name
- remote `tools/call`
- MCP error adaptation
- rich `CallToolResult` preservation
- legacy result projection

This is the execution adapter boundary.

---

### `src/api/entities_api/orchestration/tool_abi.py`

Owns the provider-neutral internal execution contracts:

- `ToolCallEnvelope`
- `ToolResultEnvelope`
- `ToolExecutor`
- tool definition primitives

MCP uses this ABI rather than introducing a parallel orchestration lifecycle.

---

### `src/api/entities_api/orchestration/mixins/consumer_tool_handlers_mixin.py`

Owns runtime MCP routing integration:

- ephemeral executor registry
- bind/unbind operations
- reserved internal tool protection
- lookup by `provider_name`
- MCP execution handoff
- Run/Action integration
- long-running call cancellation
- progress callback handling
- fallback to existing consumer tool behavior

This is the main MCP-to-orchestrator integration point.

---

## Supporting Core files

### `src/api/entities_api/services/assistants_service.py`

MCP-aware assistant service integration.

Uses enabled MCP attachment provider names when reasoning about assistant tool configuration.

---

### `src/api/entities_api/utilities/cache_utils.py`

Supplies assistant cache invalidation used after MCP capability mutations.

---

### `src/api/entities_api/dependencies.py`

Supplies request-scoped API authentication and database dependencies used by the MCP router.

---

### `src/api/entities_api/models/models.py`

Contains the Core `Assistant` model used by the MCP registration service, including the assistant's `tool_configs` surface.

---

### `src/api/entities_api/routers/__init__.py`

Participates in router aggregation so the MCP management router becomes part of the API surface.

---

# 26. External package dependencies

The managed design also depends on contracts outside Core.

## `projectdavid_common`

Provides validation contracts exposed through `ValidationInterface`, including MCP registration and assistant attachment request/response models.

Examples include:

```text
McpServerRegistrationCreate
McpServerRegistrationRead
McpServerRegistrationUpdate
AssistantMcpToolsAttach
AssistantMcpToolsDetach
AssistantMcpToolRead
```

## `projectdavid_orm`

Provides durable persistence models:

```text
McpServerRegistration
AssistantMcpTool
```

Core owns the service policy around these models, while ORM owns their persistence definitions.

## Official MCP Python SDK

Provides protocol-native types and transport behavior used by `RemoteMcpClient`, discovery adaptation, and execution.

---

# 27. Architectural invariants

The current managed design depends on a small set of important invariants.

### Project David owns capability policy

Remote discovery never automatically grants assistant access.

### Project David owns lifecycle

Remote execution does not replace Project David Run and Action state.

### Registration identity is stable

Mutable registration names are not used as durable server identity.

### Remote names remain remote names

`remote_name` is preserved for protocol execution.

### Provider names are Project David names

`provider_name` belongs to the model-facing routing namespace.

### Attached aliases are durable

Metadata refresh does not casually rename an already attached capability.

### Tool configuration and provenance move together

`AssistantMcpTool` and `Assistant.tool_configs` must remain coherent.

### MCP transport stays narrow

`RemoteMcpClient` does not become the policy, persistence, or orchestration layer.

### Runtime bindings are ephemeral

The executor registry belongs to an orchestrator instance, not to durable persistence.

### Internal tool names cannot be shadowed

MCP aliases must not collide with reserved platform tools.

### Remote failures remain tool failures

Transport and server errors are adapted into the normal tool execution lifecycle.

### Cancellation authority stays local

Cancelling a Project David Run can interrupt a long-running MCP call.

---

# 28. Conceptual end-to-end flow

```text
1. REGISTER

User
 |
 v
POST /mcp/servers
 |
 v
McpRegistrationService
 |
 v
McpServerRegistration


2. DISCOVER

McpServerRegistration
 |
 v
RemoteMcpClient
 |
 | tools/list
 v
Remote MCP server
 |
 v
mcp_tool_discovery
 |
 v
McpDiscoveredTool[]


3. ATTACH

selected remote_name[]
 |
 v
McpRegistrationService
 |
 +---- validate against current discovery
 |
 +---- create/update AssistantMcpTool
 |
 +---- upsert definition into Assistant.tool_configs
 |
 +---- commit
 |
 +---- invalidate assistant cache


4. RUNTIME

Assistant capability
 |
 v
provider-facing definition
 |
 v
model emits provider_name
 |
 v
ConsumerToolHandlersMixin
 |
 v
McpToolExecutor
 |
 v
RemoteMcpClient
 |
 | tools/call(remote_name, arguments)
 v
Remote MCP server


5. RESULT

CallToolResult
 |
 v
McpToolExecutor
 |
 v
ToolResultEnvelope
 |
 +---- rich MCP data
 |
 +---- legacy string projection
 |
 v
Project David tool output / Action / Run lifecycle
```

---

# 29. Design summary

Project David's MCP implementation is not a thin proxy.

It is a managed capability system built around MCP as a remote protocol.

```text
MCP provides:
    server handshake
    tool discovery
    tool schemas
    tool execution
    protocol result types
    progress events

Project David provides:
    ownership
    registration
    stable identity
    provider naming
    explicit assistant attachment
    tool configuration
    routing
    Run lifecycle
    Action lifecycle
    cancellation
    persistence
    cache coherence
    compatibility with existing tools
```

That division is the core of the architecture.

It lets MCP capabilities participate in Project David without making the rest of the system MCP-specific.
