# Vector data-plane boundary

Core owns authenticated vector storage operations and internal Qdrant connectivity.
The SDK keeps document parsing, chunking, and optional local embedding computation.
No embedding implementation was moved into Core.

## Gateway and configuration

NGINX_CHANGE_REQUIRED=NO

Both Platform Nginx configurations route `/v1/vector-stores/...` through `location /`
to `api:9000`. Both compose configurations keep Qdrant on the internal network,
without published host ports, and supply `QDRANT_URL=http://qdrant:6333` to Core.
Platform source was not changed.

Core runtime and the soft-delete purge daemon use `QDRANT_URL`, defaulting to
`http://qdrant:6333`. Collection state comes from Qdrant, without `active_stores`.
The public SDK vector client uses `base_url`, `PROJECTDAVID_BASE_URL`, or legacy
`BASE_URL`, with API-key authentication. Explicit `vector_store_host` arguments
remain accepted but warn and have no effect on network topology.

## Operations

- `POST /v1/vector-stores`: create the physical collection before DB metadata;
  compensate physical creation if metadata creation fails. Never replace an
  existing collection. Storage details are omitted from public failure messages.
- `POST /v1/vector-stores/{id}/vectors`: accept already-computed vectors. The
  request extends the shared `VectorStoreAddRequest`, accepts `metadata` or
  `meta_data`, and optionally accepts a shared file-create model under `file`.
  A file upload sends chunks, vectors, payload metadata, and file metadata in
  one operation. Core enforces file identity in payloads and removes the new
  point IDs if the file metadata transaction fails.
- `POST /v1/vector-stores/{id}/search`: accept `query_vector`, never query text or
  client Qdrant topology. Preserve filters, named vector selection, score
  threshold, offset, and top-k. Return shared search-result semantics plus
  `id`, `vector_id`, `metadata`, and `meta_data` aliases.
- Soft delete only changes DB metadata. Hard delete removes the Qdrant collection
  before DB metadata, and an already-absent collection is treated as success.
- File delete removes matching `file_path` vectors before deleting its DB record;
  a repeated deletion does not decrement the file count again.
- Deleted stores are rejected by normal read, search, upsert, and file operations.
  File-status updates also check that the file belongs to the requested store.

Limits: 256 vectors per upsert, 4096 dimensions, 65536 characters per text,
8 MiB serialized upsert body, finite numeric values, top-k 1–100, offset 0–10000,
and 64 KiB search filters. Oversized documents require smaller upload batches;
the current SDK single-file helper submits one coordinated batch.

SDK write requests are not automatically replayed after a network timeout,
preventing blind duplicate inserts. As with any Qdrant/SQL boundary, compensation
is best-effort; a process crash or failure of the compensation itself can leave
orphaned collections or points. Such failures are logged by Core.

## Optional embeddings

SDK HEAD was version 1.112.0 before these changes, with no optional embedding extra.
`projectdavid[embeddings]` now declares sentence-transformers; `text-embeddings`
is a compatibility alias for existing Core image requirements. Base SDK use and
vector-client construction do not load sentence-transformers or Torch.
Missing embedding dependencies produce the existing FileProcessor install message
only when a method actually needs embeddings. FileProcessor kwargs are honored.
Core package metadata also moves its existing sentence-transformers requirement
into an optional `embeddings` extra. The existing native file-search path still
uses the SDK FileProcessor and needs that optional embedding capability.

## Validation and release acceptance

Focused Core tests exercise routing/auth, ownership, input validation, operation
ordering, compensation, soft/hard/file deletion, DB count updates, and persisted
Qdrant state across client restart. Qdrant tests use local storage, not Docker.
SDK tests exercise local FileProcessor calls followed by HTTP requests,
deprecation behavior, missing ML dependencies, and gateway authentication.

Validation result: 33 Core tests passed (26 vector-boundary tests and 7 existing
router tests); all 70 SDK unit tests passed. Both repository diffs passed
`git diff --check`, and the implementation diff was reviewed.

No local Docker builds, commits, pushes, CI triggers, or deployments were run.
The existing user modification to `tests/integration/orc_fc_events.py` was left
byte-for-byte unchanged.

After publishing/deploying Core through CI and installing the matching released
SDK with its embeddings extra, use Gideon with
`PROJECTDAVID_BASE_URL=http://localhost:80`. Configure no Qdrant host or port.
Create a store, add a local document, search it, delete its file, soft-delete a
store, and hard-delete a store. Confirm requests go to the gateway, embeddings
are created locally, and hard deletion still succeeds after restarting Core.
This deployment-dependent live acceptance has not yet been performed.
