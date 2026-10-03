## Purpose

Project David separates **embedding compute** from **vector storage**.

The SDK or consuming application performs document parsing, chunking, embedding generation, and query embedding locally. Project David Core owns the authenticated vector data plane and is the only component that communicates with Qdrant.

This preserves a clean sovereignty boundary without forcing the Project David platform itself to carry heavyweight ML or GPU dependencies.

---

## Architecture

```text
Application / Agent
        |
        | Project David SDK
        |
        +--> FileProcessor
        |      - parse documents
        |      - chunk text
        |      - generate embeddings locally
        |
        +--> query embedding locally
        |
        | HTTPS / authenticated Project David API
        v
Nginx :80
        |
        v
Project David Core API
        |
        +--> authentication / tenancy
        +--> vector-store metadata
        +--> collection lifecycle
        +--> vector upsert
        +--> vector search
        +--> vector deletion
        |
        v
Qdrant
   internal network only
```

The consumer does **not** require direct network access to Qdrant.

---

## Design Principle

The architecture deliberately separates two concerns:

### Compute plane

Owned by the SDK or consuming application.

Responsibilities:

- document parsing;
- text chunking;
- embedding generation;
- query embedding;
- optional local ML acceleration;
- selection of the embedding implementation.

The current SDK uses `FileProcessor`, with `sentence-transformers` available through the optional `embeddings` extra.

```text
projectdavid[embeddings]
```

The base Project David SDK does not require embedding dependencies.

### Storage plane

Owned by Project David Core.

Responsibilities:

- authentication;
- tenant ownership;
- vector-store metadata;
- Qdrant collection creation;
- vector persistence;
- vector search using an already-produced query vector;
- file-vector deletion;
- collection deletion.

Qdrant remains an internal implementation detail of the runtime.

---

## Why Embeddings Stay Client-Side

Project David Core must not require a GPU, Torch, Sentence Transformers, or another heavyweight embedding runtime simply to operate vector stores.

Keeping embedding compute outside Core provides several advantages:

- Core remains lightweight.
- Deployments without GPUs remain valid.
- Applications can choose CPU or GPU embedding infrastructure independently.
- Different consumers can use different embedding providers or models.
- Qdrant remains centrally controlled without centralising ML compute.
- The Project David API remains the sovereignty and tenancy boundary.

This means Project David can own the vector **data plane** without owning the vector **compute plane**.

---

## Vector Store Creation

The SDK creates a vector store through the authenticated Project David API.

```text
SDK
  |
  | POST /v1/vector-stores
  v
Core
  |
  +--> create metadata
  +--> create Qdrant collection
```

The SDK does not create the Qdrant collection directly.

---

## File Ingestion

File ingestion is split across the compute and storage planes.

```text
file
 |
 v
SDK FileProcessor
 |
 +--> parse
 +--> chunk
 +--> embed
 |
 v
[texts + vectors + metadata]
 |
 | authenticated HTTP
 v
Project David Core
 |
 +--> validate ownership
 +--> persist vectors
 +--> persist file/vector metadata
 |
 v
Qdrant
```

The SDK may include metadata such as:

- file ID;
- file name;
- file path or declared document location;
- chunk index;
- line/page metadata;
- caller-provided metadata.

The `file_path` field is metadata associated with the document and does not imply that Qdrant or Core must be able to access a local filesystem path.

---

## Vector Search

Query embedding remains local.

```text
query text
   |
   v
SDK embedding model
   |
   v
query vector
   |
   | authenticated HTTP
   v
Project David Core
   |
   v
Qdrant search
   |
   v
normalised results
   |
   v
SDK / application
```

Core receives the already-generated vector and performs the storage-side search.

Core does not need to know which embedding model generated it.

The application is responsible for ensuring that query vectors are compatible with the vectors stored in the collection.

---

## Deletion

Deletion is also routed through Core.

For file deletion:

```text
SDK
  |
  | DELETE / authenticated API
  v
Core
  |
  +--> delete matching vectors
  +--> update metadata
  v
Qdrant
```

For vector-store deletion:

```text
SDK
  |
  | DELETE / authenticated API
  v
Core
  |
  +--> delete collection
  +--> delete/update store metadata
  v
Qdrant
```

The SDK no longer needs direct Qdrant access for either operation.

---

## Network Boundary

The supported path is:

```text
Consumer
   |
   v
PROJECTDAVID_BASE_URL
   |
   v
Nginx :80
   |
   v
Project David Core
   |
   v
Qdrant :6333
```

Qdrant is internal to the Project David deployment.

There is intentionally:

- no public Qdrant port requirement;
- no `/qdrant` Nginx proxy;
- no requirement for consumers to know the Qdrant hostname;
- no requirement for Gideon or another application to join the Project David Docker network.

Existing Nginx routing to the Core API is sufficient.

---

## SDK Compatibility

Historically, `VectorStoreClient` could construct a direct `VectorStoreManager` using a `vector_store_host`.

That topology is no longer required for normal vector operations.

Where retained for compatibility, `vector_store_host` should be considered legacy/deprecated configuration rather than part of the supported storage path.

The supported contract is:

```text
base_url + API key
```

not:

```text
base_url + API key + Qdrant host
```

---

## Authentication and Tenancy

All vector data-plane operations pass through Project David Core.

This gives Core a single enforcement point for:

- API-key authentication;
- tenant ownership;
- resource access;
- metadata consistency;
- collection lifecycle;
- policy enforcement.

A consumer cannot bypass those controls by speaking directly to Qdrant in the supported architecture.

---

## Gideon

Gideon consumes this architecture as a normal Project David application.

Its vector path is:

```text
Gideon
  |
  +--> Project David SDK 1.113.0+
  |
  +--> local FileProcessor
  +--> local embeddings
  |
  | PROJECTDAVID_BASE_URL
  v
Project David Nginx
  |
  v
Project David Core
  |
  v
internal Qdrant
```

Gideon therefore owns no Qdrant topology.

It only needs:

- the Project David API base URL;
- a valid Project David API key;
- the SDK embedding extra where local embeddings are required.

---

## Installation

Install the SDK with local embedding support:

```bash
pip install "projectdavid[embeddings]>=1.113.0"
```

The optional extra currently provides the local embedding dependency without making it part of the base SDK installation.

---

## Runtime Responsibilities

| Component | Responsibility |
|---|---|
| Application / Agent | Chooses when and why vector operations occur |
| Project David SDK | Parsing, chunking, local embeddings, HTTP client |
| Nginx | Public HTTP gateway |
| Project David Core | Auth, tenancy, metadata, vector data-plane orchestration |
| Qdrant | Vector persistence and similarity search |

---

## Non-Goals

This architecture does not require Project David Core to:

- run Sentence Transformers;
- install Torch;
- own a GPU;
- parse arbitrary client documents;
- generate query embeddings;
- expose Qdrant publicly;
- know the caller's local filesystem layout.

It also does not prevent future server-side embedding implementations.

A future server-native embedding provider or worker can be added behind an explicit abstraction without making heavyweight embedding compute mandatory for every Project David deployment.

---

## Acceptance Criteria

A deployment satisfies the intended architecture when all of the following are true:

1. `VectorStoreClient` can create a vector store using only the Project David API.
2. The consumer does not connect to `localhost:6333` or another Qdrant endpoint.
3. `FileProcessor` performs document parsing and embedding locally.
4. Text, vectors, and metadata are sent to Core over authenticated HTTP.
5. Core persists those vectors into its internal Qdrant instance.
6. Query text is embedded locally.
7. Core accepts the query vector and performs vector search.
8. File and store deletion are performed through Core.
9. Core itself does not require `sentence-transformers`, Torch, or GPU support for normal vector persistence/search.
10. Nginx requires no dedicated Qdrant proxy.

---

## Summary

The architecture is deliberately:

```text
local compute
    +
authenticated central storage
```

or, more specifically:

```text
SDK / application
    = parsing + chunking + embeddings

Project David Core
    = auth + tenancy + vector-store lifecycle + storage API

Qdrant
    = internal vector database
```

The result is a vector architecture that preserves Project David's runtime boundary while avoiding unnecessary coupling between vector storage and embedding compute.
"""

path = Path("/mnt/data/EMBEDDINGS.md")
path.write_text(content, encoding="utf-8")
print(path)
