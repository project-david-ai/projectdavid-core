# ProjectDavid — Fine-Tuning Pipeline Handover Document
**Date:** 2026-03-17  
**Status:** Active development — dataset layer complete, training layer next  
**Repo:** `projectdavid-core` (monorepo)  
**Stack:** FastAPI · MySQL · Redis · Samba · Docker Compose · Alembic

---

## 1. What This Project Is

ProjectDavid is an open-source, self-hosted LLM runtime API — think OpenAI Assistants but for heterogeneous inference providers (vLLM, Ollama, TogetherAI, Hyperbolic, DeepSeek). The current sprint adds **API-driven fine-tuning as a service** — a full MLOps pipeline allowing users to upload datasets, run training jobs (Axolotl/Unsloth), register fine-tuned models, and serve them through the existing vLLM inference layer with no SDK changes.

---

## 2. Repository Structure (Relevant Paths)

```
projectdavid-core/
├── src/api/
│   ├── entities_api/           ← Main API (inference, assistants, files, etc.)
│   │   ├── models/models.py    ← SQLAlchemy models (source of truth for Alembic)
│   │   ├── routers/            ← FastAPI routers
│   │   ├── services/           ← Business logic
│   │   └── cli/
│   │       ├── docker_manager.py        ← CLI orchestration tool
│   │       └── generate_docker_compose.py ← Compose file generator
│   └── training/               ← NEW: Training service (separate FastAPI app)
│       ├── app.py              ← FastAPI entry point (port 9001)
│       ├── dependencies.py     ← JWT auth dependency
│       ├── worker.py           ← Redis queue consumer (GPU runner)
│       ├── unsloth_train.py    ← Unsloth training script
│       ├── db/database.py      ← SQLAlchemy engine (shared MySQL, own pool)
│       ├── models/models.py    ← Training-scoped ORM models (Dataset, TrainingJob, FineTunedModel)
│       ├── routers/
│       │   ├── __init__.py     ← training_router aggregator
│       │   └── datasets_router.py  ← ✅ COMPLETE
│       └── services/
│           └── dataset_service.py  ← ✅ COMPLETE
├── migrations/
│   ├── env.py                  ← Alembic env (targets entities_api Base only)
│   └── versions/               ← All migration files (idempotent SafeDDL pattern)
├── docker/
│   └── training/
│       └── Dockerfile          ← Single Dockerfile for both training-api and training-worker
├── docker-compose.yml          ← Full stack (training services under profile: training)
└── training_reqs_unhashed.txt  ← Pip requirements for training container
```

---

## 3. Shared Packages

### `projectdavid-common` (PyPI: `projectdavid-common`)
Shared Pydantic schemas and utilities used by both the core API and SDK.  
Repo: `projectdavid-core/src/entities_common` (separate repo: `entities_common`)

**Training schemas added this session** — in `projectdavid_common/schemas/training_schema.py`:
- `DatasetCreate`, `DatasetRead`, `DatasetList`, `DatasetDeleted`
- `TrainingJobCreate`, `TrainingJobRead`, `TrainingJobList`
- `FineTunedModelCreate`, `FineTunedModelRead`, `FineTunedModelList`, `FineTunedModelDeleted`
- `HubPushPayload`, `ActivateModelResponse`

All schemas registered in `ValidationInterface` in `validation.py`.

### `projectdavid` (SDK, separate repo)
Client SDK. `DatasetsClient` added at `projectdavid/clients/datasets_client.py`.  
`Entity` class needs `self.datasets = DatasetsClient(...)` wired in.

---

## 4. Architecture

```
SDK (projectdavid)
  client.datasets     → Training API :9001
  client.files        → Core API     :9000  (file upload/storage)
  client.training     → Training API :9001  (NOT YET BUILT)
  client.models       → Training API :9001  (NOT YET BUILT)

Training API (port 9001)
  /v1/datasets          ✅ COMPLETE
  /v1/training-jobs     ❌ NOT BUILT
  /v1/fine-tuned-models ❌ NOT BUILT

Training Worker (GPU container)
  Redis BRPOP → subprocess → Axolotl/Unsloth → checkpoint → DB update
  Status: worker.py exists but is a STUB — full implementation needed

Core API (port 9000)
  /v1/uploads           ✅ (existing — used for dataset file storage)
  /v1/files             ✅ (existing — used to retrieve dataset bytes)

Database: shared MySQL instance
  datasets              ✅ table exists, migrated
  training_jobs         ✅ table exists, migrated  
  fine_tuned_models     ✅ table exists, migrated

Redis: job queue
  training_jobs queue   ❌ NOT WIRED (training_service.py not built)

Samba: file storage
  dataset files         ✅ uploaded via core API FileService
  training checkpoints  ❌ worker not implemented
```

---

## 5. What Is Complete

### ✅ Database Layer
- Three tables (`datasets`, `training_jobs`, `fine_tuned_models`) created via Alembic migration `005820173bc4`
- `file_id` column added to `datasets` via migration `33111f6ac0b8`
- `storage_path` made nullable on `datasets`
- Models live in `src/api/training/models/models.py` (own `Base`, shares MySQL)
- `entities_api/models/models.py` no longer contains fine-tuning models (moved in `53ed443a77c1`)

### ✅ Training API Service
- FastAPI app boots on port 9001
- Connects to shared MySQL via own connection pool
- JWT auth via `SANDBOX_AUTH_SECRET` (same secret as sandbox service)
- `datasets_router.py` fully implemented and registered

### ✅ Dataset Service (`dataset_service.py`)
- `create_dataset` — registers metadata with `file_id` reference (no direct Samba)
- `get_dataset` — ownership-enforced retrieval
- `list_datasets` — user-scoped, status filter, pagination
- `delete_dataset` — soft delete
- `prepare_dataset` — async background task, fetches file from core API via `/v1/files/{file_id}/base64`, validates format (chatml/alpaca/sharegpt/jsonl), computes train/eval split

### ✅ Docker Infrastructure
- `docker-compose.yml` has `training-api` and `training-worker` under `profiles: ["training"]`
- `Dockerfile` (single file) builds both services
- `platform-api docker-manager --mode up --services training-api` starts the training API
- `platform-api docker-manager --mode build --services training-api` rebuilds it

### ✅ SDK `DatasetsClient`
- Two-step transparent upload: `files.upload()` then `datasets.register()`
- Full CRUD: `create`, `retrieve`, `list`, `prepare`, `delete`
- Located at `projectdavid/clients/datasets_client.py`

### ✅ Migration Pipeline
- Alembic `env.py` uses `entities_api` Base only
- Volume mount `./migrations:/app/migrations` ensures generated files appear locally
- SafeDDL helpers used for all migrations (idempotent)

---

## 6. What Is NOT Built — Next Steps in Order

### Step 1: Fix JWT in training container (IMMEDIATE — in progress)
**Problem:** `jwt` package conflict. `PyJWT` needed but wrong package installed.  
**Fix:** Add `python-jose[cryptography]` to `training_reqs_unhashed.txt`. Update `dependencies.py`:
```python
from jose import jwt, JWTError
# use JWTError instead of jwt.exceptions.*
```
Rebuild training container after this change.

---

### Step 2: `training_service.py`
**File:** `src/api/training/services/training_service.py`  
**Responsibilities:**
- `create_training_job(db, user_id, dataset_id, base_model, framework, config)` — creates `TrainingJob` record, pushes to Redis queue
- `get_training_job(db, job_id, user_id)` — ownership-enforced retrieval
- `list_training_jobs(db, user_id, status, limit, offset)`
- `cancel_training_job(db, job_id, user_id)` — sets status to `cancelling`

**Redis publishing pattern:**
```python
import redis, json, os

r = redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379/0"))
r.lpush("training_jobs", json.dumps({"job_id": job.id, "user_id": user_id}))
```

---

### Step 3: `training_jobs_router.py`
**File:** `src/api/training/routers/training_jobs_router.py`  
**Endpoints:**
```
POST   /v1/training-jobs              create + enqueue
GET    /v1/training-jobs              list (user-scoped)
GET    /v1/training-jobs/{job_id}     retrieve + status
POST   /v1/training-jobs/{job_id}/cancel
```
Register in `src/api/training/routers/__init__.py`:
```python
from src.api.training.routers.training_jobs_router import router as training_jobs_router
training_router.include_router(training_jobs_router, prefix="/training-jobs", tags=["Training Jobs"])
```

---

### Step 4: `worker.py` — Full Implementation
**File:** `src/api/training/worker.py` (currently a stub)  
**Responsibilities:**
1. `BRPOP` from Redis `training_jobs` queue (blocking)
2. Fetch job from DB, set status → `in_progress`
3. Fetch dataset file from core API: `GET http://api:9000/v1/files/{file_id}/base64`
4. Write Axolotl/Unsloth YAML config to Samba at `/mnt/training_data/configs/{job_id}/config.yml`
5. Spawn subprocess: `entrypoint.sh --framework axolotl --config /mnt/training_data/configs/{job_id}/config.yml`
6. On success: set status → `completed`, write `output_path`, call `register_from_job()`
7. On failure: set status → `failed`, write `last_error`

**Key pattern:**
```python
import redis, subprocess, json, time
from src.api.training.db.database import SessionLocal
from src.api.training.models.models import TrainingJob, StatusEnum

r = redis.from_url(os.getenv("REDIS_URL"))

while True:
    _, data = r.brpop("training_jobs")
    payload = json.loads(data)
    # process job...
```

---

### Step 5: `model_registry_service.py`
**File:** `src/api/training/services/model_registry_service.py`  
**Responsibilities:**
- `register_from_job(job)` — auto-creates `FineTunedModel` after training completes
- `activate_model(db, model_id, user_id)` — sets `is_active=True`, writes `VLLM_MODEL` to `.env`
- `push_to_hub(db, model_id, user_id, repo_id)` — async HuggingFace Hub push via `huggingface_hub`

---

### Step 6: `fine_tuned_models_router.py`
**File:** `src/api/training/routers/fine_tuned_models_router.py`  
**Endpoints:**
```
POST   /v1/fine-tuned-models                      register manually
GET    /v1/fine-tuned-models                      list (user-scoped)
GET    /v1/fine-tuned-models/{model_id}           retrieve
POST   /v1/fine-tuned-models/{model_id}/activate  set as active vLLM model
POST   /v1/fine-tuned-models/{model_id}/push      push to HF Hub
DELETE /v1/fine-tuned-models/{model_id}           soft delete
```

---

### Step 7: SDK `TrainingClient` and `ModelsClient`
**Files in `projectdavid` SDK repo:**
- `projectdavid/clients/training_client.py`
- `projectdavid/clients/models_client.py`

Wire into `Entity.__init__`:
```python
self.training = TrainingClient(base_url=training_url, api_key=api_key)
self.models = ModelsClient(base_url=training_url, api_key=api_key)
```

---

### Step 8: Integration Test (Full Cycle)
```python
client = Entity(base_url="http://localhost:80", api_key="...")

# Upload and prepare
dataset = client.datasets.create("data.jsonl", name="test", fmt="jsonl")
client.datasets.prepare(dataset.id)

# Poll until active
while dataset.status != "active":
    time.sleep(2)
    dataset = client.datasets.retrieve(dataset.id)

# Submit job
job = client.training.create(
    dataset_id=dataset.id,
    base_model="Qwen/Qwen2.5-7B-Instruct",
    framework="axolotl",
    config={"lora_r": 16, "num_epochs": 1}
)

# Poll until done
while job.status not in ("completed", "failed"):
    time.sleep(30)
    job = client.training.retrieve(job.id)

# Activate
client.models.activate(job.fine_tuned_model_id)
# Then restart vLLM: platform-api docker-manager --mode up --services vllm
```

---

## 7. Key Environment Variables

| Variable | Used By | Purpose |
|---|---|---|
| `DATABASE_URL` | all services | MySQL connection string |
| `SANDBOX_AUTH_SECRET` | training-api | JWT signing secret |
| `REDIS_URL` | training-api, training-worker | Job queue |
| `ASSISTANTS_BASE_URL` | training-api, training-worker | Core API internal URL (`http://api:9000`) |
| `WORKER_API_KEY` | training-worker | API key for calling core API |
| `SHARED_PATH` | training-worker | Samba mount path |
| `HF_TOKEN` | training-worker | HuggingFace Hub access |
| `ADMIN_API_KEY` | training-api | Worker API key source |

---

## 8. Docker Commands Reference

```powershell
# Start full stack (no training)
platform-api docker-manager --mode up

# Start training API only
platform-api docker-manager --mode up --services training-api

# Build training image
platform-api docker-manager --mode build --services training-api

# Build with no cache
platform-api docker-manager --mode build --no-cache --services training-api

# Start with training profile (training-api + training-worker)
docker compose --profile training up -d

# Run Alembic migration
docker compose exec api alembic upgrade head

# Generate new migration
docker compose exec api alembic revision --autogenerate -m "description"

# View training API logs
platform-api docker-manager --mode logs --services training-api --follow
```

---

## 9. Migration Policy

All migrations use **SafeDDL helpers** from `migrations/utils/safe_ddl.py`:
- `add_column_if_missing` — never fails if column exists
- `drop_column_if_exists` — never fails if column missing
- `safe_alter_column` — alters only if column exists
- `has_table` / `has_column` — guard checks

**Never use raw `op.add_column` or `op.alter_column` directly in production migrations.**

---

## 10. Known Issues / Watch Points

1. **JWT package conflict** — `jwt` vs `PyJWT`. Use `python-jose[cryptography]` in `training_reqs_unhashed.txt`. In `dependencies.py` use `from jose import jwt, JWTError`.

2. **`asyncio.create_task` in `prepare_dataset`** — requires a running event loop. The `prepare_dataset` endpoint is `async` for this reason. The background task opens its own `SessionLocal()` to avoid `DetachedInstanceError`.

3. **Alembic only tracks `entities_api` Base** — training model changes are NOT auto-detected by `alembic revision --autogenerate`. Any schema changes to `src/api/training/models/models.py` must also be reflected in `src/api/entities_api/models/models.py` (the source of truth for Alembic) and a migration generated from there.

4. **`docker-compose.yml` must not be regenerated** once the stack is running with real secrets — it would overwrite the existing file only if `force=True` is passed to `generate_dev_docker_compose()`. The generator defaults to `force=False` (skip if exists).

5. **`platform-api.exe` lock on Windows** — after editing `docker_manager.py` locally, `pip install -e .` may fail if a terminal has recently run `platform-api`. Close all terminals and reinstall in a fresh one.

6. **`FineTunedModelDeleted` Pydantic warning** — field `model_id` conflicts with Pydantic's protected namespace `model_`. Fix: add `model_config = ConfigDict(protected_namespaces=())` to the schema class.

---

## 11. File Checklist — What Needs To Be Created

| File | Status | Notes |
|---|---|---|
| `src/api/training/services/training_service.py` | ❌ | Redis publish + DB ops |
| `src/api/training/routers/training_jobs_router.py` | ❌ | 4 endpoints |
| `src/api/training/services/model_registry_service.py` | ❌ | register, activate, push |
| `src/api/training/routers/fine_tuned_models_router.py` | ❌ | 6 endpoints |
| `src/api/training/worker.py` | ⚠️ stub | Full Redis BRPOP + subprocess impl |
| `projectdavid/clients/training_client.py` | ❌ | SDK client |
| `projectdavid/clients/models_client.py` | ❌ | SDK client |
| `tests/integration/create_dataset.py` | ✅ exists | Working once JWT fixed |
| `tests/integration/create_training_job.py` | ❌ | Needs training_service first |