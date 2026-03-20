# ProjectDavid — Fine-Tuning Pipeline Handover Document
**Date:** 2026-03-20
**Status:** **API Layer Complete** — Dataset & Job Management verified; Worker implementation next.
**Repo:** `projectdavid-core` (monorepo)
**Stack:** FastAPI · MySQL · Redis · Samba · Docker Compose · Nginx (Dynamic DNS)

---

## 1. What This Project Is
ProjectDavid is an open-source, self-hosted LLM runtime API ("OpenAI in a box"). The current sprint implements a full MLOps pipeline. Users can upload datasets via the Core API, register and prepare them via the Training API, and queue Unsloth/Axolotl training jobs. The final result is a Fine-Tuned Model registered for inference in the vLLM layer.

---

## 2. Updated Repository Structure

projectdavid-core/
├── src/api/
│ ├── entities_api/
│ │ ├── services/file_service.py ← ✅ MODIFIED: Relaxed MIME validation for .jsonl
│ └── training/
│ ├── app.py ← ✅ COMPLETE: Factory pattern, auto-migration enabled
│ ├── worker.py ← ⚠️ STUB: Next major milestone
│ ├── models/models.py ← ✅ MODIFIED: Added updated_at, synchronized StatusEnum
│ ├── routers/
│ │ ├── init.py ← ✅ COMPLETE: Aggregated /v1/ routes
│ │ ├── datasets_router.py ← ✅ COMPLETE
│ │ └── training_jobs_router.py ← ✅ COMPLETE: Includes /queue/peek
│ └── services/
│ ├── dataset_service.py ← ✅ COMPLETE: Infrastructure-direct logic (Samba)
│ └── training_service.py ← ✅ COMPLETE: Redis LPUSH verified
├── docker/
│ └── nginx/
│ └── nginx.conf ← ✅ MODIFIED: Dynamic DNS Resolver fix (v5s)
└── projectdavid/ (SDK Repo)
├── clients/
│ ├── datasets_client.py ← ✅ COMPLETE
│ ├── training_client.py ← ✅ COMPLETE: /v1/training-jobs + peek_queue
│ └── models_client.py ← ❌ NOT BUILT
code
Code
---

## 3. Infrastructure & Reliability Fixes (The "Battle-Hardened" Layer)

### ✅ The Nginx "DNS Ghost" Fix
Nginx was previously throwing 502 Bad Gateway errors when containers were recreated. 
*   **Solution:** Implemented a dynamic resolver (`127.0.0.11`) and variable-based `proxy_pass` in `nginx.conf`. Nginx now re-checks container IPs every 5 seconds.

### ✅ The "Infrastructure-Direct" Prep Logic
Bypassed internal HTTP calls for dataset preparation to eliminate 401 Unauthorized errors.
*   **Solution:** `dataset_service.py` now queries the shared MySQL `file_storage` table directly and uses `SambaClient` to pull bytes for validation/splitting. This is faster and avoids service-to-service API key management.

### ✅ MIME Type Relaxing
The Core API was rejecting `.jsonl` files from Windows clients.
*   **Solution:** `FileService.validate_file_type` now accepts `application/octet-stream` as a valid fallback for supported extensions.

---

## 4. Architecture Flow (Current State)
SDK (DatasetsClient) → POST :80/v1/uploads (Nginx -> Core API -> Samba)
SDK (DatasetsClient) → POST :80/v1/datasets (Nginx -> Training API -> MySQL)
SDK (DatasetsClient) → POST :80/v1/datasets/{id}/prepare (Async Task)
└─ Training API → SQL: Lookup storage_path → Samba: Fetch Bytes → SQL: Active
SDK (TrainingClient) → POST :80/v1/training-jobs (Validate Dataset -> LPUSH Redis)
SDK (TrainingClient) → GET :80/v1/training-jobs/queue/peek (Verify job is in Redis)
code
Code
---

## 5. What Is NOT Built — Next Steps in Order

### Step 1: `worker.py` — The GPU Consumer
**File:** `src/api/training/worker.py` (Currently a stub)
**Responsibilities:**
1. **Queue Consumption:** `BRPOP` from Redis `training_jobs`.
2. **Job Staging:**
    - Fetch Dataset physical path from MySQL.
    - Copy file from Samba to local container scratch space (`/tmp/training/`).
3. **Config Generation:** Transform the JSON `job.config` into a YAML file for Unsloth.
4. **Subprocess:** `subprocess.Popen(["python", "unsloth_train.py", "--config", ...])`.
5. **State Management:** Update MySQL status to `in_progress`, `completed`, or `failed`.

---

### Step 2: `model_registry_service.py`
**File:** `src/api/training/services/model_registry_service.py`
**Responsibilities:**
- **Weight Export:** Move resulting LoRA adapters/merged weights from the worker scratch space back to Samba `/mnt/training_data/models/{model_id}`.
- **Registration:** Create the `fine_tuned_models` DB record automatically upon worker success.

---

### Step 3: `fine_tuned_models_router.py` & SDK
**File:** `src/api/training/routers/fine_tuned_models_router.py`
**Endpoints:**
- `GET /v1/fine-tuned-models`: User-scoped list.
- `POST /v1/fine-tuned-models/{id}/activate`: Sets the model as the active vLLM instance (Update `.env` + Restart vLLM service via CLI).

---

## 6. Key Environment Variables

| Variable | Used By | Purpose |
|---|---|---|
| `REDIS_URL` | training-api/worker | Use `decode_responses=True` in Python client |
| `SMBCLIENT_SERVER` | api/training-api | Set to `samba` (internal DNS) |
| `WORKER_API_KEY` | worker/training-api | Only used for non-direct service calls |
| `SHARED_PATH` | ALL | Path to shared Samba mount |

---

## 7. Known Issues / Watch Points

1. **Trailing Slashes:** Ensure the SDK doesn't send trailing slashes on POST requests (e.g., use `/v1/training-jobs`, not `/v1/training-jobs/`) to match Nginx/FastAPI routing exactly.
2. **Redis Corrupt Data:** If `/queue/peek` returns a 500, the `training_jobs` key in Redis might be a String instead of a List. Fix: `docker exec -it redis redis-cli del training_jobs`.
3. **Pydantic Protected Namespaces:** `FineTunedModelRead` uses `model_id`. You **must** keep `model_config = ConfigDict(protected_namespaces=())` in the schema to avoid Pydantic V2 crashes.
4. **Volume Mounts:** The `training-api` in `docker-compose.yml` now has `./src:/app/src`. Ensure this remains to enable live-coding without rebuilds.

---

## 8. Progress Tracker

| Milestone | Status | Notes |
|---|---|---|
| Core API Uploads | ✅ | .jsonl supported |
| Dataset Preparation | ✅ | Samba-direct, tested and active |
| Training Job Creation | ✅ | Database persistence validated |
| Redis Enqueueing | ✅ | Secure Peeking verified |
| Training Worker | ❌ | Subprocess logic needed |
| Model Activation | ❌ | vLLM reloading needed |