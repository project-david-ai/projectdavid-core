# Fine-tuning pipeline — architecture & implementation guide

## Overview

The fine-tuning pipeline closes the self-hosted AI loop. Users collect data, train on their own hardware, register the result, and serve it through the same vLLM inference layer already running in the stack — with no changes to the SDK, no changes to the assistants API, and no data leaving the infrastructure.

**Full cycle:** gather data → prepare dataset → train (Axolotl/Unsloth) → register → push to HF Hub (optional) → `pdavid --mode up --vllm --pull`

---

## Architecture

### Three-tier layout

```
projectdavid SDK
  client.datasets · client.training · client.models

FastAPI — projectdavid-core
  /v1/datasets · /v1/training-jobs · /v1/fine-tuned-models

Services
  DatasetService     TrainingService     ModelRegistryService

Storage & execution
  Samba (files)      Training container   HF Hub / Samba (weights)

Database
  datasets · training_jobs · fine_tuned_models   (MySQL)

Deployment
  pdavid configure --set VLLM_MODEL=your/model
  pdavid --mode up --vllm --pull
```

### New DB tables

Three tables added to `models.py` — already staged. See `finetuning_models.py`.

| Table | Purpose |
|---|---|
| `datasets` | Dataset registry — name, format, Samba path, split sizes, status |
| `training_jobs` | Training run — config, framework, lifecycle timestamps, metrics |
| `fine_tuned_models` | Model artifact — HF repo or local path, active flag, vLLM model ID |

### New compose overlay

`docker-compose.training.yml` — opt-in, GPU required, activated via `pdavid --mode up --train`.

---

## Pipeline flow

```
1. upload dataset       POST /v1/datasets              file → Samba → DB record
2. prepare dataset      POST /v1/datasets/{id}/prepare  format · split · validate
3. submit training job  POST /v1/training-jobs          config JSON → job queued
4. container runs       TrainingContainerManager        Axolotl/Unsloth · GPU · checkpoint
5. register model       auto on completion              push HF (optional) · DB record
6. activate             POST /v1/fine-tuned-models/{id}/activate  writes VLLM_MODEL to .env
7. serve                pdavid --mode up --vllm --pull  vLLM loads new model · same API
```

---

## Endpoints

### Datasets

```
POST   /v1/datasets                        register dataset
GET    /v1/datasets                        list (user-scoped)
GET    /v1/datasets/{dataset_id}           retrieve
POST   /v1/datasets/{dataset_id}/prepare   format + split + validate
DELETE /v1/datasets/{dataset_id}           soft delete
```

### Training jobs

```
POST   /v1/training-jobs                   submit job
GET    /v1/training-jobs                   list (user-scoped)
GET    /v1/training-jobs/{job_id}          retrieve + current status
POST   /v1/training-jobs/{job_id}/cancel   cancel running job
```

### Fine-tuned models

```
POST   /v1/fine-tuned-models               register manually
GET    /v1/fine-tuned-models               list (user-scoped)
GET    /v1/fine-tuned-models/{model_id}    retrieve
POST   /v1/fine-tuned-models/{model_id}/activate   set as active vLLM model
POST   /v1/fine-tuned-models/{model_id}/push       push to HF Hub
DELETE /v1/fine-tuned-models/{model_id}    soft delete
```

---

## Pseudo code

### POST /v1/datasets

```python
async def create_dataset(payload: DatasetCreate, user: User):
    dataset_id = generate_id("ds_")
    storage_path = f"datasets/{user.id}/{dataset_id}/{payload.filename}"

    samba_client.upload(payload.file, storage_path)

    dataset = Dataset(
        id=dataset_id,
        user_id=user.id,
        name=payload.name,
        format=payload.format,       # chatml | alpaca | sharegpt | jsonl
        storage_path=storage_path,
        status=StatusEnum.pending,
    )
    db.add(dataset); db.commit()
    return DatasetRead.model_validate(dataset)
```

### POST /v1/datasets/{id}/prepare

```python
async def prepare_dataset(dataset_id: str, user: User):
    dataset = get_dataset_or_404(dataset_id, user.id)
    dataset.status = StatusEnum.processing
    db.commit()
    asyncio.create_task(_run_preparation(dataset))
    return {"status": "processing", "dataset_id": dataset_id}


async def _run_preparation(dataset: Dataset):
    try:
        file = samba_client.download(dataset.storage_path)
        train_samples, eval_samples = split_and_validate(
            file, format=dataset.format, eval_ratio=0.1
        )
        prepared_path = dataset.storage_path.replace("/raw/", "/prepared/")
        samba_client.upload(prepared_splits, prepared_path)

        dataset.train_samples = train_samples
        dataset.eval_samples = eval_samples
        dataset.storage_path = prepared_path
        dataset.status = StatusEnum.active
    except Exception as e:
        dataset.status = StatusEnum.failed
        dataset.config = {"error": str(e)}
    db.commit()
```

### POST /v1/training-jobs

```python
# Payload:
# {
#     "dataset_id": "ds_abc123",
#     "base_model": "Qwen/Qwen2.5-7B-Instruct",
#     "framework": "axolotl",
#     "config": {
#         "lora_r": 16,
#         "lora_alpha": 32,
#         "lora_dropout": 0.05,
#         "learning_rate": 2e-4,
#         "num_epochs": 3,
#         "per_device_train_batch_size": 4,
#         "gradient_accumulation_steps": 4,
#         "max_seq_length": 2048,
#     }
# }

async def create_training_job(payload: TrainingJobCreate, user: User):
    dataset = get_dataset_or_404(payload.dataset_id, user.id)
    assert dataset.status == StatusEnum.active, "Dataset not prepared"

    job_id = generate_id("tj_")
    job = TrainingJob(
        id=job_id,
        user_id=user.id,
        dataset_id=dataset.id,
        base_model=payload.base_model,
        framework=payload.framework,
        config={
            **payload.config,
            "dataset_path": dataset.storage_path,
            "output_dir": f"/mnt/samba/checkpoints/{job_id}",
        },
        status=StatusEnum.queued,
    )
    db.add(job); db.commit()
    asyncio.create_task(training_container_manager.run(job))
    return TrainingJobRead.model_validate(job)
```

### TrainingContainerManager.run()

```python
async def run(job: TrainingJob):
    try:
        job.status = StatusEnum.in_progress
        job.started_at = int(time.time())
        db.commit()

        config_path = f"/mnt/samba/configs/{job.id}/config.yml"
        write_training_config(job.config, config_path, job.framework)

        result = await run_subprocess([
            "docker", "compose",
            "-f", "docker-compose.training.yml",
            "run", "--rm", "training",
            "--config", config_path,
        ])

        if result.returncode == 0:
            job.status = StatusEnum.completed
            job.completed_at = int(time.time())
            job.output_path = job.config["output_dir"]
            await model_registry_service.register_from_job(job)
        else:
            job.status = StatusEnum.failed
            job.failed_at = int(time.time())
            job.last_error = result.stderr[-500:]

    except Exception as e:
        job.status = StatusEnum.failed
        job.last_error = str(e)
    finally:
        db.commit()
```

### POST /v1/fine-tuned-models/{id}/activate

```python
async def activate_model(model_id: str, user: User):
    model = get_model_or_404(model_id, user.id)

    db.query(FineTunedModel).filter(
        FineTunedModel.user_id == user.id,
        FineTunedModel.is_active == True,
    ).update({"is_active": False})

    vllm_model_id = model.hf_repo or model.storage_path
    model.is_active = True
    model.vllm_model_id = vllm_model_id
    db.commit()

    set_key(".env", "VLLM_MODEL", vllm_model_id)

    return {
        "activated": model_id,
        "vllm_model_id": vllm_model_id,
        "next_step": "pdavid --mode up --vllm --pull",
    }
```

### POST /v1/fine-tuned-models/{id}/push

```python
async def _push_to_hub(model: FineTunedModel, repo_id: str = None):
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=os.getenv("HF_TOKEN"))
        target_repo = repo_id or f"projectdavid/{model.name}"
        api.upload_folder(
            folder_path=model.storage_path,
            repo_id=target_repo,
            repo_type="model",
        )
        model.hf_repo = target_repo
        model.status = StatusEnum.active
    except Exception as e:
        model.status = StatusEnum.failed
        model.config = {"push_error": str(e)}
    finally:
        db.commit()
```

### SDK usage

```python
client = Entity(base_url=..., api_key=...)

# Upload and prepare
dataset = client.datasets.create(
    name="my-dataset",
    format="chatml",
    file=open("data.jsonl", "rb"),
)
client.datasets.prepare(dataset.id)

# Submit training job
job = client.training.create(
    dataset_id=dataset.id,
    base_model="Qwen/Qwen2.5-7B-Instruct",
    framework="axolotl",
    config={"lora_r": 16, "num_epochs": 3},
)

# Poll until done
while job.status not in ("completed", "failed"):
    time.sleep(30)
    job = client.training.retrieve(job.id)

# Activate in vLLM
client.models.activate(job.fine_tuned_model.id)
# Then: pdavid --mode up --vllm --pull
```

### docker-compose.training.yml

```yaml
services:
  training:
    image: projectdavid/training:latest
    runtime: nvidia
    environment:
      - HF_TOKEN=${HF_TOKEN}
    volumes:
      - samba_data:/mnt/samba
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

---

## Next steps

### Immediate — unblock the build

- [ ] Confirm `conversation_truncator.py` fix is on `master` and CI has completed
- [ ] Pull new image: `docker pull thanosprime/entities-api-api:latest && docker restart fastapi_cosmic_catalyst`
- [ ] Verify inference stream end-to-end with Together AI and Hyperbolic
- [ ] Merge `ci.yml` fix (scoped GHA cache, removed docker prune) to `master`

### Fine-tuning — implementation order

- [ ] Generate Alembic migration for `datasets`, `training_jobs`, `fine_tuned_models`
- [ ] Implement `DatasetService` — upload, prepare, validate
- [ ] Implement `datasets_router.py` — all five endpoints
- [ ] Implement `TrainingService` + `TrainingContainerManager`
- [ ] Implement `training_jobs_router.py` — submit, retrieve, cancel
- [ ] Implement `ModelRegistryService` — register, activate, push
- [ ] Implement `fine_tuned_models_router.py` — all six endpoints
- [ ] Build `docker-compose.training.yml` overlay
- [ ] Add `--train` flag to `start_orchestration.py` in `projectdavid-platform`
- [ ] Build training Docker image (Axolotl base + Unsloth option)
- [ ] Add `DatasetsClient`, `TrainingClient`, `ModelsClient` to `projectdavid` SDK
- [ ] Integration test: full cycle on a small JSONL dataset

### Housekeeping

- [ ] Remove `BatfishSnapshot` model — audit references, generate drop migration
- [ ] Rename Docker Hub images from `entities-api-*` to `projectdavid-core-*`
- [ ] Update `docker-compose.yml` image references to match renamed images
- [ ] Update CI to build under new image names
- [ ] Open source `projectdavid-platform` repo
- [ ] Rotate PyPI token from account-wide to project-scoped
- [ ] Update `pyproject.toml` in both repos with new URLs (files already generated)

### Platform

- [ ] Publish updated `README.md` to `projectdavid-platform` (file already generated)
- [ ] Commit updated `ci.yml` to `projectdavid-core` (file already generated)
- [ ] Add fine-tuning section to docs site
- [ ] Add `projectdavid-stack.svg` to `assets/svg/` in platform repo (file already generated)

---

## File index — already generated this session

| File | Destination |
|---|---|
| `finetuning_models.py` | append to `src/api/entities_api/models/models.py` |
| `finetuning_pipeline.py` | reference pseudo code |
| `pyproject.toml` (core) | `projectdavid-core/pyproject.toml` |
| `pyproject.toml` (platform) | `projectdavid-platform/pyproject.toml` |
| `ci.yml` | `.github/workflows/ci.yml` in `projectdavid-core` |
| `README.md` | `projectdavid-platform/README.md` |
| `projectdavid-stack.svg` | `projectdavid-platform/assets/svg/` |