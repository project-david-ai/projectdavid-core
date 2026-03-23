# ProjectDavid — Sovereign Forge: Distributed Fine-Tuning Guide

This guide details the end-to-end flow for training custom "Brain Deltas" (LoRA adapters) across the distributed GPU mesh.

### 1. Stage the Raw Dataset
Datasets must be in `.jsonl` format. ProjectDavid handles multi-tenant isolation by persisting these files to the central Samba hub.

```python
from projectdavid import Entity
import os

client = Entity(
    base_url=os.getenv("PROJECT_DAVID_PLATFORM_BASE_URL"),
    api_key=os.getenv("DEV_PROJECT_DAVID_CORE_TEST_USER_KEY"),
)

# Upload to Samba and register metadata
dataset = client.datasets.create(
    file_path="my_data.jsonl",
    name="Specialized Knowledge Base",
    fmt="jsonl"
)
print(f"📦 Dataset ID: {dataset.id}")
```


### 2. Async Preparation (Direct Hub Access)**


```python

# Trigger background validation and train/eval split
client.datasets.prepare(dataset.id)
# Poll until status is 'active'
# The API is checking JSON integrity and counting samples

````


### 3. Dispatch the Training Job

The Smart Scheduler will now choose the healthiest node in the cluster (the one with the most free logical VRAM) and assign the job.

```python

# Submit the job to the Cluster Mesh
job = client.training.create(
    dataset_id=dataset.id,
    base_model="unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit",
    framework="unsloth",
    config={
        "learning_rate": 2e-4,
        "num_train_epochs": 1,
        "lora_r": 16
    }
)
print(f"🔥 Job {job.id} dispatched to Node: {job.node_id}")

````

## 4. Monitor the GPU Worker

The training-worker on the assigned node will pop the job from Redis, stage the JSONL to its local NVMe scratch space, and spawn the Unsloth subprocess.

**Watch the logs for the "Sovereign Forge" signature:**


````bash
training_worker  | 🚀 Node rtx4060_laptop_main claiming Training Job: job_...
training_worker  | 💓 Heartbeat: VRAM usage detected
training_worker  | [job_...] 🦥 Unsloth: Will patch your computer...
training_worker  | [job_...] 🔥 Starting GPU Training kernels...
training_worker  | [job_...] 100%|██████████| 20/20 [00:15, 1.28it/s]
training_worker  | [job_...] 💾 Exporting adapters to /mnt/training_data/models/ftm_...
````


### Fine-Tuning Troubleshooting & "Big Hammer" Commands

**A. The "VRAM Ledger" Reset (If Scheduling Fails)**

If the API refuses to start a job with "Insufficient Resources" but nvidia-smi shows the GPU is empty, the ledger is out of sync. Clear it:

````bash
docker exec -it my_mysql_cosmic_catalyst mysql -u api_user -p<PASSWORD> entities_db -e "DELETE FROM gpu_allocations; UPDATE training_jobs SET status = 'failed' WHERE status = 'in_progress';"
````

**B. Redis Queue Purge (Kill Pending Jobs)**

If you submitted 10 jobs by accident, flush the Redis list:

````bash 

docker exec -it redis redis-cli del training_jobs

````

**C. The "Signature Boss" (trl API Mismatches)**


If the worker logs show ```TypeError``` regarding ```tokenizer``` or ```max_seq_length```, ensure ```unsloth_train.py``` 
is using the ```SFTConfig``` and processing_class wrapper:


Correct: ```processing_class=tokenizer```

Correct: ```args=SFTConfig(dataset_text_field="text", ...)```


**D. Hardware Scoping (The Laptop profile)**

When you are working from a developement machine:

Ensure your .env contains TRAINING_PROFILE=laptop. This forces the following safety limits:

- Sequence Length: 1024 (instead of 2048).
- Batch Size: 1 (with 8 accumulation steps).
- Optimizer: adamw_8bit (saves ~2GB VRAM).

**E. Manual Model Registration (The "Dangling Brain" Fix)**

If the training finished but the DB record is missing, manually "Claim" the weights:


**F. List Base Models**

````bash 

docker exec -it my_mysql_cosmic_catalyst mysql -u api_user -p<PASSWORD> entities_db -e "SELECT id, name, base_model, storage_path, status FROM fine_tuned_models LIMIT 10;"

````


**G. List Fine-Tuned Models**

````bash 
INSERT INTO fine_tuned_models (id, user_id, name, base_model, storage_path, status, is_active, created_at, updated_at) 
VALUES ('ftm_XYZ', 'user_ID', 'Recovered Brain', 'Base_Model_ID', 'models/ftm_XYZ', 'active', 0, UNIX_TIMESTAMP(), UNIX_TIMESTAMP());

````

**G. Get Current Active Nodes**

````bash 
INSERT INTO fine_tuned_models (id, user_id, name, base_model, storage_path, status, is_active, created_at, updated_at) 
VALUES ('ftm_XYZ', 'user_ID', 'Recovered Brain', 'Base_Model_ID', 'models/ftm_XYZ', 'active', 0, UNIX_TIMESTAMP(), UNIX_TIMESTAMP());

````

