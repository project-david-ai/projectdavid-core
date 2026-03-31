import os
import time

from dotenv import load_dotenv
from projectdavid import Entity

# ------------------------------------------------------------------
# 0.  SDK init + env
# ------------------------------------------------------------------
load_dotenv(".tests.env")

client = Entity(api_key=os.getenv("DEV_PROJECT_DAVID_CORE_TEST_USER_KEY"))
print(client)

# -------------------------------------
# Create fine tuning dataset
# - stages training data
# --------------------------------------
dataset = client.datasets.create(
    file_path="projectdavid_sdk_finetune.jsonl",
    name="Specialized Knowledge Base",
    fmt="jsonl",
)
print(f"📦 Dataset ID: {dataset.id}")

# -----------------------------------------
# Validation time
# ------------------------------------------
client.datasets.prepare(dataset.id)

# Poll until active
while True:
    ds = client.datasets.retrieve(dataset_id=dataset.id)
    print(f"Status: {ds.status}")
    if ds.status == "active":
        print(
            f"✅ Dataset ready — {ds.train_samples} train / {ds.eval_samples} eval samples"
        )
        break
    if ds.status == "failed":
        raise Exception(f"Dataset preparation failed: {ds}")
    time.sleep(3)

# -------------------------------------------------------
# Dispatch the training job
# ----------------------------------------------------
# Submit the job to the Cluster Mesh

# -----------------------------------------
# We need some method
# that the user can use to see available nodes
# Candidate base models that they can fine tune
# - USERS SHOULD NOT BE ABLE TO TRIGGER HG DOWNLOADS OF MODELS NOT IN HG CACHE
# -------------------------------------------
job = client.training.create(
    dataset_id=dataset.id,
    base_model="unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit",
    framework="unsloth",
    config={"learning_rate": 2e-4, "num_train_epochs": 1, "lora_r": 16},
)
print(f"🔥 Job {job.id} dispatched to cluster")

# ------------------------------------------------------------
# At this point, from a users' perspective, the job goes into a
# Black hole. We need to work out user feedback / check job status
# ------------------------------------------------------------
print("\n⏳ Polling job status...\n")

TERMINAL_STATES = {"completed", "failed", "cancelled"}

while True:
    job_status = client.training.retrieve(job_id=job.id)

    print(
        f"  [{job_status.status.upper()}] "
        f"started={job_status.started_at or '—'} "
        f"output={job_status.output_path or '—'}"
    )

    if job_status.status in TERMINAL_STATES:
        if job_status.status == "completed":
            print(f"\n✨ Training complete — adapters at: {job_status.output_path}")
        else:
            print(f"\n❌ Job ended with status: {job_status.status}")
            if hasattr(job_status, "last_error") and job_status.last_error:
                print(f"   Error: {job_status.last_error}")
        break

    time.sleep(10)
