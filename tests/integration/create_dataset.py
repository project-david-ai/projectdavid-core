import json
import os
import time

from dotenv import load_dotenv
from projectdavid import Entity

# Load environment variables (Local context)
load_dotenv()

# Initialize the ProjectDavid SDK
# Presumes internal clients (datasets, training, models) are correctly wired in Entity
client = Entity(
    base_url=os.getenv("PROJECT_DAVID_PLATFORM_BASE_URL"),
    api_key=os.getenv("DEV_PROJECT_DAVID_CORE_TEST_USER_KEY"),
)


def create_local_test_file(filename="test_dataset.jsonl"):
    """Creates a small valid JSONL file for testing."""
    sample_data = [
        {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I am fine."},
            ]
        },
    ]
    with open(filename, "w") as f:
        for entry in sample_data:
            f.write(json.dumps(entry) + "\n")
    return filename


def run_integration_test():
    print("🚀 Starting Full Fine-Tuning Pipeline Test (Laptop Mode)...")

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 1: DATA INGESTION
    # SDK reads local bytes and hands them to the Core API (Port 9000).
    # The Core API persists the file to the Samba Hub.
    # ──────────────────────────────────────────────────────────────────────────
    file_name = create_local_test_file()
    print(f"📦 STAGE 1: Uploading and registering dataset from {file_name}...")
    dataset = client.datasets.create(
        name="integration_test_ds",
        fmt="jsonl",
        description="Automated integration test dataset",
        file_path=file_name,
    )
    print(f"✅ Dataset Created: ID={dataset.id}, Status={dataset.status}")

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 2: ASYNC PREPARATION
    # Training API (Port 9001) performs an infrastructure-direct read from Samba.
    # It validates the JSONL format and computes train/eval splits.
    # ──────────────────────────────────────────────────────────────────────────
    print(f"⚙️ STAGE 2: Triggering preparation for dataset {dataset.id}...")
    client.datasets.prepare(dataset.id)

    print("⏳ Polling for 'active' status...")
    max_retries = 15
    for i in range(max_retries):
        dataset = client.datasets.retrieve(dataset.id)
        if dataset.status == "active":
            print(f"✅ Dataset ready! (Samples: {dataset.train_samples})")
            break
        elif dataset.status == "failed":
            print(
                f"❌ Dataset preparation failed: {dataset.config.get('preparation_error')}"
            )
            return
        time.sleep(2)
    else:
        print("❌ Timeout: Dataset preparation taking too long.")
        return

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 3: JOB ORCHESTRATION
    # Training API creates a Job record and pushes a ticket to Redis.
    # This handoff moves the task from the Web tier to the GPU tier.
    # ──────────────────────────────────────────────────────────────────────────
    print(
        f"🔥 STAGE 3: Submitting training job [Qwen-1.5B] for dataset {dataset.id}..."
    )
    job = client.training.create(
        dataset_id=dataset.id,
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
        framework="unsloth",
        config={"learning_rate": 2e-4, "num_train_epochs": 1, "lora_r": 16},
    )
    print(f"✅ Training Job Submitted: ID={job.id}, Status={job.status}")

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 4: REDIS PIPELINE VERIFICATION
    # Verification of the destructive BRPOP handoff.
    # Checks if the job is either in the queue OR already picked up by the worker.
    # ──────────────────────────────────────────────────────────────────────────
    print("🔍 STAGE 4: Verifying pipeline connection (Redis -> Worker handoff)...")
    time.sleep(1)
    verified = False
    for attempt in range(5):
        queue_state = client.training.peek_queue()
        in_queue = any(item.job_id == job.id for item in queue_state.data)

        job = client.training.retrieve(job.id)
        started = job.status in ["in_progress", "completed"]

        if in_queue or started:
            print(f"✨ Handoff verified! (In Queue: {in_queue}, Started: {started})")
            verified = True
            break
        time.sleep(2)

    if not verified:
        print(f"❌ FAIL: Job {job.id} never reached the queue or worker.")
        return

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 5: GPU EXECUTION (MONITORING)
    # The Worker stages data from Samba to local NVMe and runs the ML subprocess.
    # We poll the MySQL state until the LoRA adapters are exported.
    # ──────────────────────────────────────────────────────────────────────────
    print(
        f"⏳ STAGE 5: Monitoring GPU training (Check Worker Terminal for live kernels)..."
    )
    while True:
        job = client.training.retrieve(job.id)
        if job.status == "completed":
            print(f"\n🏆 VICTORY: Training finished successfully!")
            break
        elif job.status == "failed":
            print(f"\n💥 CRASH: Training failed. Error: {job.last_error}")
            return
        else:
            print(f"   Worker Status: {job.status}...", end="\r")
        time.sleep(10)

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 6: REGISTRY & ACTIVATION
    # We promote the resulting adapter to be the 'Active' model in the DB.
    # This instructs the Orchestrator/vLLM to load these weights on next boot.
    # ──────────────────────────────────────────────────────────────────────────
    print(f"🎯 STAGE 6: Promoting model from Job {job.id} to ACTIVE status...")

    # Locate the model artifact registered by the worker
    models = client.models.list()
    new_model = next((m for m in models.data if m.training_job_id == job.id), None)

    if new_model:
        print(f"✅ Found Registry Artifact: {new_model.id}")
        activation = client.models.activate(new_model.id)
        print(f"💡 Next Step: {activation.next_step}")

        # Final Database Verification
        final_check = client.models.retrieve(new_model.id)
        if final_check.is_active:
            print(
                f"✨ PIPELINE COMPLETE: Model {new_model.id} is now the Active Brain."
            )
    else:
        print("❌ FAIL: Could not find fine_tuned_model record in registry.")

    # Cleanup local file
    if os.path.exists(file_name):
        os.remove(file_name)


if __name__ == "__main__":
    try:
        run_integration_test()
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
