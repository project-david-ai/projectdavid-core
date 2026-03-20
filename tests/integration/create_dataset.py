import json
import os
import time

from dotenv import load_dotenv
from projectdavid import Entity

# Load environment variables
load_dotenv()

# Initialize the ProjectDavid SDK
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

    # 1. Create Local File
    file_name = create_local_test_file()

    # 2. Upload and Register Dataset
    print(f"📦 Registering dataset from {file_name}...")
    dataset = client.datasets.create(
        name="integration_test_ds",
        fmt="jsonl",
        description="Automated integration test dataset",
        file_path=file_name,
    )
    print(f"✅ Dataset Created: ID={dataset.id}, Status={dataset.status}")

    # 3. Trigger Preparation
    print(f"⚙️ Triggering preparation for dataset {dataset.id}...")
    client.datasets.prepare(dataset.id)

    # 4. Wait for Dataset to be 'active'
    print("⏳ Polling for 'active' status...")
    max_retries = 15
    for i in range(max_retries):
        dataset = client.datasets.retrieve(dataset.id)
        if dataset.status == "active":
            print(f"✅ Dataset is ready! (Samples: {dataset.train_samples})")
            break
        elif dataset.status == "failed":
            print(f"❌ Dataset preparation failed: {dataset.config.get('preparation_error')}")
            return
        time.sleep(2)
    else:
        print("❌ Timeout: Dataset preparation taking too long.")
        return

    # 5. Submit Training Job
    print(f"🔥 Submitting training job [Qwen-1.5B] for dataset {dataset.id}...")
    job = client.training.create(
        dataset_id=dataset.id,
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
        framework="unsloth",
        config={"learning_rate": 2e-4, "num_train_epochs": 1, "lora_r": 16},
    )
    print(f"✅ Training Job Submitted: ID={job.id}, Status={job.status}")

    # 6. Resilient Pipeline Verification
    print("🔍 Verifying pipeline connection (SDK -> API -> Redis -> Worker)...")
    time.sleep(1)  # Brief pause for Redis LPUSH

    verified = False
    for attempt in range(5):
        # A. Check if still in Redis queue
        queue_state = client.training.peek_queue()
        in_queue = any(item.job_id == job.id for item in queue_state.data)

        # B. Check if Worker already updated the DB status
        job = client.training.retrieve(job.id)
        started = job.status in ["in_progress", "completed"]

        if in_queue or started:
            print(
                f"✨ SUCCESS: Pipeline handoff verified! (In Queue: {in_queue}, Started: {started})"
            )
            verified = True
            break

        print(f"   ...waiting for worker pickup (Attempt {attempt+1})")
        time.sleep(2)

    if not verified:
        print(f"❌ FAIL: Job {job.id} never reached the queue or worker.")
        return

    # 7. Final Step: Monitor the REAL Training Process
    print(f"⏳ Monitoring GPU training (Check Worker Terminal for live logs)...")
    while True:
        job = client.training.retrieve(job.id)

        if job.status == "completed":
            print(f"\n🏆 VICTORY: Training finished successfully!")
            print(f"📂 Weights saved to Samba at: {job.output_path}")
            break
        elif job.status == "failed":
            print(f"\n💥 CRASH: Training failed. Error: {job.last_error}")
            break
        else:
            # Simple progress ticker
            print(f"   Worker Status: {job.status}...", end="\r")

        time.sleep(10)

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
