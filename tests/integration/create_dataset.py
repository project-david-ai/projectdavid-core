import json
import os
import time

from dotenv import load_dotenv
from projectdavid import Entity

# Load environment variables
load_dotenv()

# Initialize the ProjectDavid SDK
# Ensure your Entity class has self.datasets and self.training wired in
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
    print("🚀 Starting Full Fine-Tuning Pipeline Test...")

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
    prep_response = client.datasets.prepare(dataset.id)
    print(f"   Response: {prep_response}")

    # 4. Wait for Dataset to be 'active'
    print("⏳ Polling for 'active' status (validating file and computing splits)...")
    max_retries = 15
    is_ready = False
    for i in range(max_retries):
        dataset = client.datasets.retrieve(dataset.id)
        print(f"   - Attempt {i+1}: Status is '{dataset.status}'")

        if dataset.status == "active":
            is_ready = True
            print(f"✅ Dataset is ready! (Train samples: {dataset.train_samples})")
            break
        elif dataset.status == "failed":
            error = dataset.config.get("preparation_error", "Unknown error")
            print(f"❌ Dataset preparation failed: {error}")
            return

        time.sleep(2)

    if not is_ready:
        print("❌ Timeout: Dataset did not reach 'active' status in time.")
        return

    # 5. Submit Training Job
    print(f"🔥 Submitting training job for dataset {dataset.id}...")
    job = client.training.create(
        dataset_id=dataset.id,
        base_model="unsloth/Llama-3.2-1B-Instruct",
        framework="unsloth",
        config={"learning_rate": 2e-4, "num_train_epochs": 1, "lora_r": 16},
    )
    print(f"✅ Training Job Submitted: ID={job.id}, Status={job.status}")

    # 6. Secure Multi-tenant Queue Verification (Peek)
    print("🔍 Verifying Redis queue via secure API gateway...")
    # Give the API a moment to complete the Redis LPUSH
    time.sleep(1)

    try:
        queue_state = client.training.peek_queue()
        print(f"   Queue Check: {queue_state.total_in_queue} items found for your user.")

        # Verify our specific job is in the list
        found = any(item.job_id == job.id for item in queue_state.data)

        if found:
            print(f"\n✨ SUCCESS: Job {job.id} is confirmed in the Redis queue!")
            print("The pipeline is fully connected: SDK -> API -> Redis.")
        else:
            print(f"\n❌ FAIL: Job {job.id} was not found in the queue peek.")
            print(f"Current Queue: {[item.job_id for item in queue_state.data]}")

    except Exception as e:
        print(f"❌ FAIL: Error during queue verification: {e}")

    # Cleanup local file
    if os.path.exists(file_name):
        os.remove(file_name)


if __name__ == "__main__":
    run_integration_test()
