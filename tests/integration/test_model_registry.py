import os

from dotenv import load_dotenv
from projectdavid import Entity

load_dotenv()

client = Entity(
    base_url=os.getenv("PROJECT_DAVID_PLATFORM_BASE_URL"),
    api_key=os.getenv("DEV_PROJECT_DAVID_CORE_TEST_USER_KEY"),
)


def test_registry():
    print("🚀 Testing Model Registry...")

    try:
        # 1. List all models for this user
        models = client.models.list()
        print(f"✅ Found {models.total} models in the registry.")

        if models.total == 0:
            print("❓ No models found. Ensure the worker successfully finished at least one job.")
            return

        for m in models.data:
            print(f"\n--- Model: {m.id} ---")
            print(f"   Name:        {m.name}")
            print(f"   Job Ref:     {m.training_job_id}")
            print(f"   Base:        {m.base_model}")
            print(f"   Samba Path:  {m.storage_path}")
            print(f"   Status:      {m.status}")
            print(f"   Active:      {m.is_active}")

        test_activation()

    except Exception as e:
        print(f"❌ Registry Test Failed: {e}")


def test_activation():
    # 1. Get the latest model
    models = client.models.list()
    target_model = models.data[0]
    print(f"🎯 Target for activation: {target_model.id}")

    # 2. Activate it
    result = client.models.activate(target_model.id)
    print(f"✅ {result.next_step}")

    # 3. Verify status changed in DB
    updated = client.models.retrieve(target_model.id)
    print(f"🧐 Database Check: {updated.id} Is Active? {updated.is_active}")

    if updated.is_active:
        print("\n✨ PIPELINE COMPLETE: You have successfully trained and deployed a custom model!")


if __name__ == "__main__":
    test_registry()
