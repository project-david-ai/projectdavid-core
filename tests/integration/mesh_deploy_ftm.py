import os
import time

from dotenv import load_dotenv
from projectdavid import Entity

load_dotenv()

# ANSI Colors
MAGENTA = "\033[95m"
GREEN = "\033[92m"
CYAN = "\033[96m"
RESET = "\033[0m"

client = Entity(
    base_url=os.getenv("PROJECT_DAVID_PLATFORM_BASE_URL"),
    api_key=os.getenv("DEV_PROJECT_DAVID_CORE_TEST_USER_KEY"),
)

# The REAL weights we verified earlier
TARGET_FTM = "ftm_7i8THeyHtMvMk6Ns5TGxZ9"


def run_ftm_deployment():
    print(f"\n{MAGENTA}🧠 STAGE 1: Requesting Mesh Deployment for LoRA: {TARGET_FTM}{RESET}")

    # 1. Trigger the Cluster Scheduler
    try:
        result = client.models.activate(TARGET_FTM)
        print(
            f"{GREEN}✅ API Response: Deploying {result.activated} to Node: {result.target_node}{RESET}"
        )
    except Exception as e:
        print(f"❌ API Error: {e}")
        return

    # 2. Verify the Mutex and Ledger
    print(f"\n{CYAN}⏳ STAGE 2: Verifying Cluster State Synchronization...{RESET}")
    time.sleep(2)

    model = client.models.retrieve(TARGET_FTM)
    if model.is_active and model.node_id:
        print(f"{GREEN}✅ DB Record Updated: {model.id} is now ACTIVE on {model.node_id}{RESET}")

    print(f"\n{MAGENTA}🚀 FINAL STEP: Check the 'training_worker' logs.{RESET}")
    print(f"The supervisor will stop any old vLLM and restart with:")
    print(f"   '--enable-lora --lora-modules {TARGET_FTM}=...'")


if __name__ == "__main__":
    run_ftm_deployment()
