import os
import time

from dotenv import load_dotenv
from projectdavid import Entity

load_dotenv()

# ANSI Colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

client = Entity(
    base_url=os.getenv("PROJECT_DAVID_PLATFORM_BASE_URL"),
    api_key=os.getenv("DEV_PROJECT_DAVID_CORE_TEST_USER_KEY"),
)

# This must match an ID in your 'base_models' table (from the seed script)
TARGET_BASE = "unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit"


def run_base_deployment():
    print(f"\n{CYAN}🌐 STAGE 1: Requesting Mesh Deployment for Base Model: {TARGET_BASE}{RESET}")

    # 1. Trigger the Cluster Scheduler
    # This creates the 'pending' Deployment record and locks the VRAM Ledger
    try:
        result = client.models.activate_base(TARGET_BASE)
        print(f"{GREEN}✅ API Response: {result['status']} on Node: {result['node']}{RESET}")
        print(f"💡 Instruction: {result['next_step']}")
    except Exception as e:
        print(f"❌ API Error: {e}")
        return

    # 2. Poll the Mesh Ledger
    # We wait for the Node Agent (Worker) to see the ticket and start the container
    print(f"\n{CYAN}⏳ STAGE 2: Waiting for Node Agent to physically provision vLLM...{RESET}")
    for i in range(20):
        # We check the actual deployment list
        # Note: We look for the deployment where fine_tuned_model_id is None
        models = client.models.list()
        # In a real setup, we'd add a 'client.models.list_deployments()' method
        # For now, we verify the node heartbeat and status via the worker logs
        print(f"   - Polling cluster state (Attempt {i+1})...", end="\r")
        time.sleep(3)

    print(f"\n\n{YELLOW}✨ MESH VERIFICATION: Check your 'training_worker' terminal.{RESET}")
    print(f"You should see: '🚢 Spawning vLLM... for model {TARGET_BASE}'")


if __name__ == "__main__":
    run_base_deployment()
