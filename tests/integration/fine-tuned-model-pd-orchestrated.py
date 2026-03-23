"""
ProjectDavid — Fine-Tuned Assistant Orchestration Test
---------------------------------------------------
Flow:
  SDK → Thread/Message → Run(Assistant + Model Override) →
  Redis → Orchestrator → vLLM (LoRA david-ft)

This confirms that the Fine-Tuned weights are correctly integrated
into the full Assistant logic.
"""

import json
import os
import time

from dotenv import load_dotenv
from projectdavid import ContentEvent, DecisionEvent, Entity, ReasoningEvent, ToolCallRequestEvent

load_dotenv()

# ------------------------------------------------------------------
# ANSI Colors
# ------------------------------------------------------------------
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
GREY = "\033[90m"
RESET = "\033[0m"

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
BASE_URL = os.getenv("BASE_URL", "http://localhost:80")
API_KEY = os.getenv("DEV_PROJECT_DAVID_CORE_TEST_USER_KEY")

# CRITICAL: The identity of the behavior container
ASSISTANT_ID = os.getenv("DEV_PROJECT_DAVID_CORE_TEST_ASSISTANT_ID")

# Target the LoRA adapter name defined in the orchestrator/vllm config
# MODEL_ID = "vllm/david-ft"

MODEL_ID = "vllm/unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit"

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://vllm_server:8000")

TEST_PROMPT = "Hello Assistant! Can you confirm you are currently utilizing your new fine-tuned weights ('david-ft') to answer me?"

# ------------------------------------------------------------------
# Execution
# ------------------------------------------------------------------
client = Entity(base_url=BASE_URL, api_key=API_KEY)


def run_assistant_ft_test():
    print(f"\n{CYAN}[▶] Initializing Fine-Tuned Assistant Test...{RESET}")
    print(f"{CYAN}[▶] ASSISTANT_ID: {ASSISTANT_ID}{RESET}")

    assistant = client.assistants.create_assistant(
        name="Sovereign-Forge Test",
        instructions="You are a helpful assistant that can answer questions about fine-tuned weights.",
    )

    # 1. Setup Thread
    thread = client.threads.create_thread()

    # 2. Add Message (linked to Assistant)
    message = client.messages.create_message(
        thread_id=thread.id,
        role="user",
        content=TEST_PROMPT,
        assistant_id=assistant.id,  # Key mapping
    )
    print(f"{GREEN}[✓] Thread/Message Created: {thread.id}{RESET}")

    # 3. Create Run
    # We pass the assistant_id to pull instructions,
    # and override the model to use our specific LoRA adapters.
    run = client.runs.create_run(assistant_id=assistant.id, thread_id=thread.id, model=MODEL_ID)

    # 4. Setup the Stream
    stream = client.synchronous_inference_stream
    stream.setup(
        thread_id=thread.id,
        assistant_id=assistant.id,
        message_id=message.id,
        run_id=run.id,
    )

    print(f"{CYAN}[▶] TARGET MODEL: {MODEL_ID}{RESET}")
    print(f"{CYAN}[▶] PROMPT:       {TEST_PROMPT}{RESET}\n")
    print(f"{'LATENCY':<12} | {'EVENT':<25} | PAYLOAD")
    print("-" * 100)

    last_tick = time.perf_counter()
    global_start = last_tick

    try:
        # 5. Execute unified stream
        for event in stream.stream_events(
            model=MODEL_ID,
            meta_data={"vllm_base_url": VLLM_BASE_URL},
        ):
            now = time.perf_counter()
            delta = now - last_tick
            last_tick = now

            if isinstance(event, ContentEvent):
                # Print assistant text stream
                print(f"{GREEN}{event.content}{RESET}", end="", flush=True)
            else:
                # Print orchestration events
                print(
                    f"\n{GREY}[{delta:+.4f}s]{RESET:<4} "
                    f"| {CYAN}{event.__class__.__name__:<25}{RESET} "
                    f"| {json.dumps(event.to_dict())}"
                )

    except Exception as e:
        print(f"\n{RED}[ERROR] {e}{RESET}")

    finally:
        total = time.perf_counter() - global_start
        print(f"\n\n{YELLOW}{'='*50}")
        print(f"  ASSISTANT INFERENCE COMPLETE")
        print(f"  TOTAL ROUND TRIP: {total:.4f}s")
        print(f"{'='*50}{RESET}\n")


if __name__ == "__main__":
    run_assistant_ft_test()
