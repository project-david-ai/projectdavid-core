"""
Standard Inference Test (No Tools)
---------------------------------------------------
1. Simple prompt -> response.
2. Handles Reasoning (DeepSeek) and Content events.
3. No tool execution or recursion logic.
"""

import os
import time

from config_orc_fc import config
from dotenv import load_dotenv
# Import only what we need for text generation
from projectdavid import ContentEvent, Entity, ReasoningEvent

# ------------------------------------------------------------------
# 0. CONFIGURATION
# ------------------------------------------------------------------
load_dotenv()

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
GREY = "\033[90m"
RESET = "\033[0m"

BASE_URL = config.get("base_url") or os.getenv("BASE_URL", "http://localhost:9000")
ENTITIES_API_KEY = os.getenv("ENTITIES_API_KEY") or config.get("entities_api_key")
ENTITIES_USER_ID = os.getenv("ENTITIES_USER_ID") or config.get("entities_user_id")
HYPERBOLIC_API_KEY = os.getenv("HYPERBOLIC_API_KEY")

# Default to a standard text model prompt
MODEL_ID = config.get("model", "hyperbolic/deepseek-ai/DeepSeek-V3")
PROVIDER_KW = config.get("provider", "Hyperbolic")
ASSISTANT_ID = config.get("assistant_id", "asst_13HyDgBnZxVwh5XexYu74F")
TEST_PROMPT = "Explain the difference between TCP and UDP in one paragraph."

print(f"{GREY}[CONFIG] Model: {MODEL_ID} | Provider: {PROVIDER_KW}{RESET}")

client = Entity(base_url=BASE_URL, api_key=ENTITIES_API_KEY)

# Note: We do NOT need to bind_clients() here,
# because we are not asking the SDK to handle recursive tool loops.

# ==================================================================
# 1. Setup & Global Timer
# ==================================================================
print(f"\n{YELLOW}[TIMER] Starting Standard Inference...{RESET}")
global_start = time.perf_counter()

thread = client.threads.create_thread()
message = client.messages.create_message(
    thread_id=thread.id,
    role="user",
    content=TEST_PROMPT,
    assistant_id=ASSISTANT_ID,
)
run = client.runs.create_run(assistant_id=ASSISTANT_ID, thread_id=thread.id)

# ------------------------------------------------------------------
# 2. Stream Setup
# ------------------------------------------------------------------
stream = client.synchronous_inference_stream
stream.setup(
    thread_id=thread.id,
    assistant_id=ASSISTANT_ID,
    message_id=message.id,
    run_id=run.id,
    api_key=HYPERBOLIC_API_KEY,
)

print(f"\n{CYAN}[▶] STREAM STARTED: {TEST_PROMPT}{RESET}\n")

last_tick = time.perf_counter()
current_mode = None  # To track if we are printing thoughts or answer

try:
    for event in stream.stream_events(provider=PROVIDER_KW, model=MODEL_ID):

        # --- A. Handle Reasoning (Optional, for Chain-of-Thought models) ---
        if isinstance(event, ReasoningEvent):
            if current_mode != "reasoning":
                print(f"{GREY}🤔 [THOUGHTS]{RESET}")
                current_mode = "reasoning"

            print(f"{GREY}{event.content}{RESET}", end="", flush=True)

        # --- B. Handle Content (The actual answer) ---
        elif isinstance(event, ContentEvent):
            if current_mode != "content":
                if current_mode == "reasoning":
                    print("\n")  # Newline after thoughts
                print(f"{GREEN}🤖 [ANSWER]{RESET}")
                current_mode = "content"

            print(f"{GREEN}{event.content}{RESET}", end="", flush=True)

except Exception as e:
    print(f"\n{RED}[!] Error in loop: {e}{RESET}")

# ==================================================================
# TIMER RESULTS
# ==================================================================
global_end = time.perf_counter()
total_time = global_end - global_start

print(f"\n\n{YELLOW}" + "=" * 60)
print(f" TOTAL TIME: {total_time:.4f}s")
print("=" * 60 + f"{RESET}\n")
