"""
Debug Mode: Standard Generation (Granular Timing)
---------------------------------------------------
1. Streams high-level Event Instances for a standard prompt.
2. Prints the Python Class Type and payload with granular latency.
3. Skips tool execution logic if no tool is called.
INCLUDES GRANULAR PER-EVENT TIMING.
"""

import json
import os
import time

from config_orc_prompt import config
from dotenv import load_dotenv

# Import the new Event classes
from projectdavid import (
    ContentEvent,
    DecisionEvent,
    Entity,
    ReasoningEvent,
    StatusEvent,
    ToolCallRequestEvent,
)

# ------------------------------------------------------------------
# 0. CONFIGURATION & SDK INIT
# ------------------------------------------------------------------
load_dotenv()

# ANSI Colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
GREY = "\033[90m"
MAGENTA = "\033[95m"
RESET = "\033[0m"

# Resolve Constants
BASE_URL = config.get("base_url") or os.getenv("BASE_URL", "http://localhost:9000")
ENTITIES_API_KEY = os.getenv("ENTITIES_API_KEY") or config.get("entities_api_key")
ENTITIES_USER_ID = os.getenv("ENTITIES_USER_ID") or config.get("entities_user_id")

# Inference Params
HYPERBOLIC_API_KEY = os.getenv("HYPERBOLIC_API_KEY")
MODEL_ID = config.get("model", "together-ai/mistralai/Ministral-3-14B-Instruct-2512")
PROVIDER_KW = config.get("provider", "Hyperbolic")
ASSISTANT_ID = config.get("assistant_id", "asst_13HyDgBnZxVwh5XexYu74F")

# [CHANGE] Standard Prompt (No tools needed)
TEST_PROMPT = config.get("test_prompt", "Write a haiku about Python code.")

print(f"{GREY}[CONFIG] Model: {MODEL_ID} | Provider: {PROVIDER_KW}{RESET}")
print(f"{GREY}[CONFIG] Assistant: {ASSISTANT_ID}{RESET}")

# Initialize Client
client = Entity(base_url=BASE_URL, api_key=ENTITIES_API_KEY)

# Bind clients for synchronous inference
if hasattr(client, "synchronous_inference_stream"):
    client.synchronous_inference_stream.bind_clients(
        client.runs, client.actions, client.messages
    )

# ==================================================================
# TIMER START
# ==================================================================
print(f"\n{YELLOW}[TIMER] Starting Standard Gen Timer...{RESET}")
global_start = time.perf_counter()


# ------------------------------------------------------------------
# 1. Setup Thread & Run
# ------------------------------------------------------------------
t_setup_start = time.perf_counter()

print(f"\n{GREY}[1/2] Creating Thread, Message & Run...{RESET}")
thread = client.threads.create_thread()
message = client.messages.create_message(
    thread_id=thread.id,
    role="user",
    content=TEST_PROMPT,
    assistant_id=ASSISTANT_ID,
)
run = client.runs.create_run(assistant_id=ASSISTANT_ID, thread_id=thread.id)

t_setup_end = time.perf_counter()


# ------------------------------------------------------------------
# 2. Smart Event Stream (Granular Timing)
# ------------------------------------------------------------------
stream = client.synchronous_inference_stream
stream.setup(
    thread_id=thread.id,
    assistant_id=ASSISTANT_ID,
    message_id=message.id,
    run_id=run.id,
    api_key=HYPERBOLIC_API_KEY,
)

print(f"\n{CYAN}[▶] STREAM: Event Instance Inspection{RESET}")
print(f"{'LATENCY':<12} | {'EVENT CLASS':<25} | {'TYPED JSON PAYLOAD'}")
print("-" * 110)

tool_event: ToolCallRequestEvent = None
t_stream_start = time.perf_counter()
last_tick = t_stream_start

# We iterate over EVENTS
try:
    for event in stream.stream_events(provider=PROVIDER_KW, model=MODEL_ID):

        # --- GRANULAR TIMING ---
        current_tick = time.perf_counter()
        delta = current_tick - last_tick
        last_tick = current_tick
        # -----------------------

        # Debug the Type
        class_name = event.__class__.__name__
        payload = event.to_dict()

        # Pick color based on class
        color = RESET
        if isinstance(event, ContentEvent):
            color = GREEN
        elif isinstance(event, ToolCallRequestEvent):
            color = YELLOW
        elif isinstance(event, ReasoningEvent):
            color = CYAN
        elif isinstance(event, DecisionEvent):
            color = MAGENTA
        elif isinstance(event, StatusEvent):
            color = GREY

        # Print with Delta Time
        time_str = f"[{delta:+.4f}s]"
        print(
            f"{GREY}{time_str:<12}{RESET} | {color}{class_name:<25}{RESET} | {json.dumps(payload)}"
        )

        # Track if tool is called (Unexpected but possible)
        if isinstance(event, ToolCallRequestEvent):
            tool_event = event

except Exception as e:
    print(f"{RED}[!] Stream Error: {e}{RESET}")

t_stream_end = time.perf_counter()


# ------------------------------------------------------------------
# 3. Completion Handling
# ------------------------------------------------------------------
if tool_event:
    print(f"\n{YELLOW}[!] Unexpected Tool Call detected: {tool_event.tool_name}{RESET}")
    print(f"{GREY}Skipping execution as this is a standard prompt test.{RESET}")
else:
    print(f"\n{GREEN}[✓] Standard Generation Complete (No tools called).{RESET}")


# ==================================================================
# TIMER RESULTS
# ==================================================================
global_end = time.perf_counter()

setup_time = t_setup_end - t_setup_start
stream_time = t_stream_end - t_stream_start
total_time = global_end - global_start

print(f"\n{YELLOW}" + "=" * 60)
print(f" PERFORMANCE REPORT ({PROVIDER_KW})")
print("=" * 60 + f"{RESET}")
print(f"1. Setup (Thread/Msg/Run) : {setup_time:.4f}s")
print(f"2. Stream (Generation)    : {stream_time:.4f}s")
print("-" * 60)
print(f"{GREEN}TOTAL TIME                : {total_time:.4f}s{RESET}")
print(f"{YELLOW}" + "=" * 60 + f"{RESET}\n")

print(f"{GREY}--- End of Debug Script ---{RESET}")
