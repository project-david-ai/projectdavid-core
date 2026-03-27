"""
Debug Mode: Event-Driven Round-Trip (Type Inspection)
---------------------------------------------------
1. Streams high-level Event Instances (ContentEvent, ToolCallRequestEvent, etc.).
2. Prints the Python Class Type and the underlying Payload for every event.
3. Uses event.execute() for the tool execution.
4. Loads configuration from config.json.
INCLUDES GRANULAR PER-EVENT TIMING.
"""

import json
import os
import time  # <--- Added for timing

from config_orc_fc import config
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

# ANSI Colors for Debugging
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
GREY = "\033[90m"
MAGENTA = "\033[95m"
RESET = "\033[0m"


# Resolve Constants (Config > Env > Default)
BASE_URL = config.get("base_url") or os.getenv("BASE_URL", "http://localhost:9000")
ENTITIES_API_KEY = os.getenv("ENTITIES_API_KEY") or config.get("entities_api_key")
ENTITIES_USER_ID = os.getenv("ENTITIES_USER_ID") or config.get("entities_user_id")

# Inference Params
HYPERBOLIC_API_KEY = os.getenv("HYPERBOLIC_API_KEY")

MODEL_ID = config.get("model", "together-ai/mistralai/Ministral-3-14B-Instruct-2512")
PROVIDER_KW = config.get("provider", "Hyperbolic")
ASSISTANT_ID = config.get("assistant_id", "asst_13HyDgBnZxVwh5XexYu74F")
TEST_PROMPT = config.get(
    "test_prompt", "Please fetch me the flight times between LAX and JFK."
)

print(f"{GREY}[CONFIG] Model: {MODEL_ID} | Provider: {PROVIDER_KW}{RESET}")
print(f"{GREY}[CONFIG] Assistant: {ASSISTANT_ID}{RESET}")


# Initialize Client
client = Entity(base_url=BASE_URL, api_key=ENTITIES_API_KEY)

# Bind clients for synchronous inference (Internal SDK requirement)
if hasattr(client, "synchronous_inference_stream"):
    client.synchronous_inference_stream.bind_clients(
        client.runs, client.actions, client.messages
    )


# ------------------------------------------------------------------
# 1. Tool Executor Logic
# ------------------------------------------------------------------
def get_flight_times(tool_name: str, arguments: dict) -> str:
    """Fake flight-time lookup."""
    print(
        f"{YELLOW}   -> [LOCAL] Calculating flight times for {arguments.get('departure')}...{RESET}"
    )
    return json.dumps(
        {
            "status": "success",
            "departure": arguments.get("departure", "UNK"),
            "arrival": arguments.get("arrival", "UNK"),
            "duration": "4h 30m",
        }
    )


# ==================================================================
# TIMER START
# ==================================================================
print(f"\n{YELLOW}[TIMER] Starting Round-Trip Timer...{RESET}")
global_start = time.perf_counter()


# ------------------------------------------------------------------
# 2. Setup Thread & Run
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
# 3. Smart Event Stream (DEBUG MODE)
# ------------------------------------------------------------------
stream = client.synchronous_inference_stream
stream.setup(
    user_id=ENTITIES_USER_ID,
    thread_id=thread.id,
    assistant_id=ASSISTANT_ID,
    message_id=message.id,
    run_id=run.id,
    api_key=HYPERBOLIC_API_KEY,
)

print(f"\n{CYAN}[▶] STREAM 1: Event Instance Inspection{RESET}")
print(f"{'LATENCY':<12} | {'EVENT CLASS':<25} | {'TYPED JSON PAYLOAD'}")
print("-" * 110)

tool_event: ToolCallRequestEvent = None
t_stream1_start = time.perf_counter()
last_tick = t_stream1_start  # Track time between events

# We iterate over EVENTS
try:
    for event in stream.stream_events(provider=PROVIDER_KW, model=MODEL_ID):

        # --- GRANULAR TIMING ---
        current_tick = time.perf_counter()
        delta = current_tick - last_tick
        last_tick = current_tick
        # -----------------------

        # A. Debug the Type
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

        # B. Track if we need to execute a tool
        if isinstance(event, ToolCallRequestEvent):
            tool_event = event

except Exception as e:
    print(f"{RED}[!] Stream Error: {e}{RESET}")

t_stream1_end = time.perf_counter()


# ------------------------------------------------------------------
# 4. Execution Round-Trip
# ------------------------------------------------------------------
t_tool_start = 0
t_tool_end = 0
t_stream2_start = 0
t_stream2_end = 0
tool_executed_successfully = False

if tool_event:
    print(f"\n{YELLOW}[LOCAL EXEC] Tool Detected: {tool_event.tool_name}{RESET}")

    t_tool_start = time.perf_counter()

    # Execute the tool using the Event's internal logic
    # This usually handles local function execution AND submitting the result to API
    tool_executed_successfully = tool_event.execute(get_flight_times)

    t_tool_end = time.perf_counter()

    if tool_executed_successfully:
        print(
            f"{GREEN}[✓] Tool Executed & Result Submitted (Took {t_tool_end - t_tool_start:.4f}s).{RESET}"
        )

        print(f"\n{CYAN}[▶] STREAM 2: Final Response Event Inspection{RESET}")
        print(f"{'LATENCY':<12} | {'EVENT CLASS':<25} | {'TYPED JSON PAYLOAD'}")
        print("-" * 110)

        # Re-setup for final answer
        stream.setup(
            user_id=ENTITIES_USER_ID,
            thread_id=thread.id,
            assistant_id=ASSISTANT_ID,
            message_id=message.id,
            run_id=run.id,
            api_key=HYPERBOLIC_API_KEY,
        )

        t_stream2_start = time.perf_counter()
        last_tick = t_stream2_start  # Reset tick for second stream

        for event in stream.stream_events(provider=PROVIDER_KW, model=MODEL_ID):

            # --- GRANULAR TIMING ---
            current_tick = time.perf_counter()
            delta = current_tick - last_tick
            last_tick = current_tick
            # -----------------------

            class_name = event.__class__.__name__
            payload = event.to_dict()
            color = RESET

            if isinstance(event, ContentEvent):
                color = GREEN
            elif isinstance(event, DecisionEvent):
                color = MAGENTA
            elif isinstance(event, StatusEvent):
                color = GREY

            time_str = f"[{delta:+.4f}s]"
            print(
                f"{GREY}{time_str:<12}{RESET} | {color}{class_name:<25}{RESET} | {json.dumps(payload)}"
            )

        t_stream2_end = time.perf_counter()

    else:
        print(f"{RED}[!] Tool execution submission failed.{RESET}")
else:
    print(f"\n{RED}[!] No ToolCallRequestEvent found in stream.{RESET}")


# ==================================================================
# TIMER RESULTS
# ==================================================================
global_end = time.perf_counter()

setup_time = t_setup_end - t_setup_start
stream1_time = t_stream1_end - t_stream1_start
tool_time = t_tool_end - t_tool_start
stream2_time = t_stream2_end - t_stream2_start if tool_executed_successfully else 0
total_time = global_end - global_start

print(f"\n{YELLOW}" + "=" * 60)
print(f" PERFORMANCE REPORT ({PROVIDER_KW})")
print("=" * 60 + f"{RESET}")
print(f"1. Setup (Thread/Msg/Run) : {setup_time:.4f}s")
print(f"2. Stream 1 (Gen ToolCall): {stream1_time:.4f}s")
print(f"3. Tool Exec & Submit     : {tool_time:.4f}s")
print(f"4. Stream 2 (Final Answer): {stream2_time:.4f}s")
print("-" * 60)
print(f"{GREEN}TOTAL ROUND TRIP TIME     : {total_time:.4f}s{RESET}")
print(f"{YELLOW}" + "=" * 60 + f"{RESET}\n")

print(f"{GREY}--- End of Debug Script ---{RESET}")
