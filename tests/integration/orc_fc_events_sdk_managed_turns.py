"""
Level 2 Unified Function Calling Test
---------------------------------------------------
1. Streams Events via a SINGLE loop.
2. Automatically handles Turn 2 (Final Answer) or Turn N (Correction)
   within the same generator loop.
3. Demonstrates the SDK-managed recursive turn logic.
"""

import json
import os
import time

from config_orc_fc import config
from dotenv import load_dotenv

# Import the project classes
from projectdavid import (
    ContentEvent,
    DecisionEvent,
    Entity,
    ReasoningEvent,
    ToolCallRequestEvent,
)

# ------------------------------------------------------------------
# 0. CONFIGURATION & SDK INIT
# ------------------------------------------------------------------
load_dotenv()

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
GREY = "\033[90m"
MAGENTA = "\033[95m"
RESET = "\033[0m"

BASE_URL = config.get("base_url") or os.getenv("BASE_URL", "http://localhost:9000")
ENTITIES_API_KEY = os.getenv("ENTITIES_API_KEY") or config.get("entities_api_key")
ENTITIES_USER_ID = os.getenv("ENTITIES_USER_ID") or config.get("entities_user_id")
HYPERBOLIC_API_KEY = os.getenv("HYPERBOLIC_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")


MODEL_ID = config.get("model", "together-ai/mistralai/Ministral-3-14B-Instruct-2512")
PROVIDER_KW = config.get("provider", "Hyperbolic")
ASSISTANT_ID = config.get("assistant_id", "asst_13HyDgBnZxVwh5XexYu74F")
TEST_PROMPT = config.get(
    "test_prompt", "Please fetch me the flight times between LAX and JFK."
)

print(f"{GREY}[CONFIG] Model: {MODEL_ID} | Provider: {PROVIDER_KW}{RESET}")

client = Entity(base_url=BASE_URL, api_key=ENTITIES_API_KEY)


# ------------------------------------------------------------------
# 1. Tool Executor Logic
# ------------------------------------------------------------------
def get_flight_times(tool_name: str, arguments: dict) -> str:
    """
    Fake tool.
    TESTING TIP: Raise an exception here to test Level 2 Self-Correction!
    """
    print(
        f"{YELLOW}   -> [TOOL EXEC] {tool_name} for {arguments.get('departure')}...{RESET}"
    )

    # Example logic:
    return json.dumps(
        {
            "status": "success",
            "departure": arguments.get("departure", "UNK"),
            "arrival": arguments.get("arrival", "UNK"),
            "duration": "4h 30m",
        }
    )


# --- NEW: DYNAMIC TOOL REGISTRY ---
# Map the string name (from the LLM) to the Python function object
TOOL_REGISTRY = {
    "get_flight_times": get_flight_times,
    # "get_weather": get_weather,  <-- Add more tools here
}

if config.get("provider") == "together":
    API_KEY = TOGETHER_API_KEY

if config.get("provider") == "hyperbolic":
    API_KEY = HYPERBOLIC_API_KEY


# ==================================================================
# 2. Setup & Global Timer
# ==================================================================
print(f"\n{YELLOW}[TIMER] Starting Unified Round-Trip...{RESET}")
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
# 3. Unified Recursive Stream
# ------------------------------------------------------------------
stream = client.synchronous_inference_stream
stream.setup(
    thread_id=thread.id,
    assistant_id=ASSISTANT_ID,
    message_id=message.id,
    run_id=run.id,
)

print(f"\n{CYAN}[▶] UNIFIED STREAM: SDK-Managed Turns{RESET}")
print(f"{'LATENCY':<12} | {'EVENT CLASS':<25} | {'PAYLOAD'}")
print("-" * 110)

last_tick = time.perf_counter()

try:
    # This ONE loop now handles Turn 1 (Tool Call) AND Turn 2 (Answer)
    for event in stream.stream_events(model=MODEL_ID):

        # Timing logic
        current_tick = time.perf_counter()
        delta = current_tick - last_tick
        last_tick = current_tick
        time_str = f"[{delta:+.4f}s]"

        # Debug Printing
        class_name = event.__class__.__name__
        payload = event.to_dict()

        color = RESET
        if isinstance(event, ContentEvent):
            color = GREEN
        elif isinstance(event, ToolCallRequestEvent):
            color = YELLOW
        elif isinstance(event, ReasoningEvent):
            color = CYAN
        elif isinstance(event, DecisionEvent):
            color = MAGENTA

        print(
            f"{GREY}{time_str:<12}{RESET} | {color}{class_name:<25}{RESET} | {json.dumps(payload)}"
        )

        # ---------------------------------------------------------
        # THE MAGIC BIT:
        # When a tool is requested, we just call .execute().
        # The generator (stream_events) will automatically pause,
        # submit the output, and resume the next turn from the model.
        # ---------------------------------------------------------
        if isinstance(event, ToolCallRequestEvent):
            print(f"\n{YELLOW}[LOCAL] AI requested tool: {event.tool_name}{RESET}")

            # --- DYNAMIC DISPATCH ---
            # Lookup the function by name in our registry
            handler = TOOL_REGISTRY.get(event.tool_name)

            if handler:
                event.execute(handler)
                print(
                    f"{GREEN}[✓] Tool Result Submitted. Resuming stream for final answer...{RESET}\n"
                )
            else:
                # Handle cases where AI hallucinates a tool name not in our registry
                print(
                    f"{RED}[!] No local handler found for tool: {event.tool_name}{RESET}"
                )
                # Level 2 Tip: You could call messages_client.submit_tool_output
                # with an error here to tell the AI it's using a non-existent tool.

except Exception as e:
    print(f"{RED}[!] Error in loop: {e}{RESET}")

# ==================================================================
# TIMER RESULTS
# ==================================================================
global_end = time.perf_counter()
total_time = global_end - global_start

print(f"\n{YELLOW}" + "=" * 60)
print(f" TOTAL ROUND TRIP TIME: {total_time:.4f}s")
print("=" * 60 + f"{RESET}\n")
