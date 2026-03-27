"""
Level 2 Unified Function Calling Test: Error Recovery Simulation
---------------------------------------------------
1. Streams Events via a SINGLE loop.
2. SIMULATES an error on the first tool execution.
3. Demonstrates the SDK catching the error and the LLM attempting to recover.
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

MODEL_ID = config.get("model", "together-ai/mistralai/Ministral-3-14B-Instruct-2512")
PROVIDER_KW = config.get("provider", "Hyperbolic")
ASSISTANT_ID = config.get("assistant_id", "asst_13HyDgBnZxVwh5XexYu74F")
TEST_PROMPT = config.get(
    "test_prompt", "Please fetch me the flight times between LAX and JFK."
)

print(f"{GREY}[CONFIG] Model: {MODEL_ID} | Provider: {PROVIDER_KW}{RESET}")

client = Entity(base_url=BASE_URL, api_key=ENTITIES_API_KEY)

# ------------------------------------------------------------------
# 1. Tool Executor Logic (WITH SIMULATED FAILURE)
# ------------------------------------------------------------------
# We use a simple counter to fail the first time and succeed the second
EXECUTION_ATTEMPTS = 0


def get_flight_times(tool_name: str, arguments: dict) -> str:
    """
    Fake tool that simulates a database failure on the first attempt.
    """
    global EXECUTION_ATTEMPTS
    EXECUTION_ATTEMPTS += 1

    print(
        f"{YELLOW}   -> [TOOL ATTEMPT {EXECUTION_ATTEMPTS}] {tool_name} for {arguments.get('departure')}...{RESET}"
    )

    # SIMULATE A CRASH ON TURN 1
    if EXECUTION_ATTEMPTS == 1:
        print(
            f"{RED}   -> [SIMULATED ERROR] Triggering fake Database Timeout...{RESET}"
        )
        raise Exception(
            "Database Connection Timeout: The flight server is currently not responding."
        )

    # SUCCESS ON TURN 2
    print(f"{GREEN}   -> [SUCCESS] Database recovered for turn 2.{RESET}")
    return json.dumps(
        {
            "status": "success",
            "departure": arguments.get("departure", "UNK"),
            "arrival": arguments.get("arrival", "UNK"),
            "duration": "4h 30m",
        }
    )


# --- DYNAMIC TOOL REGISTRY ---
TOOL_REGISTRY = {
    "get_flight_times": get_flight_times,
}


# ==================================================================
# 2. Setup & Global Timer
# ==================================================================
print(f"\n{YELLOW}[TIMER] Starting Unified Round-Trip with Error Simulation...{RESET}")
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
    user_id=ENTITIES_USER_ID,
    thread_id=thread.id,
    assistant_id=ASSISTANT_ID,
    message_id=message.id,
    run_id=run.id,
    api_key=HYPERBOLIC_API_KEY,
)

print(f"\n{CYAN}[▶] UNIFIED STREAM: Testing Agentic Recovery{RESET}")
print(f"{'LATENCY':<12} | {'EVENT CLASS':<25} | {'PAYLOAD'}")
print("-" * 110)

last_tick = time.perf_counter()

try:
    # This loop will now likely run for THREE turns:
    # Turn 1: AI calls tool -> Handler crashes -> SDK submits Error.
    # Turn 2: AI sees error -> Retries tool call -> Handler succeeds.
    # Turn 3: AI sees success -> Generates final answer.
    for event in stream.stream_events(provider=PROVIDER_KW, model=MODEL_ID):

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
        # LEVEL 2 EXECUTION:
        # If the handler crashes, event.execute() returns True (because
        # it successfully submitted the error feedback). The SDK loop
        # then triggers the next turn automatically.
        # ---------------------------------------------------------
        if isinstance(event, ToolCallRequestEvent):
            print(f"\n{YELLOW}[LOCAL] AI requested tool: {event.tool_name}{RESET}")

            handler = TOOL_REGISTRY.get(event.tool_name)

            if handler:
                success = event.execute(handler)
                if success:
                    print(
                        f"{GREEN}[✓] Feedback/Result Submitted. Turn complete.{RESET}\n"
                    )
            else:
                print(
                    f"{RED}[!] No local handler found for tool: {event.tool_name}{RESET}"
                )

except Exception as e:
    print(f"{RED}[!] Error in loop: {e}{RESET}")

# ==================================================================
# TIMER RESULTS
# ==================================================================
global_end = time.perf_counter()
total_time = global_end - global_start

print(f"\n{YELLOW}" + "=" * 60)
print(f" TOTAL ROUND TRIP TIME (INCLUDING RECOVERY): {total_time:.4f}s")
print("=" * 60 + f"{RESET}\n")
