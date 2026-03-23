"""
Level 2/3 Unified Function Calling Test
---------------------------------------------------
1. Streams Events via a SINGLE loop.
2. Automatically handles Turn 2 (Final Answer) or Turn N (Correction).
3. Supports Multi-Tool Parallel Dispatch (Level 3).
4. Visualizes Level 3 Strategic Planning events.
"""

import json
import os
import time

from config_orc_fc import config
from dotenv import load_dotenv

# Import the project classes
from projectdavid import PlanEvent  # [NEW] Imported for Level 3 visibility
from projectdavid import ContentEvent, DecisionEvent, Entity, ReasoningEvent, ToolCallRequestEvent

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
BLUE = "\033[94m"  # [NEW] Color for Planning
RESET = "\033[0m"

BASE_URL = config.get("base_url") or os.getenv("BASE_URL", "http://localhost:9000")
ENTITIES_API_KEY = os.getenv("ENTITIES_API_KEY") or config.get("entities_api_key")
ENTITIES_USER_ID = os.getenv("ENTITIES_USER_ID") or config.get("entities_user_id")
HYPERBOLIC_API_KEY = os.getenv("HYPERBOLIC_API_KEY")

MODEL_ID = config.get("model", "together-ai/mistralai/Ministral-3-14B-Instruct-2512")
PROVIDER_KW = config.get("provider", "Hyperbolic")
ASSISTANT_ID = config.get("assistant_id", "asst_13HyDgBnZxVwh5XexYu74F")

# L3 PROMPT: Asks for two different things to trigger parallel planning
TEST_PROMPT = "What is the weather in London and what are the flight times from JFK to LHR?"

print(f"{GREY}[CONFIG] Model: {MODEL_ID} | Provider: {PROVIDER_KW}{RESET}")

client = Entity(base_url=BASE_URL, api_key=ENTITIES_API_KEY)

# Bind clients for internal SDK turn management
if hasattr(client, "synchronous_inference_stream"):
    client.synchronous_inference_stream.bind_clients(
        client.runs, client.actions, client.messages, client.assistants
    )


# ------------------------------------------------------------------
# 1. Tool Executor Logic
# ------------------------------------------------------------------


def get_flight_times(tool_name: str, arguments: dict) -> str:
    """Fake flight-time lookup."""
    print(
        f"{YELLOW}   -> [TOOL EXEC] {tool_name}: {arguments.get('departure')} -> {arguments.get('arrival')}...{RESET}"
    )
    return json.dumps(
        {
            "status": "success",
            "departure": arguments.get("departure", "UNK"),
            "arrival": arguments.get("arrival", "UNK"),
            "duration": "7h 15m",
            "flights": ["BA112", "VS4"],
        }
    )


def get_weather(tool_name: str, arguments: dict) -> str:
    """Fake weather lookup."""
    location = arguments.get("location", "Unknown")
    unit = arguments.get("unit", "celsius")
    print(f"{YELLOW}   -> [TOOL EXEC] {tool_name} for {location} ({unit})...{RESET}")
    return json.dumps(
        {
            "status": "success",
            "location": location,
            "temperature": "12",
            "condition": "Light Rain",
            "unit": unit,
        }
    )


if config.get("provider") == "together":
    API_KEY = os.environ.get("TOGETHER_API_KEY")

if config.get("provider") == "hyperbolic":
    API_KEY = os.getenv("HYPERBOLIC_API_KEY")


# --- DYNAMIC TOOL REGISTRY ---
TOOL_REGISTRY = {
    "get_flight_times": get_flight_times,
    "get_weather": get_weather,
}


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
    api_key=API_KEY,
)

print(f"\n{CYAN}[▶] UNIFIED STREAM: SDK-Managed Turns (L3 Planning Enabled){RESET}")
print(f"{'LATENCY':<12} | {'EVENT CLASS':<25} | {'PAYLOAD'}")
print("-" * 110)

last_tick = time.perf_counter()

try:
    for event in stream.stream_events(provider=PROVIDER_KW, model=MODEL_ID):

        # Timing logic
        current_tick = time.perf_counter()
        delta = current_tick - last_tick
        last_tick = current_tick
        time_str = f"[{delta:+.4f}s]"

        class_name = event.__class__.__name__
        payload = event.to_dict()

        # Level 3 Color Coding
        color = RESET
        if isinstance(event, ContentEvent):
            color = GREEN
        elif isinstance(event, ToolCallRequestEvent):
            color = YELLOW
        elif isinstance(event, ReasoningEvent):
            color = CYAN
        elif isinstance(event, PlanEvent):
            color = BLUE  # [NEW] Highlight the agent's strategy plan
        elif isinstance(event, DecisionEvent):
            color = MAGENTA

        print(
            f"{GREY}{time_str:<12}{RESET} | {color}{class_name:<25}{RESET} | {json.dumps(payload)}"
        )

        # ---------------------------------------------------------
        # BATCH DISPATCH (Level 3 Friendly)
        # ---------------------------------------------------------
        if isinstance(event, ToolCallRequestEvent):
            print(f"\n{YELLOW}[LOCAL] AI requested tool: {event.tool_name}{RESET}")

            handler = TOOL_REGISTRY.get(event.tool_name)

            if handler:
                # This executes the tool and signals the internal L2/L3 turn manager to loop
                event.execute(handler)
                print(f"{GREEN}[✓] Result for {event.tool_name} submitted.{RESET}\n")
            else:
                print(f"{RED}[!] No local handler found for: {event.tool_name}{RESET}")

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
