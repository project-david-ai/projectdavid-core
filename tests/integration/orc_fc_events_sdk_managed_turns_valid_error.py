"""
Level 2 Unified Function Calling Test: Error Recovery Simulation
---------------------------------------------------
1. Streams Events via a SINGLE loop.
2. SIMULATES an error on the first tool execution.
3. Demonstrates the SDK catching the error and the LLM attempting to recover.
"""

import json
import os

from config_orc_fc import config
from dotenv import load_dotenv

# Import the project classes
from projectdavid import ContentEvent, Entity, ToolCallRequestEvent

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
# 1. Tool Logic (This shouldn't even be reached on the first try!)
# ------------------------------------------------------------------
def get_flight_times(tool_name: str, arguments: dict) -> str:
    print(f"{GREEN}   -> [HANDLER REACHED] Arguments: {arguments}{RESET}")
    return json.dumps({"status": "success", "data": "Flight AF123"})


TOOL_REGISTRY = {"get_flight_times": get_flight_times}

# ==================================================================
# 2. Setup Stream
# ==================================================================
thread = client.threads.create_thread()
message = client.messages.create_message(
    thread_id=thread.id,
    role="user",
    content="Please fetch me the flight times between LAX and JFK.",
    assistant_id=ASSISTANT_ID,
)
run = client.runs.create_run(assistant_id=ASSISTANT_ID, thread_id=thread.id)

stream = client.synchronous_inference_stream
stream.setup(
    user_id=ENTITIES_USER_ID,
    thread_id=thread.id,
    assistant_id=ASSISTANT_ID,
    message_id=message.id,
    run_id=run.id,
    api_key=HYPERBOLIC_API_KEY,
)

# ------------------------------------------------------------------
# 3. [SIMULATION] POISON THE SCHEMA REGISTRY
# ------------------------------------------------------------------
# We manually tell the validator that 'passenger_name' is required.
# The LLM doesn't know this, so it will fail validation on Turn 1.
stream.validator.schema_registry["get_flight_times"] = [
    "departure",
    "arrival",
    "passenger_name",
]

print(f"\n{MAGENTA}[SIMULATION] Injected 'passenger_name' as a required field.{RESET}")
print(
    f"{MAGENTA}The LLM will fail Turn 1 because it doesn't know to provide this.{RESET}\n"
)

# ------------------------------------------------------------------
# 4. Unified Recursive Stream
# ------------------------------------------------------------------
print(f"{CYAN}[▶] UNIFIED STREAM: Testing Validation Intercept{RESET}")

try:
    for event in stream.stream_events(provider=PROVIDER_KW, model=MODEL_ID):

        if isinstance(event, ContentEvent):
            print(f"{GREEN}Assistant: {event.content}{RESET}")

        if isinstance(event, ToolCallRequestEvent):
            print(
                f"\n{YELLOW}[LOCAL] AI successfully passed validation with: {event.args}{RESET}"
            )
            handler = TOOL_REGISTRY.get(event.tool_name)
            if handler:
                event.execute(handler)

except Exception as e:
    print(f"{RED}[!] Error: {e}{RESET}")
