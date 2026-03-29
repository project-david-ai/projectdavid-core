"""
vLLM Raw Inference Function Calling Test
---------------------------------------------------
Tests streaming tool calls via vLLM raw (/v1/completions) using
the unified SDK stream. Single loop handles all turns.

Adapted from: test_ollama_fc.py
"""

import json
import os
import time

from config_orc_fc import config
from dotenv import load_dotenv
from projectdavid import (
    ContentEvent,
    DecisionEvent,
    Entity,
    ReasoningEvent,
    ToolCallRequestEvent,
)

load_dotenv()

# ------------------------------------------------------------------
# ANSI Colors
# ------------------------------------------------------------------
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
GREY = "\033[90m"
MAGENTA = "\033[95m"
RESET = "\033[0m"

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
BASE_URL = os.getenv("BASE_URL", "http://localhost:9000")
API_KEY = os.getenv("ENTITIES_API_KEY")
ASSISTANT_ID = config.get("assistant_id", "asst_YscCYmaK7xCVzGylYFe1pN")

# ── vLLM specific ──────────────────────────────────────────────────
# Format: "vllm/<model_id_as_loaded_in_vllm_server>"
# Must match exactly what /v1/models returns
MODEL_ID = "vllm/Qwen/Qwen2.5-3B-Instruct"

# vLLM server URL — passed via meta_data so the worker knows where to connect
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://vllm_server:8000")

TEST_PROMPT = "Please provide the flight times for a trip departing from Tokyo and arriving in Sydney?"


# ------------------------------------------------------------------
# Tool Definitions
# ------------------------------------------------------------------
def get_flight_times(tool_name: str, arguments: dict) -> str:
    print(f"{YELLOW}   -> [TOOL] {tool_name}({arguments}){RESET}")
    return json.dumps(
        {
            "status": "success",
            "departure": arguments.get("departure", "UNK"),
            "arrival": arguments.get("arrival", "UNK"),
            "duration": "4h 30m",
        }
    )


TOOL_REGISTRY = {
    "get_flight_times": get_flight_times,
}

# ------------------------------------------------------------------
# SDK Init & Run Setup
# ------------------------------------------------------------------
client = Entity(base_url=BASE_URL, api_key=API_KEY)
thread = client.threads.create_thread()
message = client.messages.create_message(
    thread_id=thread.id,
    role="user",
    content=TEST_PROMPT,
    assistant_id=ASSISTANT_ID,
)

run = client.runs.create_run(assistant_id=ASSISTANT_ID, thread_id=thread.id)

stream = client.synchronous_inference_stream
stream.setup(
    thread_id=thread.id,
    assistant_id=ASSISTANT_ID,
    message_id=message.id,
    run_id=run.id,
)

# ------------------------------------------------------------------
# Unified Stream Loop
# ------------------------------------------------------------------
print(f"\n{CYAN}[▶] MODEL:    {MODEL_ID}{RESET}")
print(f"{CYAN}[▶] VLLM URL: {VLLM_BASE_URL}{RESET}")
print(f"{CYAN}[▶] PROMPT:   {TEST_PROMPT}{RESET}\n")
print(f"{'LATENCY':<12} | {'EVENT':<25} | PAYLOAD")
print("-" * 100)

last_tick = time.perf_counter()
global_start = last_tick

try:
    for event in stream.stream_events(
        model=MODEL_ID,
        # Pass vLLM URL dynamically — worker picks this up from meta_data
        meta_data={"vllm_base_url": VLLM_BASE_URL},
    ):
        now = time.perf_counter()
        delta = now - last_tick
        last_tick = now

        color = {
            ContentEvent: GREEN,
            ToolCallRequestEvent: YELLOW,
            ReasoningEvent: CYAN,
            DecisionEvent: MAGENTA,
        }.get(type(event), RESET)

        print(
            f"{GREY}[{delta:+.4f}s]{RESET:<4} "
            f"| {color}{event.__class__.__name__:<25}{RESET} "
            f"| {json.dumps(event.to_dict())}"
        )

        if isinstance(event, ToolCallRequestEvent):
            handler = TOOL_REGISTRY.get(event.tool_name)
            if handler:
                print(f"\n{YELLOW}[TOOL CALL] {event.tool_name} → executing...{RESET}")
                event.execute(handler)
                print(f"{GREEN}[✓] Result submitted — resuming stream...{RESET}\n")
            else:
                print(
                    f"{RED}[!] Unknown tool: '{event.tool_name}' — not in registry{RESET}"
                )

except Exception as e:
    print(f"{RED}[ERROR] {e}{RESET}")

finally:
    total = time.perf_counter() - global_start
    print(f"\n{YELLOW}{'='*50}")
    print(f"  TOTAL: {total:.4f}s")
    print(f"{'='*50}{RESET}\n")
