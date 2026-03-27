"""
Debug Mode: Function-Call Round-Trip (Hyperbolic/GPT-OSS)
---------------------------------------------------------
This version prints EVERY chunk received from the stream raw,
allowing you to inspect exactly what the `DeltaNormalizer` is yielding.
INCLUDES TIMING METRICS.
"""

import json
import os
import time  # <--- Added for timing

from dotenv import load_dotenv
from projectdavid import Entity

# ------------------------------------------------------------------
# 0.  SDK init + env
# ------------------------------------------------------------------
load_dotenv()

# ANSI Colors for Debugging
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
GREY = "\033[90m"
RESET = "\033[0m"

client = Entity(
    base_url=os.getenv("BASE_URL", "http://localhost:9000"),
    api_key=os.getenv("ENTITIES_API_KEY"),
)

USER_ID = os.getenv("ENTITIES_USER_ID")

NORMAL_ASSISTANT_ID = "asst_bToQ4dQQnuhJuhlU5M6COG"
GPT_OSS_ASSISTANT_ID = "asst_kaK6ZNRxhERIjKqxpHmrVf"

ASSISTANT_ID = NORMAL_ASSISTANT_ID
MODEL_ID = "hyperbolic/Qwen/Qwen2.5-Coder-32B-Instruct"
PROVIDER_KW = "Hyperbolic"
HYPERBOLIC_API_KEY = os.getenv("HYPERBOLIC_API_KEY")


# ------------------------------------------------------------------
# 1.  Tool executor
# ------------------------------------------------------------------
def get_flight_times(tool_name: str, arguments) -> str:
    """Fake flight-time lookup."""
    if tool_name != "get_flight_times":
        return json.dumps({"status": "error", "message": f"unknown tool '{tool_name}'"})

    print(
        f"\n{YELLOW}[LOCAL EXEC] Tool invoked: {tool_name} | Args: {arguments}{RESET}"
    )

    return json.dumps(
        {
            "status": "success",
            "departure": arguments.get("departure", "UNK"),
            "arrival": arguments.get("arrival", "UNK"),
            "duration": "4h 30m",
            "departure_time": "10:00 AM PST",
            "arrival_time": "06:30 PM EST",
        }
    )


# ==================================================================
# TIMER START
# ==================================================================
print(f"\n{YELLOW}[TIMER] Starting Round-Trip Timer...{RESET}")
global_start = time.perf_counter()


# ------------------------------------------------------------------
# 2.  Thread + message + run
# ------------------------------------------------------------------
t_setup_start = time.perf_counter()

print(f"{GREY}[1/4] Creating Thread & Message...{RESET}")
thread = client.threads.create_thread()

message = client.messages.create_message(
    thread_id=thread.id,
    role="user",
    content="Please fetch me the flight times between LAX and JFK.",
    assistant_id=ASSISTANT_ID,
)

print(f"{GREY}[2/4] Creating Run...{RESET}")
run = client.runs.create_run(assistant_id=ASSISTANT_ID, thread_id=thread.id)

t_setup_end = time.perf_counter()


# ------------------------------------------------------------------
# 3.  Stream initial LLM response (RAW DEBUG MODE)
# ------------------------------------------------------------------
stream = client.synchronous_inference_stream
stream.setup(
    user_id=USER_ID,
    thread_id=thread.id,
    assistant_id=ASSISTANT_ID,
    message_id=message.id,
    run_id=run.id,
    api_key=HYPERBOLIC_API_KEY,
)

print(f"\n{CYAN}[▶] STREAM 1: Initial Generation (Raw Inspection){RESET}")
print(f"{'TYPE':<20} | {'PAYLOAD'}")
print("-" * 80)

t_stream1_start = time.perf_counter()

for chunk in stream.stream_chunks(
    provider=PROVIDER_KW, model=MODEL_ID, suppress_fc=False, timeout_per_chunk=10.0
):
    c_type = chunk.get("type", "unknown")

    # Visual Coloring based on type
    row_color = RESET
    if c_type == "content":
        row_color = GREEN
    elif c_type in ["tool_name", "call_arguments", "tool_call"]:
        row_color = YELLOW
    elif c_type == "reasoning":
        row_color = CYAN
    elif c_type == "error":
        row_color = RED
    elif c_type == "status":
        row_color = GREY

    # Print the full dictionary
    print(f"{row_color}{c_type:<20} | {json.dumps(chunk, default=str)}{RESET}")

t_stream1_end = time.perf_counter()


# ------------------------------------------------------------------
# 4.  Poll run → execute tool → send tool result
# ------------------------------------------------------------------
print(f"\n{GREY}[3/4] Polling for Tool Execution...{RESET}")

t_tool_start = time.perf_counter()

handled = client.runs.poll_and_execute_action(
    run_id=run.id,
    thread_id=thread.id,
    assistant_id=ASSISTANT_ID,
    tool_executor=get_flight_times,
    actions_client=client.actions,
    messages_client=client.messages,
    timeout=10,
    interval=0.1,
)

t_tool_end = time.perf_counter()


# ------------------------------------------------------------------
# 5.  Stream final assistant response (RAW DEBUG MODE)
# ------------------------------------------------------------------
t_stream2_start = 0
t_stream2_end = 0

if handled:
    print(f"\n{CYAN}[▶] STREAM 2: Final Response (Raw Inspection){RESET}")
    print(f"{'TYPE':<20} | {'PAYLOAD'}")
    print("-" * 80)

    stream.setup(
        user_id=USER_ID,
        thread_id=thread.id,
        assistant_id=ASSISTANT_ID,
        message_id=message.id,
        run_id=run.id,
        api_key=HYPERBOLIC_API_KEY,
    )

    t_stream2_start = time.perf_counter()

    for chunk in stream.stream_chunks(
        provider=PROVIDER_KW, model=MODEL_ID, timeout_per_chunk=180.0
    ):
        c_type = chunk.get("type", "unknown")

        row_color = RESET
        if c_type == "content":
            row_color = GREEN
        elif c_type == "status":
            row_color = GREY
        elif c_type == "reasoning":
            row_color = CYAN

        print(f"{row_color}{c_type:<20} | {json.dumps(chunk, default=str)}{RESET}")

    t_stream2_end = time.perf_counter()
    print(f"\n{GREY}--- End of Stream ---{RESET}")
else:
    print(f"\n{RED}[!] No function call detected or execution failed.{RESET}")

# ==================================================================
# TIMER RESULTS
# ==================================================================
global_end = time.perf_counter()

setup_time = t_setup_end - t_setup_start
stream1_time = t_stream1_end - t_stream1_start
tool_time = t_tool_end - t_tool_start
stream2_time = t_stream2_end - t_stream2_start if handled else 0
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
