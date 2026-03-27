# tests/integration/basic_function_call_without_polling.py
"""
Debug Mode: Function-Call Round-Trip (Optimized/No-Polling)
-----------------------------------------------------------
1. Streams chunks and accumulates 'call_arguments' in real-time.
2. Detects completion via stream status.
3. Uses SDK helper 'execute_pending_action' for deterministic execution.
4. Streams final response.
"""

import json
import os
import time

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
ASSISTANT_ID = "asst_bToQ4dQQnuhJuhlU5M6COG"  # Normal Assistant
MODEL_ID = "hyperbolic/deepseek-ai/DeepSeek-V3-0324"
PROVIDER_KW = "Hyperbolic"
HYPERBOLIC_API_KEY = os.getenv("HYPERBOLIC_API_KEY")


# ------------------------------------------------------------------
# 1.  Tool executor
# ------------------------------------------------------------------
def get_flight_times(tool_name: str, arguments: dict) -> str:
    """Fake flight-time lookup."""
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


# ------------------------------------------------------------------
# 2.  Thread + message + run
# ------------------------------------------------------------------
print(f"{GREY}[1/3] Creating Thread & Message...{RESET}")
thread = client.threads.create_thread()

message = client.messages.create_message(
    thread_id=thread.id,
    role="user",
    content="Please fetch me the flight times between LAX and JFK.",
    assistant_id=ASSISTANT_ID,
)

print(f"{GREY}[2/3] Creating Run...{RESET}")
run = client.runs.create_run(assistant_id=ASSISTANT_ID, thread_id=thread.id)

# ------------------------------------------------------------------
# 3.  Stream 1: Accumulate Arguments
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

print(f"\n{CYAN}[▶] STREAM 1: Initial Generation (Accumulating Args){RESET}")
print(f"{'TYPE':<20} | {'PAYLOAD'}")
print("-" * 80)

# Buffer to capture the streaming JSON arguments
tool_args_buffer = ""
tool_detected = False

start_time = time.time()

for chunk in stream.stream_chunks(
    provider=PROVIDER_KW, model=MODEL_ID, suppress_fc=False, timeout_per_chunk=10.0
):
    c_type = chunk.get("type", "unknown")

    # Visual Coloring
    row_color = RESET
    if c_type == "content":
        row_color = GREEN
    elif c_type in ["tool_name", "call_arguments", "tool_call"]:
        row_color = YELLOW
        tool_detected = True
        # ACCUMULATION LOGIC:
        if c_type == "call_arguments":
            tool_args_buffer += chunk.get("content", "")
    elif c_type == "reasoning":
        row_color = CYAN
    elif c_type == "error":
        row_color = RED
    elif c_type == "status":
        row_color = GREY

    print(f"{row_color}{c_type:<20} | {json.dumps(chunk, default=str)}{RESET}")

# ------------------------------------------------------------------
# 4.  Immediate Execution (via SDK Helper)
# ------------------------------------------------------------------
if tool_detected and tool_args_buffer:
    print(f"\n{CYAN}[⚡] Fast-Path: Tool Call Detected.{RESET}")

    try:
        # Optional: Pre-parse arguments from stream to save time/verification
        captured_args = json.loads(tool_args_buffer)

        # CALL THE NEW HELPER
        # Note: We pass client.actions and client.messages explicitly
        success = client.runs.execute_pending_action(
            run_id=run.id,
            thread_id=thread.id,
            assistant_id=ASSISTANT_ID,
            tool_executor=get_flight_times,
            actions_client=client.actions,  # <--- FIX: Pass this explicitly
            messages_client=client.messages,  # <--- FIX: Pass this explicitly
            streamed_args=captured_args,  # <--- OPTIONAL: Use args from stream
        )

        if success:
            print(f"{GREEN}[✓] Tool execution verified.{RESET}")

            # ------------------------------------------------------------------
            # 5.  Stream 2: Final Response
            # ------------------------------------------------------------------
            print(f"\n{CYAN}[▶] STREAM 2: Final Response{RESET}")
            print(f"{'TYPE':<20} | {'PAYLOAD'}")
            print("-" * 80)

            # Re-setup stream for the final leg
            stream.setup(
                user_id=USER_ID,
                thread_id=thread.id,
                assistant_id=ASSISTANT_ID,
                message_id=message.id,
                run_id=run.id,
                api_key=HYPERBOLIC_API_KEY,
            )

            for chunk in stream.stream_chunks(
                provider=PROVIDER_KW, model=MODEL_ID, timeout_per_chunk=180.0
            ):
                c_type = chunk.get("type", "unknown")

                row_color = RESET
                if c_type == "content":
                    row_color = GREEN
                elif c_type == "status":
                    row_color = GREY

                print(
                    f"{row_color}{c_type:<20} | {json.dumps(chunk, default=str)}{RESET}"
                )

        else:
            print(f"{RED}[!] Tool execution failed.{RESET}")

    except json.JSONDecodeError:
        print(
            f"{RED}[!] Failed to parse captured JSON arguments: {tool_args_buffer}{RESET}"
        )
    except Exception as e:
        print(f"{RED}[!] Error during fast-path execution: {e}{RESET}")

else:
    print(f"\n{RED}[!] No function call arguments detected in stream.{RESET}")

print(f"\n{GREY}--- End of Script ---{RESET}")
