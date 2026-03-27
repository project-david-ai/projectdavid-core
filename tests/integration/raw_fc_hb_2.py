import json
import os
import sys

import requests
from config_raw_fc_hb import config
from dotenv import load_dotenv

# ------------------------------------------------------------------
# CONFIGURATION LOAD
# ------------------------------------------------------------------

# 1. Load .env
load_dotenv()


# Logic: Env -> Config -> Default (Hyperbolic settings from your snippet)

# Try to find a key in this order: HYPERBOLIC_API_KEY (env) -> generic 'api_key' (config) -> TOGETHER_API_KEY (env)
API_KEY = (
    os.getenv("HYPERBOLIC_API_KEY")
    or config.get("hyperbolic_api_key")
    or config.get("together_api_key")
    or os.getenv("TOGETHER_API_KEY")
)

if not API_KEY:
    print(
        "❌ [ERROR] Missing API Key. Set HYPERBOLIC_API_KEY or together_api_key in .env or config.json"
    )
    sys.exit(1)

# Default to Hyperbolic if not in config, as per your snippet
URL = config.get("url", "https://api.hyperbolic.xyz/v1/chat/completions")
MODEL = config.get("model", "openai/gpt-oss-120b")
TEST_PROMPT = config.get(
    "test_prompt",
    "Please fetch me the flight times between LAX and JFK. Use the get_flight_times tool.",
)

print(f"⚙️  [CONFIG] Target URL: {URL}")
print(f"⚙️  [CONFIG] Model: {MODEL}")

# ------------------------------------------------------------------
# 1. Tool Definitions
# ------------------------------------------------------------------

flight_func_schema = {
    "type": "function",
    "function": {
        "name": "get_flight_times",
        "description": "Return flight times between two airport codes.",
        "parameters": {
            "type": "object",
            "properties": {
                "departure": {"type": "string"},
                "arrival": {"type": "string"},
            },
            "required": ["departure", "arrival"],
        },
    },
}


def execute_local_tool(tool_name: str, arguments) -> str:
    """Executes the specific local function logic."""
    print(f"\n[LOCAL EXECUTION] Running {tool_name} with args: {arguments}")

    if tool_name == "get_flight_times":
        dep = arguments.get("departure")
        arr = arguments.get("arrival")
        return json.dumps(
            {
                "status": "success",
                "departure": dep,
                "arrival": arr,
                "duration": "4h 30m",
                "departure_time": "10:30 AM PST",
                "arrival_time": "06:30 PM EST",
            }
        )

    return json.dumps({"error": f"Tool {tool_name} not found"})


# ------------------------------------------------------------------
# 2. Stream Handler (Stateful)
# ------------------------------------------------------------------


def run_turn(messages):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    payload = {
        "model": MODEL,
        "messages": messages,
        "tools": [flight_func_schema],
        "tool_choice": "auto",
        "max_tokens": 1024,
        "temperature": 0.4,
        "stream": True,
    }

    # ------------------------------------------------------------------
    # VISUAL DEBUG: PRINT FULL DIALOGUE LIST
    # ------------------------------------------------------------------
    print(f"\n=== 🔍 VISUAL DEBUG: OUTBOUND PAYLOAD (Length: {len(messages)}) ===")
    print(json.dumps(messages, indent=2))
    print("===================================================================\n")
    # ------------------------------------------------------------------

    print(f"--- STARTING STREAM TURN (Msgs: {len(messages)}) ---")

    # Accumulators
    full_content = ""
    tool_calls_buffer = {}

    with requests.post(URL, headers=headers, json=payload, stream=True) as response:
        if response.status_code == 422:
            print("\n[ERROR 422] Payload rejected. Dumping sent messages:")
            print(json.dumps(messages, indent=2))

        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8")
            if not decoded.startswith("data: "):
                continue

            data_str = decoded[6:].strip()
            if data_str == "[DONE]":
                print("\n[DEBUG STREAM] Received [DONE]")
                break

            try:
                chunk = json.loads(data_str)
                choices = chunk.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                finish_reason = choices[0].get("finish_reason", None)

                # --------------------------------------------------------
                # DEBUG: IDENTIFY CHUNK TYPE
                # --------------------------------------------------------
                chunk_type = "UNKNOWN"
                debug_info = ""

                # Check for Tool Calls (Verify it is not None/Empty)
                if delta.get("tool_calls"):
                    chunk_type = "TOOL_DELTA"
                    debug_info = str(delta["tool_calls"])

                # Check for Content (Verify it is not None)
                elif delta.get("content") is not None:
                    chunk_type = "CONTENT"
                    # Escape newlines for cleaner debug printing
                    debug_info = delta["content"].replace("\n", "\\n")

                # Check for Role
                elif delta.get("role"):
                    chunk_type = "ROLE_DEF"
                    debug_info = delta["role"]

                # Check for Finish Reason
                elif finish_reason:
                    chunk_type = "FINISH"
                    debug_info = f"Reason: {finish_reason}"

                # Fallback
                else:
                    chunk_type = "EMPTY/PING"
                    debug_info = f"Raw Delta: {delta}"

                print(f"[DEBUG STREAM] Type: {chunk_type:<12} | Payload: {debug_info}")
                # --------------------------------------------------------

                # 1. Handle Text Content
                content = delta.get("content")
                if content:
                    full_content += content
                    # Note: We aren't printing standard output here to keep the debug log clean.

                # 2. Handle Tool Calls
                if delta.get("tool_calls"):
                    for tc in delta["tool_calls"]:
                        idx = tc["index"]
                        if idx not in tool_calls_buffer:
                            tool_calls_buffer[idx] = {
                                "id": tc.get("id", ""),
                                "function": {"name": "", "arguments": ""},
                            }

                        fn = tc.get("function", {})
                        if fn.get("name"):
                            tool_calls_buffer[idx]["function"]["name"] += fn["name"]
                        if fn.get("arguments"):
                            tool_calls_buffer[idx]["function"]["arguments"] += fn[
                                "arguments"
                            ]

            except json.JSONDecodeError:
                print(f"[DEBUG STREAM] JSON Error on line: {data_str}")
                pass

    print("\n--- STREAM FINISHED ---")

    # Construct the final message object
    message_obj = {"role": "assistant", "content": full_content if full_content else ""}

    # Format tools into the message object
    if tool_calls_buffer:
        formatted_tools = []
        for idx in sorted(tool_calls_buffer.keys()):
            tool_data = tool_calls_buffer[idx]
            raw_args = tool_data["function"]["arguments"]

            # --- RECURSIVE JSON FIX ---
            final_args_str = raw_args
            try:
                parsed_args = json.loads(raw_args)
                if (
                    isinstance(parsed_args, dict)
                    and "name" in parsed_args
                    and "arguments" in parsed_args
                ):
                    print("\n[FIX] Unwrap recursive JSON detected")
                    inner_args = parsed_args["arguments"]
                    if isinstance(inner_args, dict):
                        final_args_str = json.dumps(inner_args)
                    elif isinstance(inner_args, str):
                        final_args_str = inner_args
            except Exception:
                pass

            # FIX: Ensure we have an ID.
            t_id = (
                tool_data["id"] if tool_data["id"] else f"call_{idx}_{hash(raw_args)}"
            )

            formatted_tools.append(
                {
                    "id": t_id,
                    "type": "function",
                    "function": {
                        "name": tool_data["function"]["name"],
                        "arguments": final_args_str,
                    },
                }
            )

        message_obj["tool_calls"] = formatted_tools

    return message_obj


# ------------------------------------------------------------------
# 3. Main Logic
# ------------------------------------------------------------------

if __name__ == "__main__":
    conversation_history = [{"role": "user", "content": TEST_PROMPT}]

    # --- TURN 1 ---
    assistant_msg = run_turn(conversation_history)
    conversation_history.append(assistant_msg)

    # Check if tools were called
    if assistant_msg.get("tool_calls"):
        print(
            f"\n[SYSTEM] Assistant requested {len(assistant_msg['tool_calls'])} tool(s). Executing..."
        )

        for tc in assistant_msg["tool_calls"]:
            func_name = tc["function"]["name"]
            func_args_str = tc["function"]["arguments"]
            call_id = tc["id"]

            try:
                func_args = json.loads(func_args_str)
            except Exception:
                print(f"[ERROR] Failed to parse args: {func_args_str}")
                func_args = {}

            tool_result = execute_local_tool(func_name, func_args)

            # Append Tool Result
            conversation_history.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": func_name,
                    "content": tool_result,
                }
            )

        # --- TURN 2 ---
        print("\n[SYSTEM] Sending tool outputs back to model...")
        final_response = run_turn(conversation_history)
        print(f"\n[FINAL RESPONSE] {final_response['content']}")

    else:
        print("\n[SYSTEM] No tool calls detected in first turn.")
        print(f"[RESPONSE] {assistant_msg['content']}")
