import json
import os

import dotenv
import requests

# Load environment variables
dotenv.load_dotenv()

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------

# NOTE: Ensure your .env file has TOGETHER_API_KEY
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "YOUR_TOGETHER_API_KEY_HERE")

# Together AI Endpoint
URL = "https://api.together.xyz/v1/chat/completions"

MODEL = "ServiceNow-AI/Apriel-1.6-15b-Thinker"

TEST_PROMPT = "Please fetch me the flight times between LAX and JFK. Use the get_flight_times tool."

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
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
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
        # --- FIX: ROBUST ERROR HANDLING ---
        if response.status_code != 200:
            print(f"\n[CRITICAL ERROR] Status Code: {response.status_code}")
            try:
                # Try to print the JSON error message from the provider
                print(f"Provider Message: {response.json()}")
            except Exception:
                # Fallback to raw text
                print(f"Raw Response: {response.text}")

            # Stop execution here so we don't process garbage data
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

                if delta.get("tool_calls"):
                    chunk_type = "TOOL_DELTA"
                    debug_info = str(delta["tool_calls"])
                elif delta.get("content") is not None:
                    chunk_type = "CONTENT"
                    debug_info = delta["content"].replace("\n", "\\n")
                elif delta.get("role"):
                    chunk_type = "ROLE_DEF"
                    debug_info = delta["role"]
                elif finish_reason:
                    chunk_type = "FINISH"
                    debug_info = f"Reason: {finish_reason}"
                else:
                    chunk_type = "EMPTY/PING"
                    debug_info = f"Raw Delta: {delta}"

                print(f"[DEBUG STREAM] Type: {chunk_type:<12} | Payload: {debug_info}")
                # --------------------------------------------------------

                # 1. Handle Text Content
                content = delta.get("content")
                if content:
                    full_content += content

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
        print(f"\n[FINAL RESPONSE] {final_response.get('content')}")

    else:
        print("\n[SYSTEM] No tool calls detected in first turn.")
        print(f"[RESPONSE] {assistant_msg.get('content')}")
