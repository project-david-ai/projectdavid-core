import json
import os
import re
import sys

import dotenv
import requests
from config_raw_fc_tg import config

# ------------------------------------------------------------------
# CONFIGURATION LOAD
# ------------------------------------------------------------------

# 1. Load .env (for security, environment variables are preferred for keys)
dotenv.load_dotenv()

# 3. Resolve Constants
# Logic: Try Environment Variable first -> Then Config File -> Then Error
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY") or config.get("together_api_key")

if not TOGETHER_API_KEY or TOGETHER_API_KEY == "YOUR_TOGETHER_API_KEY_HERE":
    print(
        "❌ [ERROR] Missing TOGETHER_API_KEY. Set it in .env or raw_function_call_test_config.json"
    )
    sys.exit(1)

URL = config.get("url", "https://api.together.xyz/v1/chat/completions")
MODEL = config.get("model", "moonshotai/Kimi-K2.5")
TEST_PROMPT = config.get("test_prompt", "Please fetch me the flight times...")

print(f"⚙️  [CONFIG] Model: {MODEL}")
print(f"⚙️  [CONFIG] Prompt: {TEST_PROMPT[:50]}...")

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
# 2. Parsing Logic (The "Dual" Detector)
# ------------------------------------------------------------------


def parse_hermes_style(content: str):
    """
    Attempts to extract function calls from text content using common
    XML or JSON patterns used by Hermes/Llama-3 finetunes.
    """
    tools_found = []

    # Pattern 1: <fc> {JSON} </fc> (Hyperbolic/Hermes specific)
    fc_matches = re.findall(r"<fc>(.*?)</fc>", content, re.DOTALL)

    # Pattern 2: <tool_code> {JSON} </tool_code> (Hermes generic)
    tc_matches = re.findall(r"<tool_code>(.*?)</tool_code>", content, re.DOTALL)

    # Combine matches
    all_matches = fc_matches + tc_matches

    for raw_json in all_matches:
        try:
            # Cleanup common formatting issues
            cleaned_json = raw_json.strip()
            # Handle "name": "foo", "arguments": ... structure
            data = json.loads(cleaned_json)

            # If it's the {name, arguments} wrapper
            if "name" in data and "arguments" in data:
                tools_found.append(
                    {
                        "type": "hermes_wrapper",
                        "name": data["name"],
                        "arguments": data["arguments"],
                    }
                )
            # If it's just the arguments (rare, but happens if name is implied)
            else:
                # We assume the name was mentioned prior, strictly this is harder to handle generically
                # skipping for now unless explicit structure found
                pass
        except json.JSONDecodeError:
            # Fallback: Regex extraction for FunctionName { args }
            func_match = re.match(
                r"^\s*([a-zA-Z0-9_]+)\s*(\{.*)", cleaned_json, re.DOTALL
            )
            if func_match:
                try:
                    name = func_match.group(1)
                    args = json.loads(func_match.group(2))
                    tools_found.append(
                        {"type": "hermes_text", "name": name, "arguments": args}
                    )
                except Exception:
                    pass

    return tools_found


# ------------------------------------------------------------------
# 3. Stream Handler
# ------------------------------------------------------------------


def run_turn(messages):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
    }

    # We send the tools definition.
    # Even for Hermes style, the model often needs to 'see' the tools in the schema
    # to know they exist, even if it decides to text-complete the call.
    payload = {
        "model": MODEL,
        "messages": messages,
        "tools": [flight_func_schema],
        "tool_choice": "auto",
        "max_tokens": 1024,
        "temperature": 0.4,
        "stream": True,
    }

    print(f"\n=== 🔍 VISUAL DEBUG: OUTBOUND PAYLOAD (Length: {len(messages)}) ===")
    # Print last message for brevity
    print(json.dumps(messages[-1], indent=2))
    print("===================================================================\n")
    print(f"--- STARTING STREAM TURN (Msgs: {len(messages)}) ---")

    # Accumulators
    full_content = ""
    tool_calls_buffer = {}

    # Protocol Detection
    detected_protocol = "none"  # 'native', 'hermes', 'none'

    with requests.post(URL, headers=headers, json=payload, stream=True) as response:
        if response.status_code != 200:
            print(f"\n[CRITICAL ERROR] Status Code: {response.status_code}")
            print(f"Raw Response: {response.text}")
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

                # --------------------------------------------------------
                # 1. NATIVE DETECTION
                # --------------------------------------------------------
                if delta.get("tool_calls"):
                    detected_protocol = "native"
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

                    # Visual feedback for Native
                    sys.stdout.write("N")
                    sys.stdout.flush()

                # --------------------------------------------------------
                # 2. CONTENT ACCUMULATION (Potential Hermes)
                # --------------------------------------------------------
                content = delta.get("content")
                if content:
                    full_content += content
                    # Visual feedback for Content
                    sys.stdout.write(".")
                    sys.stdout.flush()

            except json.JSONDecodeError:
                pass

    print("\n--- STREAM FINISHED ---")

    # Construct the final message object
    message_obj = {"role": "assistant", "content": full_content if full_content else ""}

    parsed_hermes_tools = []

    # ------------------------------------------------------------------
    # POST-PROCESSING: DETERMINE WINNER
    # ------------------------------------------------------------------

    # Priority 1: Native
    if tool_calls_buffer:
        detected_protocol = "native"
        formatted_tools = []
        for idx in sorted(tool_calls_buffer.keys()):
            tool_data = tool_calls_buffer[idx]
            # Ensure ID exists
            t_id = (
                tool_data["id"]
                if tool_data["id"]
                else f"call_{idx}_{hash(tool_data['function']['arguments'])}"
            )

            formatted_tools.append(
                {
                    "id": t_id,
                    "type": "function",
                    "function": {
                        "name": tool_data["function"]["name"],
                        "arguments": tool_data["function"]["arguments"],
                    },
                }
            )
        message_obj["tool_calls"] = formatted_tools

    # Priority 2: Hermes/Text (If no native calls)
    else:
        # Check for XML tags in content
        parsed_hermes_tools = parse_hermes_style(full_content)
        if parsed_hermes_tools:
            detected_protocol = "hermes"
            # We treat these as "virtual" tool calls for the script logic,
            # but we DO NOT add "tool_calls" to the message_obj because
            # the API expects "tool_calls" to map to "role: tool".
            # For Hermes, the tool call is IN the content.

    return message_obj, detected_protocol, parsed_hermes_tools


# ------------------------------------------------------------------
# 3. Main Logic
# ------------------------------------------------------------------

if __name__ == "__main__":
    conversation_history = [{"role": "user", "content": TEST_PROMPT}]

    # --- TURN 1 ---
    assistant_msg, protocol, hermes_tools = run_turn(conversation_history)
    conversation_history.append(assistant_msg)

    print(f"\n📊 [INTERMEDIATE REPORT] Protocol Detected: {protocol.upper()}")

    tools_to_execute = []

    # A. Handle Native
    if protocol == "native":
        print(
            f"[SYSTEM] Native Tool Calls Detected: {len(assistant_msg['tool_calls'])}"
        )
        for tc in assistant_msg["tool_calls"]:
            tools_to_execute.append(
                {
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "args_str": tc["function"]["arguments"],
                    "type": "native",
                }
            )

    # B. Handle Hermes
    elif protocol == "hermes":
        print(f"[SYSTEM] Hermes/Text Tool Calls Detected: {len(hermes_tools)}")
        for i, ht in enumerate(hermes_tools):
            # Generate a fake ID for tracking
            tools_to_execute.append(
                {
                    "id": f"hermes_call_{i}",
                    "name": ht["name"],
                    "args_str": (
                        json.dumps(ht["arguments"])
                        if isinstance(ht["arguments"], dict)
                        else ht["arguments"]
                    ),
                    "type": "hermes",
                }
            )

    else:
        print("[SYSTEM] No tool calls detected.")
        print(f"[RESPONSE] {assistant_msg.get('content')}")
        sys.exit(0)

    # --- EXECUTION LOOP ---
    if tools_to_execute:
        print(f"[SYSTEM] Executing {len(tools_to_execute)} tool(s)...")

        for t in tools_to_execute:
            # Parse Args
            try:
                func_args = json.loads(t["args_str"])
            except Exception:
                print(f"[ERROR] Failed to parse args: {t['args_str']}")
                func_args = {}

            # Execute
            result_str = execute_local_tool(t["name"], func_args)

            # --- CRITICAL: INJECT RESPONSE BASED ON PROTOCOL ---
            if t["type"] == "native":
                # Standard OpenAI Format
                conversation_history.append(
                    {
                        "role": "tool",
                        "tool_call_id": t["id"],
                        "name": t["name"],
                        "content": result_str,
                    }
                )
                print("   -> Appended [Native] 'tool' role response.")

            elif t["type"] == "hermes":
                # Hermes Style: <tool_response> JSON </tool_response>
                # Usually sent as "role": "tool" (newer) or "role": "user" (classic Hermes)
                # We will use "role": "tool" with the XML wrapper as it is the most robust hybrid approach.
                hermes_response_content = (
                    f"<tool_response>\n{result_str}\n</tool_response>"
                )
                conversation_history.append(
                    {
                        "role": "tool",
                        "name": t["name"],
                        "content": hermes_response_content,
                    }
                )
                print("   -> Appended [Hermes] XML response wrapper.")

        # --- TURN 2 (The Reaction) ---
        print("\n[SYSTEM] Sending tool outputs back to model...")
        final_msg, final_proto, _ = run_turn(conversation_history)

        print("\n===================================================================")
        print("🎉 FINAL ANALYSIS REPORT")
        print("===================================================================")
        print(f"1. Model: {MODEL}")
        print(f"2. Preferred Protocol: {protocol.upper()}")
        print(f"3. Tool Execution Successful: {'Yes' if tools_to_execute else 'No'}")
        print(f"4. Final Answer:\n{final_msg.get('content')}")
        print("===================================================================")
