"""
Live MCP Execution Integration Test
-----------------------------------
Purpose:
1. Use an existing assistant with real DeepWiki MCP tools attached.
2. Run inference against a TogetherAI-backed Project David model.
3. Force the model to use DeepWiki.
4. Do NOT execute any tool locally.
5. Verify the MCP tool is owned by Project David Core.
6. Stream and display the final model response.
"""

import json
import os
import time

from dotenv import load_dotenv

from projectdavid import (
    ContentEvent,
    DecisionEvent,
    Entity,
    ReasoningEvent,
    ToolCallRequestEvent,
)


# ------------------------------------------------------------------
# 0. CONFIGURATION
# ------------------------------------------------------------------

load_dotenv(".tests.env")

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
GREY = "\033[90m"
MAGENTA = "\033[95m"
RESET = "\033[0m"


BASE_URL = (
    os.getenv("PROJECT_DAVID_PLATFORM_BASE_URL")
    or os.getenv("BASE_URL")
    or "http://localhost:80"
)

API_KEY = os.getenv("DEV_PROJECT_DAVID_CORE_TEST_USER_KEY")

ASSISTANT_ID = os.getenv("ASSISTANT_ID")

# Override this in .tests.env if required:
#
# MCP_TEST_MODEL_ID=together-ai/...
#
MODEL_ID = (
    os.getenv("MCP_TEST_MODEL_ID")
    or "together-ai/mistralai/Ministral-3-14B-Instruct-2512"
)


TEST_PROMPT = """
You must use the attached DeepWiki MCP tool to answer this request.

Use the tool that retrieves the documentation structure for a repository.

Repository:
fastapi/fastapi

Do not answer from memory.

After retrieving the structure from DeepWiki, summarize the major
documentation sections that DeepWiki returned.
""".strip()


if not API_KEY:
    raise RuntimeError(
        "Missing DEV_PROJECT_DAVID_CORE_TEST_USER_KEY / ENTITIES_API_KEY"
    )

if not ASSISTANT_ID:
    raise RuntimeError("Missing ASSISTANT_ID")


print(f"{GREY}[CONFIG] Base URL:     {BASE_URL}{RESET}")
print(f"{GREY}[CONFIG] Assistant ID: {ASSISTANT_ID}{RESET}")
print(f"{GREY}[CONFIG] Model ID:      {MODEL_ID}{RESET}")


# ------------------------------------------------------------------
# 1. SDK INIT
# ------------------------------------------------------------------

client = Entity(
    base_url=BASE_URL,
    api_key=API_KEY,
)


# ------------------------------------------------------------------
# 2. PREFLIGHT: VERIFY ASSISTANT + MCP TOOL ATTACHMENT
# ------------------------------------------------------------------

print(f"\n{CYAN}=== ASSISTANT PREFLIGHT ==={RESET}")

assistant = client.assistants.retrieve_assistant(ASSISTANT_ID)

print(f"assistant.id={assistant.id}")
print(f"assistant.model={assistant.model}")
print(f"assistant.max_turns={assistant.max_turns}")

tool_names = []

for tool in assistant.tools or []:
    function = tool.get("function") or {}
    name = function.get("name")

    if name:
        tool_names.append(name)
        print(f"tool={name}")


expected_tool = "deepwiki__read_wiki_structure"

if expected_tool not in tool_names:
    raise RuntimeError(
        f"Expected MCP tool {expected_tool!r} is not attached "
        f"to assistant {ASSISTANT_ID}"
    )

print(f"{GREEN}MCP_ATTACHMENT_PREFLIGHT=PASS{RESET}")

if getattr(assistant, "max_turns", None) == 1:
    print(
        f"{YELLOW}[NOTE] Assistant currently has max_turns=1. "
        "If the post-tool model turn is blocked, inspect this first."
        f"{RESET}"
    )


# ------------------------------------------------------------------
# 3. CREATE THREAD / MESSAGE / RUN
# ------------------------------------------------------------------

print(f"\n{CYAN}=== CREATE INFERENCE STATE ==={RESET}")

global_start = time.perf_counter()

thread = client.threads.create_thread()

print(f"thread.id={thread.id}")


message = client.messages.create_message(
    thread_id=thread.id,
    role="user",
    content=TEST_PROMPT,
    assistant_id=ASSISTANT_ID,
)

print(f"message.id={message.id}")


run = client.runs.create_run(
    assistant_id=ASSISTANT_ID,
    thread_id=thread.id,
)

print(f"run.id={run.id}")


# ------------------------------------------------------------------
# 4. SETUP UNIFIED STREAM
# ------------------------------------------------------------------

stream = client.synchronous_inference_stream

stream.setup(
    thread_id=thread.id,
    assistant_id=ASSISTANT_ID,
    message_id=message.id,
    run_id=run.id,
    api_key=os.getenv("TOGETHER_API_KEY"),
)


# ------------------------------------------------------------------
# 5. STREAM
# ------------------------------------------------------------------

print(f"\n{CYAN}=== LIVE MCP INFERENCE STREAM ==={RESET}")

print(f"{'LATENCY':<14} | " f"{'EVENT CLASS':<28} | " f"PAYLOAD")

print("-" * 120)


last_tick = time.perf_counter()

content_chunks = []

saw_content = False
saw_reasoning = False
saw_decision = False
saw_client_tool_request = False
saw_deepwiki_reference = False

stream_completed = False


try:

    for event in stream.stream_events(model=MODEL_ID):

        current_tick = time.perf_counter()
        delta = current_tick - last_tick
        last_tick = current_tick

        time_str = f"[{delta:+.4f}s]"

        class_name = event.__class__.__name__

        payload = event.to_dict()

        payload_json = json.dumps(
            payload,
            ensure_ascii=False,
            default=str,
        )

        color = RESET

        if isinstance(event, ContentEvent):
            color = GREEN
            saw_content = True

        elif isinstance(event, ReasoningEvent):
            color = CYAN
            saw_reasoning = True

        elif isinstance(event, DecisionEvent):
            color = MAGENTA
            saw_decision = True

        elif isinstance(event, ToolCallRequestEvent):
            color = YELLOW
            saw_client_tool_request = True

        if "deepwiki__" in payload_json:
            saw_deepwiki_reference = True

        print(
            f"{GREY}{time_str:<14}{RESET} | "
            f"{color}{class_name:<28}{RESET} | "
            f"{payload_json}"
        )

        # ----------------------------------------------------------
        # IMPORTANT MCP BOUNDARY TEST
        # ----------------------------------------------------------
        #
        # We deliberately DO NOT call:
        #
        #     event.execute(...)
        #
        # There is no Python MCP handler in this process.
        #
        # DeepWiki execution belongs to Project David Core.
        #
        # If a ToolCallRequestEvent reaches us asking the caller
        # to execute deepwiki__*, that is extremely useful evidence:
        # the MCP call has leaked through the local-tool boundary.
        # ----------------------------------------------------------

        if isinstance(event, ToolCallRequestEvent):

            tool_name = getattr(
                event,
                "tool_name",
                None,
            )

            print(f"\n{YELLOW}" f"[OBSERVED TOOL REQUEST] {tool_name}" f"{RESET}")

            if tool_name and tool_name.startswith("deepwiki__"):
                print(
                    f"{RED}"
                    "CLIENT_SIDE_MCP_REQUEST=OBSERVED\n"
                    "No local handler will be executed."
                    f"{RESET}"
                )

        # Capture content without assuming a particular ContentEvent
        # attribute shape. The complete payload remains printed above.

        if isinstance(event, ContentEvent):

            candidate = (
                payload.get("content") or payload.get("delta") or payload.get("text")
            )

            if isinstance(candidate, str):
                content_chunks.append(candidate)

    stream_completed = True


except Exception as exc:

    print(f"\n{RED}" f"[STREAM ERROR] " f"{exc.__class__.__name__}: {exc}" f"{RESET}")


# ------------------------------------------------------------------
# 6. RESULTS
# ------------------------------------------------------------------

global_end = time.perf_counter()

total_time = global_end - global_start


print(f"\n{YELLOW}{'=' * 72}{RESET}")
print(f"{YELLOW} LIVE MCP EXECUTION TEST RESULTS{RESET}")
print(f"{YELLOW}{'=' * 72}{RESET}")

print(f"THREAD_ID={thread.id}")
print(f"MESSAGE_ID={message.id}")
print(f"RUN_ID={run.id}")
print(f"MODEL_ID={MODEL_ID}")

print("MCP_ATTACHMENT_PREFLIGHT=" + ("PASS" if expected_tool in tool_names else "FAIL"))

print("DEEPWIKI_EVENT_REFERENCE=" + ("YES" if saw_deepwiki_reference else "NO"))

print("CLIENT_TOOL_REQUEST_SURFACED=" + ("YES" if saw_client_tool_request else "NO"))

print("CONTENT_RECEIVED=" + ("YES" if saw_content else "NO"))

print("STREAM_COMPLETED=" + ("YES" if stream_completed else "NO"))

print(f"TOTAL_ROUND_TRIP_SECONDS=" f"{total_time:.4f}")


# ------------------------------------------------------------------
# 7. FINAL CONTENT
# ------------------------------------------------------------------

final_text = "".join(content_chunks).strip()


if final_text:

    print(f"\n{GREEN}=== FINAL MODEL CONTENT ==={RESET}")
    print(final_text)


# ------------------------------------------------------------------
# 8. BASIC ACCEPTANCE
# ------------------------------------------------------------------

print(f"\n{CYAN}=== ACCEPTANCE ==={RESET}")

if stream_completed and saw_content:

    print(f"{GREEN}INFERENCE_STREAM=PASS{RESET}")
    print(f"{GREEN}FINAL_ANSWER=PASS{RESET}")

else:

    print(f"{RED}INFERENCE_STREAM=FAIL{RESET}")


if saw_client_tool_request:

    print(
        f"{YELLOW}"
        "MCP_BOUNDARY=INSPECT\n"
        "A ToolCallRequestEvent reached the SDK. "
        "We intentionally did NOT execute it locally."
        f"{RESET}"
    )

else:

    print(f"{GREEN}" "LOCAL_TOOL_EXECUTOR_USED=NO" f"{RESET}")


print(f"\n{YELLOW}" + "=" * 72 + f"{RESET}")
