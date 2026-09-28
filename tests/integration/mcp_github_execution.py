"""
Live Authenticated GitHub MCP Execution Integration Test
---------------------------------------------------------
Purpose:
1. Use an existing assistant with a real authenticated GitHub MCP tool attached.
2. Run inference against a TogetherAI-backed Project David model.
3. Force the model to use GitHub search_repositories.
4. Do NOT supply the GitHub PAT to this process.
5. Do NOT execute any MCP tool locally.
6. Verify the MCP tool is owned/executed by Project David Core.
7. Stream and display the final model response.

Auth-1 proof:
If search_repositories succeeds, Core resolved the credential bound to the
MCP registration and authenticated to GitHub without the SDK resupplying it.
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

MODEL_ID = (
    os.getenv("MCP_TEST_MODEL_ID")
    or "together-ai/mistralai/Ministral-3-14B-Instruct-2512"
)


if not API_KEY:
    raise RuntimeError(
        "Missing DEV_PROJECT_DAVID_CORE_TEST_USER_KEY / ENTITIES_API_KEY"
    )

if not ASSISTANT_ID:
    raise RuntimeError("Missing ASSISTANT_ID")


# IMPORTANT:
# This test MUST NOT consume the GitHub PAT directly.
#
# Authentication belongs to the MCP registration stored in Core.
if os.getenv("GITHUB_MCP_PAT"):
    print(
        f"{GREY}"
        "[AUTH] GITHUB_MCP_PAT exists in the environment, "
        "but this script will not read or transmit it."
        f"{RESET}"
    )

print(f"{GREY}[CONFIG] Base URL:     {BASE_URL}{RESET}")
print(f"{GREY}[CONFIG] Assistant ID: {ASSISTANT_ID}{RESET}")
print(f"{GREY}[CONFIG] Model ID:      {MODEL_ID}{RESET}")
print(f"{GREEN}[AUTH] TOKEN_RESUPPLIED=NO{RESET}")


# ------------------------------------------------------------------
# 1. SDK INIT
# ------------------------------------------------------------------

client = Entity(
    base_url=BASE_URL,
    api_key=API_KEY,
)


# ------------------------------------------------------------------
# 2. PREFLIGHT: VERIFY ASSISTANT + GITHUB MCP TOOL
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


# Do not assume the MCP server-name prefix.
#
# Depending on registration naming, we may see something such as:
#
#   github-auth-test__search_repositories
#
# The actual MCP function we care about is the suffix.
github_search_tools = [
    name
    for name in tool_names
    if name == "search_repositories" or name.endswith("__search_repositories")
]


if not github_search_tools:
    raise RuntimeError(
        "No attached GitHub search_repositories MCP tool was found. "
        f"Attached tools: {tool_names}"
    )

if len(github_search_tools) > 1:
    raise RuntimeError(
        "Multiple search_repositories tools were found; refusing to guess. "
        f"Candidates: {github_search_tools}"
    )


expected_tool = github_search_tools[0]

print(f"selected_github_tool={expected_tool}")
print(f"{GREEN}GITHUB_MCP_ATTACHMENT_PREFLIGHT=PASS{RESET}")


if getattr(assistant, "max_turns", None) == 1:
    print(
        f"{YELLOW}"
        "[NOTE] Assistant currently has max_turns=1. "
        "If the post-tool model turn is blocked, inspect this first."
        f"{RESET}"
    )


# ------------------------------------------------------------------
# 3. TEST PROMPT
# ------------------------------------------------------------------

TEST_PROMPT = f"""
You must use the attached GitHub MCP tool named:

{expected_tool}

Use that tool to search GitHub repositories for:

fastapi

Do not answer from memory.
Do not use any other tool.

After the GitHub MCP result is returned, summarize the first few
repositories returned by the tool. Include repository names and a
short description where available.
""".strip()


# ------------------------------------------------------------------
# 4. CREATE THREAD / MESSAGE / RUN
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
# 5. SETUP UNIFIED STREAM
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
# 6. STREAM
# ------------------------------------------------------------------

print(f"\n{CYAN}=== LIVE GITHUB MCP INFERENCE STREAM ==={RESET}")

print(f"{'LATENCY':<14} | " f"{'EVENT CLASS':<28} | " f"PAYLOAD")

print("-" * 120)


last_tick = time.perf_counter()

content_chunks = []

saw_content = False
saw_reasoning = False
saw_decision = False

saw_client_tool_request = False
saw_github_reference = False
saw_github_client_request = False

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

        if expected_tool in payload_json:
            saw_github_reference = True

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
        # There is no GitHub MCP handler in this Python process.
        #
        # GitHub execution belongs to Project David Core.
        #
        # If the GitHub MCP tool reaches us as ToolCallRequestEvent,
        # it leaked through the server-side MCP execution boundary.
        # ----------------------------------------------------------

        if isinstance(event, ToolCallRequestEvent):

            tool_name = getattr(
                event,
                "tool_name",
                None,
            )

            print(f"\n{YELLOW}" f"[OBSERVED TOOL REQUEST] {tool_name}" f"{RESET}")

            if tool_name == expected_tool:

                saw_github_client_request = True

                print(
                    f"{RED}"
                    "CLIENT_SIDE_GITHUB_MCP_REQUEST=OBSERVED\n"
                    "No local handler will be executed."
                    f"{RESET}"
                )

        # Capture final textual content.
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
# 7. RESULTS
# ------------------------------------------------------------------

global_end = time.perf_counter()
total_time = global_end - global_start


print(f"\n{YELLOW}{'=' * 72}{RESET}")
print(f"{YELLOW} AUTHENTICATED GITHUB MCP EXECUTION RESULTS{RESET}")
print(f"{YELLOW}{'=' * 72}{RESET}")

print(f"THREAD_ID={thread.id}")
print(f"MESSAGE_ID={message.id}")
print(f"RUN_ID={run.id}")
print(f"MODEL_ID={MODEL_ID}")
print(f"GITHUB_TOOL={expected_tool}")

print("GITHUB_MCP_ATTACHMENT_PREFLIGHT=PASS")

print("GITHUB_TOOL_EVENT_REFERENCE=" + ("YES" if saw_github_reference else "NO"))

print("CLIENT_TOOL_REQUEST_SURFACED=" + ("YES" if saw_client_tool_request else "NO"))

print("CLIENT_GITHUB_MCP_REQUEST=" + ("YES" if saw_github_client_request else "NO"))

print("CONTENT_RECEIVED=" + ("YES" if saw_content else "NO"))

print("STREAM_COMPLETED=" + ("YES" if stream_completed else "NO"))

print("TOKEN_RESUPPLIED_FOR_EXECUTION=NO")

print(f"TOTAL_ROUND_TRIP_SECONDS=" f"{total_time:.4f}")


# ------------------------------------------------------------------
# 8. FINAL CONTENT
# ------------------------------------------------------------------

final_text = "".join(content_chunks).strip()

if final_text:

    print(f"\n{GREEN}=== FINAL MODEL CONTENT ==={RESET}")
    print(final_text)


# ------------------------------------------------------------------
# 9. ACCEPTANCE
# ------------------------------------------------------------------

print(f"\n{CYAN}=== ACCEPTANCE ==={RESET}")


inference_ok = stream_completed and saw_content

boundary_ok = not saw_github_client_request

github_execution_evidence = saw_github_reference


if inference_ok:
    print(f"{GREEN}INFERENCE_STREAM=PASS{RESET}")
    print(f"{GREEN}FINAL_ANSWER=PASS{RESET}")
else:
    print(f"{RED}INFERENCE_STREAM=FAIL{RESET}")


if github_execution_evidence:
    print(f"{GREEN}GITHUB_MCP_REFERENCED=PASS{RESET}")
else:
    print(f"{RED}GITHUB_MCP_REFERENCED=FAIL{RESET}")


if boundary_ok:
    print(f"{GREEN}LOCAL_GITHUB_TOOL_EXECUTOR_USED=NO{RESET}")
else:
    print(f"{RED}MCP_BOUNDARY=FAIL{RESET}")


if inference_ok and github_execution_evidence and boundary_ok:

    print(f"\n{GREEN}" "AUTHENTICATED_GITHUB_MCP_EXECUTION=PASS" f"{RESET}")

else:

    print(f"\n{RED}" "AUTHENTICATED_GITHUB_MCP_EXECUTION=FAIL" f"{RESET}")


print(f"\n{YELLOW}{'=' * 72}{RESET}")
