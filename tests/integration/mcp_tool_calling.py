"""
Live MCP Execution Integration Test
-----------------------------------
Purpose:
1. Register or reuse DeepWiki, discover and attach its required tool.
2. Run inference against the configured Project David model.
3. Force the model to use DeepWiki.
4. Do NOT execute any tool locally.
5. Verify the MCP tool is owned by Project David Core.
6. Stream and display the final model response.
7. Use an existing registered provider credential (no raw key).
"""

import inspect
import json
import os
import time
from functools import partial
from types import SimpleNamespace
from urllib.parse import urlsplit

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
PROVIDER_KEY_ID = os.getenv("PD_PROVIDER_KEY_ID", "").strip()

# Use the configured provider/model; never silently select another provider.
MODEL_ID = os.getenv("MCP_TEST_MODEL_ID", "").strip()


TEST_PROMPT = """
You must use the attached DeepWiki MCP tool to answer this request.

Use the tool that retrieves the documentation structure for a repository.

Repository:
fastapi/fastapi

Do not answer from memory.

After retrieving the structure from DeepWiki, summarize the major
documentation sections that DeepWiki returned.
""".strip()


# This integration test writes to the selected assistant. Keep it local.
if urlsplit(BASE_URL).hostname not in {"localhost", "127.0.0.1", "::1"}:
    raise RuntimeError("This integration test must target local Core")

if ASSISTANT_ID and ASSISTANT_ID.startswith("asst_playground_"):
    raise RuntimeError("Use an ordinary test assistant, not the protected Playground")

if not API_KEY:
    raise RuntimeError(
        "Missing DEV_PROJECT_DAVID_CORE_TEST_USER_KEY / ENTITIES_API_KEY"
    )

if not ASSISTANT_ID:
    raise RuntimeError("Missing ASSISTANT_ID")

if not PROVIDER_KEY_ID.startswith("pkey_"):
    raise RuntimeError(
        "Set PD_PROVIDER_KEY_ID to an existing registered provider key ID"
    )

if not MODEL_ID or "/" not in MODEL_ID:
    raise RuntimeError("Set MCP_TEST_MODEL_ID to the configured provider/model ID")


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


# The synchronous wrapper may not expose provider_key_id yet. Use its existing
# SDK inference client, binding the registered ID for this stream only.
inference = client.inference.stream_inference_response
if "provider_key_id" not in inspect.signature(inference).parameters:
    raise RuntimeError("Installed SDK inference client lacks registered BYOK support")

registered_inference = SimpleNamespace(
    stream_inference_response=partial(inference, provider_key_id=PROVIDER_KEY_ID),
)
print(f"{GREEN}[AUTH] REGISTERED_PROVIDER_KEY=YES{RESET}")


# ------------------------------------------------------------------
# 2. PREFLIGHT: VERIFY ASSISTANT + MCP TOOL ATTACHMENT
# ------------------------------------------------------------------

print(f"\n{CYAN}=== ASSISTANT PREFLIGHT ==={RESET}")

# Verify assistant access before making any registration changes.
assistant = client.assistants.retrieve_assistant(ASSISTANT_ID)

DEEPWIKI_URL = "https://mcp.deepwiki.com/mcp"
REMOTE_TOOL = "read_wiki_structure"

print(f"\n{CYAN}=== MCP REGISTRATION AND DISCOVERY ==={RESET}")
registrations = client.mcp.list_servers()
matching = [
    server
    for server in registrations
    if str(server.url) == DEEPWIKI_URL and server.auth_type == "none"
]
server_id = os.getenv("DEEPWIKI_MCP_SERVER_ID", "").strip()
if server_id:
    matching = [server for server in matching if server.id == server_id]
    if len(matching) != 1:
        raise RuntimeError(
            "DEEPWIKI_MCP_SERVER_ID must identify a matching unauthenticated registration"
        )

if len(matching) > 1:
    raise RuntimeError(
        "Multiple DeepWiki registrations; set DEEPWIKI_MCP_SERVER_ID to select one"
    )
if matching:
    server = matching[0]
    if not server.enabled:
        raise RuntimeError("Existing DeepWiki registration is disabled")
    print("MCP_REGISTRATION=REUSED")
else:
    server = client.mcp.create_server(name="deepwiki", url=DEEPWIKI_URL)
    print("MCP_REGISTRATION=CREATED")

print(f"MCP_SERVER_ID={server.id}")
# Exercise live discovery even if the tool was already attached.
discovered = client.mcp.discover_tools(server)
selected = discovered.select(REMOTE_TOOL)
print("MCP_DISCOVERY=PASS")


def find_attachment():
    return [
        tool
        for tool in client.mcp.list_assistant_tools(ASSISTANT_ID)
        if tool.registration_id == server.id and tool.remote_name == REMOTE_TOOL
    ]


attachments = find_attachment()
if not attachments:
    client.mcp.attach_tools(assistant_id=ASSISTANT_ID, tools=selected)
    print("MCP_ATTACHMENT=CREATED")
else:
    print("MCP_ATTACHMENT=REUSED")

# Use the persisted alias rather than guessing from the server's name.
attachments = find_attachment()
if len(attachments) != 1 or not attachments[0].enabled:
    raise RuntimeError("Expected one enabled DeepWiki attachment")
expected_tool = attachments[0].provider_name
print(f"MCP_PROVIDER_TOOL={expected_tool}")

assistant = client.assistants.retrieve_assistant(ASSISTANT_ID)
print(f"assistant.id={assistant.id}")
print(f"assistant.model={assistant.model}")
print(f"assistant.max_turns={assistant.max_turns}")

tool_names = []
for tool in assistant.tools or []:
    name = (tool.get("function") or {}).get("name")
    if name:
        tool_names.append(name)
        print(f"tool={name}")

if expected_tool not in tool_names:
    raise RuntimeError(
        "Persisted MCP attachment is missing from assistant tool definitions"
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
    content=f"Use the attached tool named {expected_tool}.\n\n{TEST_PROMPT}",
    assistant_id=ASSISTANT_ID,
)

print(f"message.id={message.id}")


run = client.runs.create_run(
    assistant_id=ASSISTANT_ID,
    thread_id=thread.id,
    model=MODEL_ID,
)

print(f"run.id={run.id}")
if run.model != MODEL_ID:
    raise RuntimeError(f"Unexpected persisted run model: {run.model!r}")


# ------------------------------------------------------------------
# 4. SETUP UNIFIED STREAM
# ------------------------------------------------------------------

stream = client.synchronous_inference_stream

stream.setup(
    thread_id=thread.id,
    assistant_id=ASSISTANT_ID,
    message_id=message.id,
    run_id=run.id,
    api_key=None,
)
stream.inference_client = registered_inference


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

    for event in stream.stream_events(model=run.model):

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

            if tool_name == expected_tool:
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

if stream_completed and saw_content and not saw_client_tool_request:

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


# A successful stream is not by itself proof of remote MCP execution.
print("CORE_MCP_EXECUTION=VERIFY_SERVER_LEDGER")
if not (stream_completed and saw_content and not saw_client_tool_request):
    raise SystemExit(1)
