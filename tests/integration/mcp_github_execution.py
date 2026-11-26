"""Live authenticated GitHub MCP test using registered provider BYOK.

Register the GitHub MCP server separately (one-time bearer credential upload).
This test reuses that registration, discovers and attaches read-only tools,
then streams inference using an existing provider key ID. Neither provider
secrets nor the GitHub PAT are sent by this test.
"""

import inspect
import json
import os
import time
from contextlib import closing
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

from dotenv import load_dotenv
from projectdavid import ContentEvent, Entity, ToolCallRequestEvent

GITHUB_URL = "https://api.githubcopilot.com/mcp/"
REMOTE_TOOLS = ("get_file_contents", "search_repositories")


def main():
    print("SCRIPT_VERSION=GITHUB_MCP_REGISTERED_BYOK_V2", flush=True)
    load_dotenv(Path(__file__).resolve().with_name(".tests.env"))
    base_url = (
        os.getenv("PROJECT_DAVID_PLATFORM_BASE_URL")
        or os.getenv("BASE_URL")
        or "http://localhost:80"
    )
    api_key = os.getenv("DEV_PROJECT_DAVID_CORE_TEST_USER_KEY")
    assistant_id = os.getenv("ASSISTANT_ID", "").strip()
    provider_key_id = os.getenv("PD_PROVIDER_KEY_ID", "").strip()
    model_id = os.getenv("MCP_TEST_MODEL_ID", "").strip()
    server_id = os.getenv("GITHUB_MCP_SERVER_ID", "").strip()

    if urlsplit(base_url).hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise RuntimeError("This test must target local Core")
    if not api_key or not assistant_id:
        raise RuntimeError("Set DEV_PROJECT_DAVID_CORE_TEST_USER_KEY and ASSISTANT_ID")
    if assistant_id.startswith("asst_playground_"):
        raise RuntimeError(
            "Use an ordinary test assistant, not the protected Playground"
        )
    if not provider_key_id.startswith("pkey_"):
        raise RuntimeError("Set PD_PROVIDER_KEY_ID to an existing provider key ID")
    if not model_id or "/" not in model_id:
        raise RuntimeError("Set MCP_TEST_MODEL_ID to the configured provider/model ID")

    print(f"[CONFIG] Base URL: {base_url}")
    print(f"[CONFIG] Assistant ID: {assistant_id}")
    print(f"[CONFIG] Model ID: {model_id}")

    client = Entity(base_url=base_url, api_key=api_key)

    # Same registered BYOK adapter as the successful DeepWiki test. The
    # synchronous wrapper retains its native event mapping and streaming.
    inference = client.inference.stream_inference_response
    if "provider_key_id" not in inspect.signature(inference).parameters:
        raise RuntimeError(
            "Installed SDK inference client lacks registered BYOK support"
        )
    registered_inference = SimpleNamespace(
        stream_inference_response=partial(inference, provider_key_id=provider_key_id),
    )
    print("REGISTERED_PROVIDER_KEY=YES")

    assistant = client.assistants.retrieve_assistant(assistant_id)
    if getattr(assistant, "deep_research", False) or getattr(
        assistant, "engineer", False
    ):
        raise RuntimeError(
            "Disable Deep Research / Engineer mode on the test assistant first"
        )
    if getattr(assistant, "max_turns", None) == 1:
        raise RuntimeError("Test assistant needs at least two inference turns")

    print("\n=== AUTHENTICATED MCP REGISTRATION ===")
    matches = [
        server
        for server in client.mcp.list_servers()
        if str(server.url) == GITHUB_URL
        and server.auth_type == "bearer"
        and (not server_id or server.id == server_id)
    ]
    if not matches:
        raise RuntimeError(
            "No matching authenticated GitHub MCP registration. Run the separate "
            "registration example once, then retry. Check GITHUB_MCP_SERVER_ID if set."
        )
    if len(matches) != 1:
        raise RuntimeError("Multiple GitHub registrations; set GITHUB_MCP_SERVER_ID")
    server = matches[0]
    if not server.enabled:
        raise RuntimeError("Selected GitHub MCP registration is disabled")
    if {"token", "credential_id", "encrypted_payload"} & server.model_dump().keys():
        raise RuntimeError("MCP registration response exposes credential fields")
    print("MCP_REGISTRATION=REUSED")
    print(f"MCP_SERVER_ID={server.id}")
    print(f"MCP_AUTH_TYPE={server.auth_type}")

    # Core retrieves the bearer credential. No token argument is accepted or
    # provided for discovery or attachment.
    discovered = client.mcp.discover_tools(server)
    selected = discovered.select(*REMOTE_TOOLS)
    print(f"MCP_DISCOVERY=PASS (tools={len(discovered)})")
    print("TOKEN_RESUPPLIED_FOR_DISCOVERY=NO")

    def attachments():
        return {
            tool.remote_name: tool
            for tool in client.mcp.list_assistant_tools(assistant_id)
            if tool.registration_id == server.id and tool.remote_name in REMOTE_TOOLS
        }

    existing = attachments()
    if any(not tool.enabled for tool in existing.values()):
        raise RuntimeError("A selected GitHub MCP attachment is disabled")
    missing = [tool for tool in selected if tool.remote_name not in existing]
    if missing:
        client.mcp.attach_tools(assistant_id=assistant_id, tools=missing)
        print("MCP_ATTACHMENT=CREATED")
    else:
        print("MCP_ATTACHMENT=REUSED")
    print("TOKEN_RESUPPLIED_FOR_ATTACHMENT=NO")

    attached = attachments()
    if set(attached) != set(REMOTE_TOOLS) or not all(
        t.enabled for t in attached.values()
    ):
        raise RuntimeError("Expected both enabled GitHub MCP attachments")
    expected_tool = attached["search_repositories"].provider_name
    assistant = client.assistants.retrieve_assistant(assistant_id)
    names = {(tool.get("function") or {}).get("name") for tool in assistant.tools or []}
    if not all(tool.provider_name in names for tool in attached.values()):
        raise RuntimeError(
            "Persisted MCP attachment missing from assistant definitions"
        )
    print(f"GITHUB_TOOL={expected_tool}")
    print("GITHUB_MCP_ATTACHMENT_PREFLIGHT=PASS")

    prompt = (
        f"Use the attached GitHub MCP tool named {expected_tool} to search GitHub "
        "repositories for fastapi. Do not answer from memory or use other tools. "
        "After the result returns, summarize the first few repositories returned, "
        "including repository names and short descriptions where available. "
        "If the tool fails, report the failure instead of inventing results."
    )
    started = time.perf_counter()
    thread = client.threads.create_thread()
    message = client.messages.create_message(
        thread_id=thread.id,
        role="user",
        content=prompt,
        assistant_id=assistant_id,
    )
    run = client.runs.create_run(
        assistant_id=assistant_id,
        thread_id=thread.id,
        model=model_id,
    )
    print(f"THREAD_ID={thread.id}")
    print(f"MESSAGE_ID={message.id}")
    print(f"RUN_ID={run.id}")
    if run.model != model_id:
        raise RuntimeError(f"Unexpected persisted run model: {run.model!r}")

    stream = client.synchronous_inference_stream
    stream.setup(
        thread_id=thread.id,
        assistant_id=assistant_id,
        message_id=message.id,
        run_id=run.id,
        api_key=None,
    )
    stream.inference_client = registered_inference
    print("\n=== LIVE AUTHENTICATED GITHUB MCP STREAM ===")
    chunks = []
    last_tick = time.perf_counter()
    with closing(stream.stream_events(model=run.model)) as events:
        for event in events:
            now = time.perf_counter()
            payload = event.to_dict()
            print(
                f"[+{now - last_tick:.4f}s] {type(event).__name__} | "
                f"{json.dumps(payload, ensure_ascii=False, default=str)}"
            )
            last_tick = now
            if isinstance(event, ToolCallRequestEvent):
                print("CLIENT_TOOL_REQUEST_SURFACED=YES")
                raise RuntimeError(
                    "Unexpected client-side tool request; no local execution"
                )
            if payload.get("type") == "error":
                raise RuntimeError("Inference stream reported an error")
            if isinstance(event, ContentEvent):
                text = (
                    payload.get("content")
                    or payload.get("delta")
                    or payload.get("text")
                )
                if isinstance(text, str):
                    chunks.append(text)

    final_text = "".join(chunks).strip()
    if not final_text:
        raise RuntimeError("No final model content received")
    print("\n=== FINAL MODEL CONTENT ===")
    print(final_text)
    print("\n=== ACCEPTANCE ===")
    print(f"RUN_ID={run.id}")
    print(f"MODEL_ID={run.model}")
    print("INFERENCE_STREAM=PASS")
    print("CONTENT_RECEIVED=YES")
    print("CLIENT_TOOL_REQUEST_SURFACED=NO")
    print("LOCAL_TOOL_EXECUTOR_USED=NO")
    print("TOKEN_RESUPPLIED_FOR_EXECUTION=NO")
    print(f"TOTAL_ROUND_TRIP_SECONDS={time.perf_counter() - started:.4f}")
    # Content alone (even a claimed tool result) is not proof of MCP execution.
    print("AUTHENTICATED_GITHUB_MCP_EXECUTION=VERIFY_SERVER_LEDGER")


if __name__ == "__main__":
    main()
