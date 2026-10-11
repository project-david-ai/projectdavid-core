"""
Isolated canonical CV file_search proof.

Purpose:
1. Query the existing canonical candidate vector store directly.
2. Ask the existing Gideon supervisor a basic CV question.
3. Observe whether native file_search executes successfully.

NO Gideon runtime.
NO Jobs faction.
NO application campaign.
NO vector-store mutation.
NO assistant mutation.
"""

import json
import os
import time

from dotenv import load_dotenv
from projectdavid import (
    ContentEvent,
    Entity,
    ToolCallRequestEvent,
)


# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

load_dotenv(".tests.env")

BASE_URL = os.getenv("BASE_URL", "http://localhost:80")

ENTITIES_API_KEY = os.getenv("DEV_PROJECT_DAVID_CORE_TEST_USER_KEY")

PROVIDER_API_KEY = os.getenv("TOGETHER_API_KEY") or os.getenv("HYPERBOLIC_API_KEY")

MODEL_ID = (
    os.getenv("MCP_TEST_MODEL_ID")
    or "together-ai/mistralai/Ministral-3-14B-Instruct-2512"
)

PROVIDER = os.getenv("MCP_TEST_PROVIDER") or "Together"

ASSISTANT_ID = "asst_lYd7SCmoFvf0aq5LRSfGhW"
VECTOR_STORE_ID = "vect_x9pQWTKSS6H6JiJSu5mT4I"

QUERY = (
    "What is the candidate's most recent job title? "
    "Use file_search to answer from the candidate CV. "
    "Do not answer from general knowledge or assumptions."
)


if not PROVIDER_API_KEY:
    raise RuntimeError("Missing inference provider API key.")


print("ISOLATED_CV_FILE_SEARCH=START")
print(f"BASE_URL={BASE_URL}")
print(f"ASSISTANT_ID={ASSISTANT_ID}")
print(f"VECTOR_STORE_ID={VECTOR_STORE_ID}")
print(f"MODEL_ID={MODEL_ID}")
print(f"PROVIDER={PROVIDER}")


# ------------------------------------------------------------------
# CLIENT
# ------------------------------------------------------------------

client = Entity(
    base_url=BASE_URL,
    api_key=ENTITIES_API_KEY,
)

if hasattr(client, "synchronous_inference_stream"):
    client.synchronous_inference_stream.bind_clients(
        client.runs,
        client.actions,
        client.messages,
        client.assistants,
    )


# ==================================================================
#  SAME ASSISTANT + NATIVE FILE_SEARCH
# ==================================================================
print("\n" + "=" * 80)
print("TEST 2 — SUPERVISOR NATIVE FILE_SEARCH")
print("=" * 80)

print(f"PROMPT={QUERY}")

thread = client.threads.create_thread()

message = client.messages.create_message(
    thread_id=thread.id,
    role="user",
    content=QUERY,
    assistant_id=ASSISTANT_ID,
)

run = client.runs.create_run(
    assistant_id=ASSISTANT_ID,
    thread_id=thread.id,
)

print(f"THREAD_ID={thread.id}")
print(f"MESSAGE_ID={message.id}")
print(f"RUN_ID={run.id}")


stream = client.synchronous_inference_stream

stream.setup(
    thread_id=thread.id,
    assistant_id=ASSISTANT_ID,
    message_id=message.id,
    run_id=run.id,
    api_key=PROVIDER_API_KEY,
)


tool_calls = []
content_parts = []

start = time.perf_counter()

try:
    for event in stream.stream_events(model=MODEL_ID):
        elapsed = time.perf_counter() - start
        class_name = event.__class__.__name__

        try:
            payload = event.to_dict()
        except Exception:
            payload = {"repr": repr(event)}

        print(
            f"[{elapsed:8.3f}s] "
            f"{class_name:<28} "
            f"{json.dumps(payload, default=str)}"
        )

        if isinstance(event, ToolCallRequestEvent):
            tool_name = getattr(event, "tool_name", None)
            tool_calls.append(tool_name)

        if isinstance(event, ContentEvent):
            content = getattr(event, "content", None)

            if content:
                content_parts.append(str(content))

except Exception as exc:
    print("\nSUPERVISOR_STREAM=FAIL")
    print(f"{type(exc).__name__}: {exc}")

else:
    print("\nSUPERVISOR_STREAM=COMPLETE")


# ==================================================================
# RESULTS
# ==================================================================

print("\n" + "=" * 80)
print("RESULT")
print("=" * 80)

print(f"OBSERVED_CONSUMER_TOOL_CALLS={tool_calls}")

final_text = "".join(content_parts)

print("\nFINAL_TEXT:")
print(final_text or "<no content>")

print("\nISOLATED_CV_FILE_SEARCH=COMPLETE")
