"""
Project David — Simple Execution Ledger BYOK Test
-------------------------------------------------
1. Register/reuse a Hyperbolic provider key.
2. Create a Thread.
3. Create a user Message.
4. Create a Run.
5. Execute ordinary streaming inference using provider_key_id.
6. Inspect the Run in the Portal.

No external tools. No delegation. No deep research.
"""

import json
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from projectdavid import Entity

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

ENV_FILE = Path(__file__).resolve().parent / ".tests.env"
load_dotenv(ENV_FILE, override=True)

CORE_API_KEY = os.getenv("DEV_PROJECT_DAVID_CORE_TEST_USER_KEY")
ASSISTANT_ID = os.getenv("ASSISTANT_ID")
MODEL_ID = os.getenv("MCP_TEST_MODEL_ID")
PROVIDER_API_KEY = os.getenv("HYPERBOLIC_API_KEY")

PROMPT = (
    "Reply with exactly this sentence: "
    "Project David execution accounting test successful."
)

if not CORE_API_KEY:
    raise RuntimeError("Missing DEV_PROJECT_DAVID_CORE_TEST_USER_KEY")
if not ASSISTANT_ID:
    raise RuntimeError("Missing ASSISTANT_ID")
if not MODEL_ID:
    raise RuntimeError("Missing MCP_TEST_MODEL_ID")
if not MODEL_ID.strip().lower().startswith("hyperbolic/"):
    raise RuntimeError("BYOK test requires MCP_TEST_MODEL_ID=hyperbolic/<model>")
if not PROVIDER_API_KEY:
    raise RuntimeError("Missing HYPERBOLIC_API_KEY")

# ------------------------------------------------------------
# SDK Client
# ------------------------------------------------------------

client = Entity(api_key=CORE_API_KEY)

print("\nPROJECT DAVID — EXECUTION LEDGER BYOK TEST")
print("-" * 55)
print(f"Assistant: {ASSISTANT_ID}")
print(f"Model:     {MODEL_ID}")

start = time.perf_counter()

# ------------------------------------------------------------
# 0. Register/reuse the user's Hyperbolic credential
# ------------------------------------------------------------

provider_key = client.provider_keys.create(
    provider="hyperbolic",
    api_key=PROVIDER_API_KEY,
    name="Reference ledger test",
)
print(f"Provider key ID: {provider_key.id}")

# ------------------------------------------------------------
# 1. Create Thread
# ------------------------------------------------------------

thread = client.threads.create_thread()
print(f"\nThread: {thread.id}")

# ------------------------------------------------------------
# 2. Create Message
# ------------------------------------------------------------

message = client.messages.create_message(
    thread_id=thread.id,
    role="user",
    content=PROMPT,
    assistant_id=ASSISTANT_ID,
)
print(f"Message: {message.id}")

# ------------------------------------------------------------
# 3. Create Run
# ------------------------------------------------------------

run = client.runs.create_run(
    assistant_id=ASSISTANT_ID,
    thread_id=thread.id,
)
print(f"Run:     {run.id}")

# ------------------------------------------------------------
# 4. Execute inference
# ------------------------------------------------------------
# Current SDK stream.setup() cannot pass provider_key_id yet. Call the
# same Core completion endpoint directly for this verification only.
# The raw Hyperbolic key never goes into the inference request.

payload = {
    "model": MODEL_ID,
    "thread_id": thread.id,
    "assistant_id": ASSISTANT_ID,
    "message_id": message.id,
    "run_id": run.id,
    "provider_key_id": provider_key.id,
    "stream": True,
}

print("\nMODEL RESPONSE")
print("-" * 55)

output = []
stream_done = False

try:
    with httpx.Client(
        base_url=client.base_url,
        headers={"X-API-Key": CORE_API_KEY},
        timeout=httpx.Timeout(600.0, connect=10.0),
    ) as http:
        with http.stream("POST", "/v1/completions", json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    stream_done = True
                    break
                if not data:
                    continue
                event = json.loads(data)
                kind = event.get("type")
                if kind == "tool_call_manifest":
                    raise RuntimeError(
                        f"Unexpected tool invocation: {event.get('tool_name', '(unknown)')}"
                    )
                if kind == "error":
                    raise RuntimeError(
                        f"Inference failed: {event.get('message', 'unknown error')}"
                    )
                if kind == "content" and event.get("content"):
                    output.append(event["content"])
                    print(event["content"], end="", flush=True)
finally:
    elapsed = time.perf_counter() - start
    print("\n\n" + "-" * 55)
    print(f"Run ID:  {run.id}")
    print(f"Elapsed: {elapsed:.2f}s")

if not stream_done:
    raise RuntimeError("Inference stream ended without [DONE]")
if not "".join(output).strip():
    raise RuntimeError("Inference completed without response content")

print("\nINFERENCE STREAM FINISHED")
print(f"Response characters: {len(''.join(output))}")

print("\nInspect this Run in the Project David Portal:")
print("Agents → Run activity → Execution accounting")
print(f"\nRUN_ID={run.id}")
