"""
Project David — Simple Execution Ledger Test
--------------------------------------------
1. Create a Thread.
2. Create a user Message.
3. Create a Run.
4. Execute ordinary inference.
5. Inspect the Run in the Portal.

No external tools.
No delegation.
No deep research.
"""

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from projectdavid import ContentEvent, Entity, ToolCallRequestEvent

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

if not PROVIDER_API_KEY:
    raise RuntimeError("Missing HYPERBOLIC_API_KEY")

# ------------------------------------------------------------
# SDK Client
# ------------------------------------------------------------

client = Entity(api_key=CORE_API_KEY)

print("\nPROJECT DAVID — EXECUTION LEDGER TEST")
print("-" * 55)
print(f"Assistant: {ASSISTANT_ID}")
print(f"Model:     {MODEL_ID}")

start = time.perf_counter()

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

# Model selection belongs to the inference request, not Run creation.
run = client.runs.create_run(
    assistant_id=ASSISTANT_ID,
    thread_id=thread.id,
)

print(f"Run:     {run.id}")

# ------------------------------------------------------------
# 4. Execute Inference
# ------------------------------------------------------------

stream = client.synchronous_inference_stream

stream.setup(
    thread_id=thread.id,
    assistant_id=ASSISTANT_ID,
    message_id=message.id,
    run_id=run.id,
    api_key=PROVIDER_API_KEY,
)

print("\nMODEL RESPONSE")
print("-" * 55)

output = []

try:
    for event in stream.stream_events(model=MODEL_ID):
        if isinstance(event, ToolCallRequestEvent):
            raise RuntimeError(f"Unexpected tool invocation: {event.tool_name}")

        if isinstance(event, ContentEvent) and event.content:
            output.append(event.content)
            print(event.content, end="", flush=True)
finally:
    elapsed = time.perf_counter() - start
    print("\n\n" + "-" * 55)
    print(f"Run ID:  {run.id}")
    print(f"Elapsed: {elapsed:.2f}s")

print("\nINFERENCE STREAM FINISHED")
print(f"Response characters: {len(''.join(output))}")

print("\nInspect this Run in the Project David Portal:")
print("Agents → Run activity → Execution accounting")
print(f"\nRUN_ID={run.id}")
