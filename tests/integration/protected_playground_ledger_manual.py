"""
Project David — Protected Playground Execution Ledger Test
----------------------------------------------------------
1. Create a Thread.
2. Create a user Message.
3. Create a Run using the existing protected Playground Assistant.
4. Execute ordinary inference using the protected Core API key.
5. Inspect the Run in the Portal.

No external tools. No delegation. No deep research.
One persistent DEV Thread/Message/Run per execution; no retries.
"""

import os
import time

from dotenv import load_dotenv
from projectdavid import ContentEvent, Entity, ToolCallRequestEvent

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent / ".tests.env"

load_dotenv(ENV_FILE, override=True)

if not ENV_FILE.is_file():
    raise FileNotFoundError(f"Missing environment file: {ENV_FILE}")


# This must be the *protected* Core key belonging to the same owner
# as the Playground Assistant, not the ordinary Core test-user key.
CORE_API_KEY = os.getenv("DEV_PROJECT_DAVID_CORE_PROTECTED_KEY")

# Existing protected assistant created by the Playground bootstrap test.
ASSISTANT_ID = os.getenv(
    "PLAYGROUND_ASSISTANT_ID",
    "asst_playground_ecf4e9e7b4e295e9b301aa8379910d8280515fba0e604915",
)

# Match the model used when this protected assistant was bootstrapped.
MODEL_ID = "hyperbolic/deepseek-v4-flash-0731"
PROVIDER_API_KEY = os.getenv("HYPERBOLIC_API_KEY")

PROMPT = (
    "Reply with exactly this sentence: "
    "Project David execution accounting test successful."
)

if not CORE_API_KEY:
    raise RuntimeError(
        "Missing DEV_PROJECT_DAVID_CORE_PROTECTED_KEY in .tests.env; "
        "do not use DEV_PROJECT_DAVID_CORE_TEST_USER_KEY"
    )
if not ASSISTANT_ID.startswith("asst_playground_"):
    raise RuntimeError("PLAYGROUND_ASSISTANT_ID is not a protected Playground ID")
if not PROVIDER_API_KEY:
    raise RuntimeError("Missing HYPERBOLIC_API_KEY in .tests.env")

# ------------------------------------------------------------
# SDK Client: protected Core identity, normal inference SDK
# ------------------------------------------------------------

client = Entity(api_key=CORE_API_KEY)

print("\nPROJECT DAVID — PROTECTED PLAYGROUND LEDGER TEST")
print("-" * 55)
print(f"Assistant: {ASSISTANT_ID}")
print(f"Model:     {MODEL_ID}")

start = time.perf_counter()
run = None
output = []

try:
    # 1. Create Thread
    thread = client.threads.create_thread()
    print(f"\nThread:  {thread.id}")

    # 2. Create Message
    message = client.messages.create_message(
        thread_id=thread.id,
        role="user",
        content=PROMPT,
        assistant_id=ASSISTANT_ID,
    )
    print(f"Message: {message.id}")

    # 3. Create Run
    run = client.runs.create_run(
        assistant_id=ASSISTANT_ID,
        thread_id=thread.id,
        model=MODEL_ID,
    )

    print(f"Run:     {run.id}")

    # 4. Execute ordinary SDK inference; Core auth and provider auth are distinct.
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

    for event in stream.stream_events(model=MODEL_ID):
        if isinstance(event, ToolCallRequestEvent):
            raise RuntimeError(f"Unexpected tool invocation: {event.tool_name}")
        if isinstance(event, ContentEvent) and event.content:
            output.append(event.content)
            print(event.content, end="", flush=True)

    if not "".join(output).strip():
        raise RuntimeError("Inference stream finished without response content")

    print("\n\nINFERENCE STREAM FINISHED")
    print(f"Response characters: {len(''.join(output))}")

finally:
    elapsed = time.perf_counter() - start
    print("\n" + "-" * 55)
    print(f"Elapsed: {elapsed:.2f}s")
    if run is not None:
        print(f"Run ID:  {run.id}")
        print("Inspect in Portal: Agents → Run activity → Execution accounting")
        print(f"RUN_ID={run.id}")
