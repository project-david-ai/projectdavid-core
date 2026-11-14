"""
Deep Research & Hybrid Supervisor Integration Test.

RESEARCH:
    Verify research progress, scratchpad activity and final content.

EXTERNAL:
    Verify that external tool requests reach the SDK client.

Run against a running Project David Core instance.
"""

import json
import os
import time
import traceback
from collections import Counter

from config_orc_fc_deep_research import PROMPTS, config
from dotenv import load_dotenv

from projectdavid import (
    ContentEvent,
    DecisionEvent,
    Entity,
    ReasoningEvent,
    ToolCallRequestEvent,
    WebStatusEvent,
)

from projectdavid.events import (
    ResearchStatusEvent,
    ScratchpadEvent,
)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent / ".tests.env"

load_dotenv(ENV_FILE, override=True)

if not ENV_FILE.is_file():
    raise FileNotFoundError(f"Missing environment file: {ENV_FILE}")

load_dotenv(ENV_FILE)

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
GREY = "\033[90m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"
RESET = "\033[0m"

BASE_URL = os.getenv("BASE_URL", "http://localhost:80")
ASSISTANT_ID = os.getenv("ASSISTANT_ID")

TEST_SCENARIO = os.getenv("PD_TEST_SCENARIO", "RESEARCH").upper()

if TEST_SCENARIO not in ("RESEARCH", "EXTERNAL"):
    raise ValueError(f"Unsupported scenario: {TEST_SCENARIO}")

if not ASSISTANT_ID:
    raise RuntimeError("ASSISTANT_ID is not configured")

USER_API_KEY = os.getenv("DEV_PROJECT_DAVID_CORE_TEST_USER_KEY")

if not USER_API_KEY:
    raise RuntimeError("DEV_PROJECT_DAVID_CORE_TEST_USER_KEY is missing")

CURRENT_PROMPT = (
    config["test_prompt"] if TEST_SCENARIO == "RESEARCH" else PROMPTS["EXTERNAL"]
)

# Optional override. Without one, inherit the Assistant model.
REQUESTED_MODEL = os.getenv("MCP_TEST_MODEL_ID") or None


# ------------------------------------------------------------
# External tool registry
# ------------------------------------------------------------


def get_flight_times(tool_name: str, arguments: dict) -> str:
    print(f"{YELLOW}[CLIENT TOOL] " f"{tool_name}: {arguments}{RESET}")

    return json.dumps(
        {
            "status": "success",
            "route": "LAX-JFK",
            "flights": [
                "DL404 (08:00)",
                "AA101 (09:30)",
            ],
        }
    )


EXTERNAL_TOOL_REGISTRY = {
    "get_flight_times": get_flight_times,
}


# ------------------------------------------------------------
# Main integration test
# ------------------------------------------------------------


def main():
    counts = Counter()
    start = time.perf_counter()
    last_tick = start

    print(f"[CONFIG] Base URL: {BASE_URL}")
    print(f"[CONFIG] Scenario: {TEST_SCENARIO}")

    client = Entity(
        base_url=BASE_URL,
        api_key=USER_API_KEY,
    )

    print(f"\n{YELLOW}[INIT] Creating Run...{RESET}")

    thread = client.threads.create_thread()

    message = client.messages.create_message(
        thread_id=thread.id,
        role="user",
        content=CURRENT_PROMPT,
        assistant_id=ASSISTANT_ID,
    )

    run = client.runs.create_run(
        assistant_id=ASSISTANT_ID,
        thread_id=thread.id,
        model=REQUESTED_MODEL,
    )

    # The persisted Run model is authoritative.
    model = run.model

    if not model:
        raise RuntimeError("Core returned a Run without a resolved model")

    print(f"[RUN] {run.id}")
    print(f"[MODEL] {model}")

    # Select the inference provider key for this model.
    provider_key = None

    if model.startswith("hyperbolic/"):
        provider_key = os.getenv("HYPERBOLIC_API_KEY")

    elif model.startswith("together-ai/"):
        provider_key = os.getenv("TOGETHER_API_KEY")

    stream = client.synchronous_inference_stream

    stream.setup(
        thread_id=thread.id,
        assistant_id=ASSISTANT_ID,
        message_id=message.id,
        run_id=run.id,
        api_key=provider_key,
    )

    print(f"\n{CYAN}[STREAM STARTED]{RESET}")
    print(f"{'LATENCY':<12} | " f"{'EVENT TYPE':<25} | " f"CONTENT / PAYLOAD")
    print("-" * 120)

    def display(delta, label, content, color=GREY):
        elapsed = f"[+{delta:.4f}s]"
        content = str(content).replace("\n", "\\n")

        print(
            f"{GREY}{elapsed:<12}{RESET} | "
            f"{color}{label:<25}{RESET} | "
            f"{content[:350]}"
        )

    try:
        for event in stream.stream_events(model=model):

            now = time.perf_counter()
            delta = now - last_tick
            last_tick = now

            # --------------------------------------------
            # Research progress
            # --------------------------------------------

            if isinstance(event, ResearchStatusEvent):
                counts["research_status"] += 1

                display(
                    delta,
                    "RESEARCH STATUS",
                    (
                        f"{event.activity} "
                        f"[state={event.state}, "
                        f"tool={event.tool or '-'}]"
                    ),
                    BLUE,
                )

            # --------------------------------------------
            # Scratchpad operations
            # --------------------------------------------

            elif isinstance(event, ScratchpadEvent):
                counts["scratchpad"] += 1

                detail = event.activity or event.entry or event.content or ""

                display(
                    delta,
                    "SCRATCHPAD",
                    (f"{event.operation} " f"[state={event.state}] " f"{detail}"),
                    BLUE,
                )

            # --------------------------------------------
            # Web / lifecycle status
            # --------------------------------------------

            elif isinstance(event, WebStatusEvent):
                counts["web_status"] += 1

                display(
                    delta,
                    "WEB STATUS",
                    (f"{event.status}: " f"{event.message or ''}"),
                    BLUE,
                )

                if event.status == "failed":
                    raise RuntimeError(f"Stream reported failure: " f"{event.message}")

            # --------------------------------------------
            # Content
            # --------------------------------------------

            elif isinstance(event, ContentEvent):
                counts["content"] += 1

                if event.content:
                    display(
                        delta,
                        "CONTENT",
                        event.content,
                        GREEN,
                    )

            # --------------------------------------------
            # Reasoning
            # --------------------------------------------

            elif isinstance(event, ReasoningEvent):
                counts["reasoning"] += 1

                display(
                    delta,
                    "REASONING",
                    event.content,
                    CYAN,
                )

            # --------------------------------------------
            # Decision
            # --------------------------------------------

            elif isinstance(event, DecisionEvent):
                counts["decision"] += 1

                display(
                    delta,
                    "DECISION",
                    event.content,
                    MAGENTA,
                )

            # --------------------------------------------
            # External tool calls
            # --------------------------------------------

            elif isinstance(event, ToolCallRequestEvent):
                counts["tool_calls"] += 1

                display(
                    delta,
                    "EXTERNAL TOOL",
                    event.tool_name,
                    YELLOW,
                )

                if event.tool_name == "delegate_research_task":
                    raise RuntimeError(
                        "Internal delegation leaked to the "
                        "SDK client. Server-side supervisor "
                        "execution did not handle the tool."
                    )

                handler = EXTERNAL_TOOL_REGISTRY.get(event.tool_name)

                if handler is None:
                    raise RuntimeError(
                        f"No client handler registered for "
                        f"external tool: {event.tool_name}"
                    )

                success = event.execute(handler)

                if not success:
                    raise RuntimeError(
                        f"External tool execution failed: " f"{event.tool_name}"
                    )

                counts["executed_tools"] += 1

                print(f"{GREEN}[TOOL EXECUTED] " f"{event.tool_name}{RESET}")

            # --------------------------------------------
            # Other supported SDK events
            # --------------------------------------------

            else:
                counts["other"] += 1

                display(
                    delta,
                    type(event).__name__,
                    getattr(
                        event,
                        "activity",
                        getattr(event, "content", ""),
                    ),
                )

        # --------------------------------------------
        # Scenario validation
        # --------------------------------------------

        if TEST_SCENARIO == "RESEARCH":

            if counts["research_status"] == 0:
                raise AssertionError("No ResearchStatusEvent received")

            if counts["scratchpad"] == 0:
                raise AssertionError("No ScratchpadEvent received")

            if counts["content"] == 0:
                raise AssertionError("No final content was streamed")

        elif TEST_SCENARIO == "EXTERNAL":

            if counts["executed_tools"] == 0:
                raise AssertionError("No external tool was executed")

        print(f"\n{GREEN}[PASS] {TEST_SCENARIO}{RESET}")

    except Exception:
        print(f"\n{RED}[FAIL] Integration test failed{RESET}")
        traceback.print_exc()
        raise

    finally:
        duration = time.perf_counter() - start

        print("\n" + "=" * 60)
        print(f"RUN ID: {run.id}")
        print(f"SCENARIO: {TEST_SCENARIO}")
        print(f"TOTAL TIME: {duration:.4f}s")
        print("\nEVENT COUNTS:")

        for name, count in sorted(counts.items()):
            print(f"  {name}: {count}")

        print("=" * 60)


if __name__ == "__main__":
    main()
