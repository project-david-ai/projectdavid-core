"""
Portal Local Function — Native Registered BYOK Integration Test.

Validates:
    1. Retrieval of a Portal-created assistant.
    2. Persistence of its local function definition.
    3. Registered BYOK inference through the native SDK transport.
    4. External function dispatch to the SDK.
    5. Local execution and tool-output submission.
    6. SDK-managed continuation and final response with credential forwarding.
    7. Successful Run completion.

No assistant creation or configuration mutations.
"""

import json
import os
import time

from collections import Counter
from pathlib import Path
from unittest.mock import patch

from dotenv import load_dotenv

from projectdavid import (
    ContentEvent,
    DecisionEvent,
    Entity,
    ReasoningEvent,
    ToolCallRequestEvent,
)


# ============================================================
# CONFIGURATION
# ============================================================

ENV_FILE = Path(__file__).resolve().parent / ".tests.env"

if not ENV_FILE.is_file():
    raise FileNotFoundError(f"Missing environment file: {ENV_FILE}")

load_dotenv(ENV_FILE, override=True)

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
GREY = "\033[90m"
MAGENTA = "\033[95m"
RESET = "\033[0m"

BASE_URL = os.getenv("BASE_URL", "http://localhost:80")

ASSISTANT_ID = "asst_08uPeGvlfBQqDOo5Tl0ZkN"

USER_API_KEY = os.getenv("DEV_PROJECT_DAVID_CORE_TEST_USER_KEY")
PROVIDER_KEY_ID = os.getenv("PD_PROVIDER_KEY_ID")
EXPECTED_PROVIDER = os.getenv("PD_BYOK_PROVIDER", "hyperbolic").lower()

if not USER_API_KEY:
    raise RuntimeError("DEV_PROJECT_DAVID_CORE_TEST_USER_KEY is missing")

if not PROVIDER_KEY_ID:
    raise RuntimeError("PD_PROVIDER_KEY_ID is missing")


TEST_PROMPT = (
    "Please use get_flight_times to retrieve scheduled flights "
    "from LAX to JFK. Use only the returned flight information. "
    "Include the returned verification reference verbatim "
    "in your final answer."
)

VERIFICATION_REFERENCE = "FLIGHTY-PD-72941"

COUNTS = Counter()


# ============================================================
# LOCAL FUNCTION EXECUTOR
# ============================================================


def get_flight_times(
    tool_name: str,
    arguments: dict,
) -> str:
    """
    Deterministic local test fixture.

    No external flight API is called.
    The verification reference proves that the model received
    the actual tool output rather than inventing an answer.
    """

    print(f"\n{YELLOW}[LOCAL EXECUTION] " f"{tool_name}: {arguments}{RESET}")

    departure = arguments.get("departure")
    arrival = arguments.get("arrival")

    if departure != "LAX" or arrival != "JFK":
        raise ValueError("Expected departure=LAX and arrival=JFK")

    COUNTS["handler_invocations"] += 1

    result = {
        "status": "success",
        "source": "synthetic_test_fixture",
        "departure": departure,
        "arrival": arrival,
        "flights": [
            {
                "flight": "PD404",
                "departure_time": "08:15",
                "arrival_time": "16:45",
            },
            {
                "flight": "PD101",
                "departure_time": "09:30",
                "arrival_time": "18:00",
            },
        ],
        "verification_reference": VERIFICATION_REFERENCE,
    }

    print(f"{GREEN}[LOCAL RESULT] " f"{json.dumps(result)}{RESET}")

    return json.dumps(result)


TOOL_REGISTRY = {
    "get_flight_times": get_flight_times,
}


# ============================================================
# ASSISTANT CONTRACT VERIFICATION
# ============================================================


def verify_assistant(assistant):
    if assistant.id != ASSISTANT_ID:
        raise AssertionError("Unexpected assistant identity")

    if not assistant.model:
        raise AssertionError("Assistant has no configured model")

    if assistant.deep_research:
        raise AssertionError("Expected an ordinary function-calling assistant")

    functions = [
        tool["function"]
        for tool in assistant.tools or []
        if isinstance(tool, dict)
        and tool.get("type") == "function"
        and isinstance(tool.get("function"), dict)
        and tool["function"].get("name") == "get_flight_times"
    ]

    if len(functions) != 1:
        raise AssertionError("Expected exactly one get_flight_times definition")

    definition = functions[0]
    parameters = definition.get("parameters") or {}
    properties = parameters.get("properties") or {}

    if parameters.get("type") != "object":
        raise AssertionError("Invalid parameter schema")

    if set(parameters.get("required") or []) != {
        "departure",
        "arrival",
    }:
        raise AssertionError("Required parameters were not preserved")

    for name in ("departure", "arrival"):
        if properties.get(name, {}).get("type") != "string":
            raise AssertionError(f"Parameter {name!r} was not preserved")

    if parameters.get("additionalProperties") is not False:
        raise AssertionError("additionalProperties constraint was not preserved")

    if definition.get("description") != (
        "Retrieve scheduled flights between two IATA airport codes"
    ):
        raise AssertionError("Portal function description update was not preserved")

    print(f"{GREEN}[ASSISTANT VERIFIED]{RESET}")
    print(f"  ID:    {assistant.id}")
    print(f"  Name:  {assistant.name}")
    print(f"  Model: {assistant.model}")

    print("\nPersisted function definition:")
    print(json.dumps(definition, indent=2))


# ============================================================
# MAIN INTEGRATION TEST
# ============================================================


def main():
    start = time.perf_counter()
    run = None
    final_content = []

    print(f"{GREY}[CONFIG] Base URL: {BASE_URL}{RESET}")
    print(f"{GREY}[CONFIG] Assistant: {ASSISTANT_ID}{RESET}")
    print(f"{GREY}[CONFIG] Provider: {EXPECTED_PROVIDER}{RESET}")

    client = Entity(
        base_url=BASE_URL,
        api_key=USER_API_KEY,
    )

    try:
        # ----------------------------------------------------
        # 1. Retrieve the Portal-created assistant.
        # ----------------------------------------------------

        assistant = client.assistants.retrieve_assistant(
            assistant_id=ASSISTANT_ID,
        )

        verify_assistant(assistant)

        configured_provider = assistant.model.lower().split("/", 1)[0]

        if configured_provider != EXPECTED_PROVIDER:
            raise AssertionError(
                "Assistant model does not match " "the configured BYOK provider"
            )

        # ----------------------------------------------------
        # 2. Create Thread, Message and Run.
        # ----------------------------------------------------

        thread = client.threads.create_thread()

        message = client.messages.create_message(
            thread_id=thread.id,
            assistant_id=ASSISTANT_ID,
            role="user",
            content=TEST_PROMPT,
        )

        # Do not override the Portal-configured model.
        run = client.runs.create_run(
            assistant_id=ASSISTANT_ID,
            thread_id=thread.id,
        )

        model = run.model

        if not model:
            raise AssertionError("Core returned a Run without a resolved model")

        if model != assistant.model:
            raise AssertionError(
                f"Run model {model!r} differs from "
                f"assistant model {assistant.model!r}"
            )

        print(f"\n{CYAN}[RUN CREATED]{RESET}")
        print(f"  Thread:  {thread.id}")
        print(f"  Message: {message.id}")
        print(f"  Run:     {run.id}")
        print(f"  Model:   {model}")

        # ----------------------------------------------------
        # 3. Configure unified SDK event stream.
        # ----------------------------------------------------

        stream = client.synchronous_inference_stream

        stream.setup(
            thread_id=thread.id,
            assistant_id=ASSISTANT_ID,
            message_id=message.id,
            run_id=run.id,
            api_key=None,
            provider_key_id=PROVIDER_KEY_ID,
        )

        print(f"\n{CYAN}[UNIFIED STREAM STARTED]{RESET}")

        last_tick = time.perf_counter()

        # ----------------------------------------------------
        # 4. Single SDK-managed event loop.
        # ----------------------------------------------------

        # Observe calls while executing the actual SDK transport unchanged.
        with patch.object(
            client.inference,
            "stream_inference_response",
            wraps=client.inference.stream_inference_response,
        ) as native_transport:
            for event in stream.stream_events(model=model):
                now = time.perf_counter()
                delta = now - last_tick
                last_tick = now

                event_name = type(event).__name__
                COUNTS[event_name] += 1

                if isinstance(event, ToolCallRequestEvent):
                    print(
                        f"\n{YELLOW}[+{delta:.4f}s] "
                        f"[TOOL REQUEST] {event.tool_name}{RESET}"
                    )

                    handler = TOOL_REGISTRY.get(event.tool_name)

                    if handler is None:
                        raise RuntimeError(
                            "Unexpected external tool request: " f"{event.tool_name}"
                        )

                    success = event.execute(handler)

                    if not success:
                        raise RuntimeError(f"Tool execution failed: {event.tool_name}")

                    COUNTS["executed_tools"] += 1

                    print(
                        f"{GREEN}[TOOL OUTPUT SUBMITTED] " f"{event.tool_name}{RESET}"
                    )

                elif isinstance(event, ContentEvent):
                    content = event.content or ""

                    print(
                        f"{GREEN}{content}{RESET}",
                        end="",
                        flush=True,
                    )

                    # Only collect content received after successful
                    # external function execution.
                    if COUNTS["executed_tools"]:
                        final_content.append(content)

                elif isinstance(event, ReasoningEvent):
                    print(
                        f"{CYAN}{event.content or ''}{RESET}",
                        end="",
                        flush=True,
                    )

                elif isinstance(event, DecisionEvent):
                    print(f"\n{MAGENTA}[DECISION] " f"{event.content or ''}{RESET}")

                else:
                    print(f"\n{GREY}[+{delta:.4f}s] " f"{event_name}{RESET}")

            COUNTS["inference_requests"] = native_transport.call_count
            for call in native_transport.call_args_list:
                if call.kwargs.get("provider_key_id") != PROVIDER_KEY_ID:
                    raise AssertionError("Registered credential was not forwarded")
                if call.kwargs.get("api_key") is not None:
                    raise AssertionError("Direct provider credential was supplied")

        # ----------------------------------------------------
        # 5. Runtime acceptance assertions.
        # ----------------------------------------------------

        print(f"\n\n{CYAN}[ACCEPTANCE CHECKS]{RESET}")

        if COUNTS["ToolCallRequestEvent"] == 0:
            raise AssertionError("No external function request received")

        if COUNTS["handler_invocations"] == 0:
            raise AssertionError("Local function did not execute successfully")

        if COUNTS["executed_tools"] == 0:
            raise AssertionError("SDK did not confirm tool-output submission")

        if COUNTS["inference_requests"] < 2:
            raise AssertionError("SDK did not initiate the continuation turn")

        answer = "".join(final_content)

        if not answer.strip():
            raise AssertionError("No final content received after tool execution")

        if VERIFICATION_REFERENCE not in answer:
            raise AssertionError(
                "Final response did not include "
                "the tool-output verification reference"
            )

        # ----------------------------------------------------
        # 6. Verify persisted Run.
        # ----------------------------------------------------

        persisted_run = client.runs.retrieve_run(
            run_id=run.id,
        )

        status = getattr(persisted_run, "status", None)
        status = getattr(status, "value", status)

        if status != "completed":
            raise AssertionError(f"Expected completed Run, got {status!r}")

        meta_data = getattr(persisted_run, "meta_data", None) or {}

        if hasattr(meta_data, "model_dump"):
            meta_data = meta_data.model_dump()

        if not isinstance(meta_data, dict):
            raise AssertionError("Unexpected Run metadata format")

        for field in (
            "api_key",
            "inference_api_key",
            "project_david_api_key",
        ):
            if field in meta_data:
                raise AssertionError(f"Sensitive credential field persisted: {field}")

        if USER_API_KEY in json.dumps(meta_data):
            raise AssertionError("Core API credential leaked into Run metadata")

        print(f"{GREEN}[PASS] Portal function persistence{RESET}")
        print(f"{GREEN}[PASS] Registered BYOK inference{RESET}")
        print(f"{GREEN}[PASS] External function dispatch{RESET}")
        print(f"{GREEN}[PASS] Local function execution{RESET}")
        print(f"{GREEN}[PASS] Tool-output submission{RESET}")
        print(f"{GREEN}[PASS] SDK continuation turn{RESET}")
        print(f"{GREEN}[PASS] Tool-output consumption{RESET}")
        print(f"{GREEN}[PASS] Run completion{RESET}")
        print(f"{GREEN}[PASS] Run metadata credential check{RESET}")

        print(f"\n{GREEN}" "[PASS] PORTAL LOCAL FUNCTION — END TO END" f"{RESET}")

    except Exception as exc:
        print(f"\n{RED}[FAIL] {type(exc).__name__}: " f"{exc}{RESET}")
        raise

    finally:
        duration = time.perf_counter() - start

        print("\n" + "=" * 60)
        print(f"ASSISTANT: {ASSISTANT_ID}")
        print(f"RUN:       {run.id if run else 'Not created'}")
        print(f"DURATION:  {duration:.4f}s")

        print("\nEVENT COUNTS:")

        for name, count in sorted(COUNTS.items()):
            print(f"  {name}: {count}")

        print("=" * 60)


if __name__ == "__main__":
    main()
