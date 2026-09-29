"""Deep Research integration test using the existing live-test configuration.

Run from the same working directory as the earlier integration script:
    python deep_research_shared_scratchpad.py

Configuration is loaded from .tests.env, with the same variable names, precedence,
localhost:80 fallback, and model default as that script. No config module is needed.
The selected assistant must have Deep Research enabled; the test does not alter it.

The inline prompt drives two dependent research delegations. Scratchpad evidence
must show supervisor -> worker A -> supervisor -> worker B -> supervisor reads
and writes, using distinct worker IDs and markers unique to this run.
Final-answer claims are not evidence. A successful check demonstrates functional
shared visibility, not Redis-key identity or concurrent-write isolation.

Research tools execute in Core. Any consumer-tool request is recorded as a test
failure, and no local executor is invoked. Server records are retained for
inspection; the script prints their IDs and returns a nonzero exit code on failure.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys
import time
import traceback
from collections import Counter
from contextlib import closing
from dataclasses import dataclass
from typing import Any, Callable, Literal
from uuid import uuid4

from dotenv import load_dotenv
from projectdavid import Entity
from projectdavid.events import (
    ContentEvent,
    DecisionEvent,
    ReasoningEvent,
    ResearchStatusEvent,
    ScratchpadEvent,
    ToolCallRequestEvent,
    ToolInterceptEvent,
    WebStatusEvent,
)
from pydantic import BaseModel


# ------------------------------------------------------------------
# 0. CONFIGURATION -- retained from the earlier live integration script
# ------------------------------------------------------------------
load_dotenv(Path(__file__).with_name(".tests.env"))

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
GREY = "\033[90m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"
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

TIMEOUT_PER_CHUNK = 600.0
MAX_SDK_TURNS = 10

# Inline prompt and run-specific scratchpad markers.
TEST_ID = uuid4().hex[:12]

SEED_VALUE = uuid4().hex

SEED_KEY = f"SP_{TEST_ID}_SEED"

WORKER_A_KEY = f"SP_{TEST_ID}_WORKER_A"

WORKER_B_KEY = f"SP_{TEST_ID}_WORKER_B_ACK"

SEED_LINE = f"{SEED_KEY}={SEED_VALUE}"

TEST_PROMPT = f"""
Run a small Deep Research task that also checks shared scratchpad visibility.
Test identifier: {TEST_ID}.

Follow your normal research and citation rules. Use actual tools; describing
tool calls or claiming that the scratchpad is shared does not satisfy this task.
There are two dependent research tasks, so delegate A and B separately, in order.

1. SUPERVISOR INITIALISATION
   First call update_scratchpad with a [STRATEGY] block containing your plan and
   this exact standalone line:
   {SEED_LINE}
   Complete that write before delegating. Preserve all SP_{TEST_ID}_ marker
   lines in any subsequent scratchpad update, including worker-written lines.

2. DELEGATE WORKER A
   Research one narrow fact: the command for creating a virtual environment,
   using official Python documentation retrieved during this run.
   Give the worker a task, strategy, and output format. It must read_scratchpad
   with its first research action, find the value under {SEED_KEY}, claim its
   task, search for the official documentation, read the retrieved page, and
   use search_web_page to verify the command (fallback search term: "venv").
   Before returning, it must append a source-backed finding and a fresh marker:
   {WORKER_A_KEY}=<a newly chosen 16-character lowercase hexadecimal token>
   The worker chooses that token; you must not choose or supply it.
   You may pass marker KEY NAMES and the test identifier in the delegation,
   but MUST NOT copy the seed VALUE or scratchpad contents into its prompt.
   The seed must be obtained from its actual scratchpad read.
   On an unavailable source, record the failure and make at most one fallback
   search. Never invent a source or claim verification without reading it.
   The worker's text return must be a one-line confirmation only, without
   repeating marker values or research results from the scratchpad.

3. SUPERVISOR REVIEW A
   After A returns, call read_scratchpad yourself. Confirm that your seed line
   and A's actual marker are present before starting the next delegation.

4. DELEGATE A DISTINCT WORKER B
   Research a second narrow fact: the command for starting Python's built-in
   HTTP server, using official Python documentation retrieved during this run.
   Its first actions must include read_scratchpad and a research action.
   Give it the same search -> read -> search_web_page strategy, with fallback
   search term "http.server", a task claim, and the same source-failure policy.
   It must obtain {SEED_KEY} and {WORKER_A_KEY} from its scratchpad read, then
   append its source-backed finding together with this acknowledgement:
   {WORKER_B_KEY}=<the exact token read under {WORKER_A_KEY}>
   Pass only marker KEY NAMES and this test identifier to B. Do not copy the
   seed VALUE, A's token, A's report, or other scratchpad contents into B's task.
   Again, require a one-line text confirmation without repeating marker values.

5. SUPERVISOR FINAL REVIEW
   After B returns, call read_scratchpad yourself. Verify that the original seed,
   A's marker, and B's acknowledgement are all visible, and that the two worker
   marker values match. Do not create or repair worker markers yourself.
   If a marker is missing, report exactly what is missing rather than guessing.
   Finish with the two researched commands and their source URLs, followed by
   a brief account of the marker check. Keep the final answer under 250 words.
""".strip()


@dataclass(frozen=True)
class PadObservation:
    event_index: int
    scope: Literal["supervisor", "worker"]
    delegation: int | None
    assistant_id: str
    operation: str
    text: str


class EvidenceCheck(BaseModel):
    passed: bool
    evidence: str


class ProofReport(BaseModel):
    test_id: str
    passed: bool
    checks: dict[str, EvidenceCheck]


def evaluate_scratchpad(observations: list[PadObservation]) -> ProofReport:
    """Require an ordered chain of real write/read events across distinct actors.

    Supervisor identity can change between Core inference turns. Roles therefore
    use delegation lifecycle boundaries, not equality to the configured assistant.
    Worker identity must stay consistent within each write/read pair.
    """
    checks: dict[str, EvidenceCheck] = {}

    def find(
        label: str, predicate: Callable[[PadObservation], bool]
    ) -> PadObservation | None:
        match = next(
            (item for item in observations if item.assistant_id and predicate(item)),
            None,
        )
        checks[label] = EvidenceCheck(
            passed=match is not None,
            evidence=(
                f"event={match.event_index} assistant={match.assistant_id} "
                f"scope={match.scope} delegation={match.delegation}"
                if match
                else "Required scratchpad evidence was not observed."
            ),
        )
        return match

    def contains(text: str, marker: str) -> bool:
        return re.search(re.escape(marker) + r"(?![A-Za-z0-9_])", text) is not None

    seed = find(
        "supervisor_wrote_seed",
        lambda o: o.scope == "supervisor"
        and o.operation == "update"
        and contains(o.text, SEED_LINE),
    )
    read_a = find(
        "worker_a_read_supervisor_seed",
        lambda o: bool(seed)
        and o.event_index > seed.event_index
        and o.scope == "worker"
        and o.operation == "read"
        and o.assistant_id != seed.assistant_id
        and contains(o.text, SEED_LINE),
    )
    token_pattern = re.compile(
        re.escape(WORKER_A_KEY) + r"=([0-9a-f]{16})(?![A-Za-z0-9_])"
    )
    write_a = find(
        "worker_a_appended_own_marker",
        lambda o: bool(read_a)
        and o.event_index > read_a.event_index
        and o.scope == "worker"
        and o.operation == "append"
        and o.delegation == read_a.delegation
        and o.assistant_id == read_a.assistant_id
        and token_pattern.search(o.text) is not None,
    )
    token = token_pattern.search(write_a.text).group(1) if write_a else ""
    line_a, line_b = f"{WORKER_A_KEY}={token}", f"{WORKER_B_KEY}={token}"
    review_a = find(
        "supervisor_read_worker_a_marker",
        lambda o: bool(write_a)
        and o.event_index > write_a.event_index
        and o.scope == "supervisor"
        and o.operation == "read"
        and o.assistant_id != write_a.assistant_id
        and contains(o.text, SEED_LINE)
        and contains(o.text, line_a),
    )
    read_b = find(
        "distinct_worker_b_read_seed_and_a_marker",
        lambda o: bool(review_a)
        and o.event_index > review_a.event_index
        and o.scope == "worker"
        and o.operation == "read"
        and o.delegation != read_a.delegation
        and o.assistant_id
        not in {read_a.assistant_id, review_a.assistant_id, seed.assistant_id}
        and contains(o.text, SEED_LINE)
        and contains(o.text, line_a),
    )
    write_b = find(
        "worker_b_appended_matching_acknowledgement",
        lambda o: bool(read_b)
        and o.event_index > read_b.event_index
        and o.scope == "worker"
        and o.operation == "append"
        and o.delegation == read_b.delegation
        and o.assistant_id == read_b.assistant_id
        and contains(o.text, line_b),
    )
    find(
        "supervisor_read_both_worker_markers",
        lambda o: bool(write_b)
        and o.event_index > write_b.event_index
        and o.scope == "supervisor"
        and o.operation == "read"
        and o.assistant_id not in {read_a.assistant_id, read_b.assistant_id}
        and all(contains(o.text, line) for line in (SEED_LINE, line_a, line_b)),
    )
    return ProofReport(
        test_id=TEST_ID,
        passed=all(check.passed for check in checks.values()),
        checks=checks,
    )


def field(obj: Any, name: str, default: Any = None) -> Any:
    return (
        obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)
    )


def main() -> int:
    if not API_KEY:
        raise RuntimeError("Missing DEV_PROJECT_DAVID_CORE_TEST_USER_KEY")
    if not ASSISTANT_ID:
        raise RuntimeError("Missing ASSISTANT_ID")
    if not os.getenv("TOGETHER_API_KEY"):
        raise RuntimeError("Missing TOGETHER_API_KEY")

    print(f"{GREY}[CONFIG] Base URL:     {BASE_URL}{RESET}")
    print(f"{GREY}[CONFIG] Assistant ID: {ASSISTANT_ID}{RESET}")
    print(f"{GREY}[CONFIG] Model ID:     {MODEL_ID}{RESET}")
    print(f"{GREY}[CONFIG] Test ID:      {TEST_ID}{RESET}")

    # 1. SDK INIT
    client = Entity(base_url=BASE_URL, api_key=API_KEY)

    # 2. ASSISTANT PREFLIGHT
    print(f"\n{CYAN}=== ASSISTANT PREFLIGHT ==={RESET}")
    assistant = client.assistants.retrieve_assistant(ASSISTANT_ID)
    for name in ("id", "model", "max_turns", "deep_research", "is_engineer"):
        print(f"assistant.{name}={field(assistant, name, 'not exposed')}")
    for tool in field(assistant, "tools", []) or []:
        if hasattr(tool, "model_dump"):
            tool = tool.model_dump()
        if isinstance(tool, dict):
            function = tool.get("function") or {}
            name = field(function, "name") or tool.get("name") or tool.get("type")
            if name:
                print(f"tool={name}")

    # Supervisor tools may be supplied by Core's ephemeral identity swap.
    # The original assistant's tool list is informational, not a required list.
    deep_research = field(assistant, "deep_research")
    if deep_research is not None and str(deep_research).lower() in {"false", "0"}:
        raise RuntimeError(
            "ASSISTANT_ID identifies an assistant with deep_research disabled. "
            "Enable Deep Research on that assistant before running this test."
        )
    if str(field(assistant, "is_engineer", False)).lower() in {"true", "1"}:
        raise RuntimeError(
            "The selected assistant has Engineer mode enabled, which takes priority over Deep Research."
        )
    if field(assistant, "max_turns") == 1:
        print(
            f"{YELLOW}[NOTE] Assistant max_turns=1; inspect this if research cannot finish.{RESET}"
        )
    print(f"{GREEN}ASSISTANT_PREFLIGHT=PASS{RESET}")

    # 3. INLINE TEST PROMPT
    print(f"\n{CYAN}=== TEST PROMPT ==={RESET}\n{TEST_PROMPT}")

    # 4. CREATE THREAD / MESSAGE / RUN
    print(f"\n{CYAN}=== CREATE INFERENCE STATE ==={RESET}")
    global_start = time.perf_counter()
    thread = client.threads.create_thread()
    print(f"thread.id={thread.id}", flush=True)
    message = client.messages.create_message(
        thread_id=thread.id,
        role="user",
        content=TEST_PROMPT,
        assistant_id=ASSISTANT_ID,
    )
    print(f"message.id={message.id}", flush=True)
    run = client.runs.create_run(assistant_id=ASSISTANT_ID, thread_id=thread.id)
    print(f"run.id={run.id}", flush=True)

    # 5. SETUP UNIFIED STREAM -- same provider credential as the base script
    stream = client.synchronous_inference_stream
    stream.setup(
        thread_id=thread.id,
        assistant_id=ASSISTANT_ID,
        message_id=message.id,
        run_id=run.id,
        api_key=os.getenv("TOGETHER_API_KEY"),
    )

    # 6. STREAM
    print(f"\n{CYAN}=== LIVE DEEP RESEARCH STREAM ==={RESET}")
    print(f"{'LATENCY':<14} | {'EVENT CLASS':<28} | PAYLOAD")
    print("-" * 120)
    observations: list[PadObservation] = []
    counts: Counter[str] = Counter()
    content_chunks: list[str] = []
    failures: list[str] = []
    delegation_count = 0
    active_delegation: int | None = None
    saw_client_tool_request = False
    stream_completed = False
    last_tick = time.perf_counter()

    try:
        with closing(
            stream.stream_events(
                model=MODEL_ID,
                timeout_per_chunk=TIMEOUT_PER_CHUNK,
                max_turns=MAX_SDK_TURNS,
            )
        ) as events:
            for index, event in enumerate(events, 1):
                current_tick = time.perf_counter()
                time_str = f"[{current_tick - last_tick:+.4f}s]"
                last_tick = current_tick
                class_name = type(event).__name__
                counts[class_name] += 1
                payload = event.to_dict()
                payload_json = json.dumps(payload, ensure_ascii=False, default=str)

                color = RESET
                if isinstance(event, ContentEvent):
                    color = GREEN
                elif isinstance(event, ReasoningEvent):
                    color = CYAN
                elif isinstance(event, DecisionEvent):
                    color = MAGENTA
                elif isinstance(
                    event, (ResearchStatusEvent, ScratchpadEvent, WebStatusEvent)
                ):
                    color = BLUE
                elif isinstance(event, (ToolCallRequestEvent, ToolInterceptEvent)):
                    color = RED
                print(
                    f"{GREY}{time_str:<14}{RESET} | "
                    f"{color}{class_name:<28}{RESET} | #{index} {payload_json}",
                    flush=True,
                )

                if isinstance(event, ResearchStatusEvent):
                    # Current Core dispatches delegations sequentially. Worker
                    # run IDs are remapped to the parent, and the SDK drops origin;
                    # classify roles by lifecycle boundaries plus assistant IDs.
                    if event.tool == "delegate_research_task":
                        if event.state == "in_progress" and active_delegation is None:
                            delegation_count += 1
                            active_delegation = delegation_count
                        elif event.state in {"completed", "error", "failed"}:
                            active_delegation = None
                        if event.state in {"error", "failed"}:
                            failures.append(f"Delegation error: {event.activity}")
                elif isinstance(event, ScratchpadEvent):
                    entry = event.entry or event.content or ""
                    if event.state == "success" and entry:
                        observations.append(
                            PadObservation(
                                event_index=index,
                                scope=(
                                    "worker"
                                    if active_delegation is not None
                                    else "supervisor"
                                ),
                                delegation=active_delegation,
                                assistant_id=event.assistant_id or "",
                                operation=event.operation,
                                text=entry,
                            )
                        )
                    elif event.state in {"error", "failed"}:
                        failures.append(f"Scratchpad error: {event.activity}")
                elif isinstance(event, (ToolCallRequestEvent, ToolInterceptEvent)):
                    # Core owns every tool needed for this research scenario.
                    # Observe the boundary failure without executing local tools.
                    saw_client_tool_request = True
                    failures.append(
                        f"Unexpected client-side tool request: {event.tool_name}"
                    )
                elif isinstance(event, ContentEvent):
                    if event.content:
                        content_chunks.append(event.content)
                elif isinstance(event, WebStatusEvent) and event.status == "failed":
                    failures.append(f"Stream failure: {event.message or event.status}")
            stream_completed = True
    except Exception as exc:
        failures.append(f"Stream raised {type(exc).__name__}: {exc}")
        traceback.print_exc()

    # 7. RESULTS
    proof = evaluate_scratchpad(observations)
    try:
        final_run = client.runs.retrieve_run(run.id)
        status = field(final_run, "status", "unknown")
        final_status = str(getattr(status, "value", status))
    except Exception as exc:
        final_status = "unknown"
        failures.append(f"Could not retrieve final run status: {exc}")
    if final_status != "completed":
        failures.append(f"Run status={final_status}; expected completed.")
    if active_delegation is not None:
        failures.append("Stream ended before the active delegation completed.")
    if not counts["ResearchStatusEvent"]:
        failures.append("No ResearchStatusEvent was observed.")
    streamed_text = "".join(content_chunks).strip()
    if not streamed_text:
        failures.append("No non-empty ContentEvent was observed.")

    print(f"\n{YELLOW}{'=' * 72}\nDEEP RESEARCH RESULTS\n{'=' * 72}{RESET}")
    print(f"THREAD_ID={thread.id}\nMESSAGE_ID={message.id}\nRUN_ID={run.id}")
    print(f"MODEL_ID={MODEL_ID}\nTEST_ID={TEST_ID}")
    print(f"STREAM_COMPLETED={'YES' if stream_completed else 'NO'}")
    print(f"CLIENT_TOOL_REQUEST_SURFACED={'YES' if saw_client_tool_request else 'NO'}")
    print("LOCAL_TOOL_EXECUTOR_USED=NO")
    print(f"DELEGATIONS_OBSERVED={delegation_count}\nRUN_STATUS={final_status}")
    print("EVENT_COUNTS=" + json.dumps(dict(counts), sort_keys=True))
    print(f"TOTAL_ROUND_TRIP_SECONDS={time.perf_counter() - global_start:.4f}")
    print("\nSCRATCHPAD PROOF (successful tool events only):")
    print(proof.model_dump_json(indent=2))

    # 8. STREAMED CONTENT -- may contain forwarded worker text as well as synthesis
    if streamed_text:
        print(f"\n{GREEN}=== STREAMED MODEL CONTENT ==={RESET}\n{streamed_text}")

    # 9. ACCEPTANCE
    print(f"\n{CYAN}=== ACCEPTANCE ==={RESET}")
    inference_ok = stream_completed and bool(streamed_text)
    boundary_ok = not saw_client_tool_request
    print(f"INFERENCE_STREAM={'PASS' if inference_ok else 'FAIL'}")
    print(f"CORE_TOOL_BOUNDARY={'PASS' if boundary_ok else 'FAIL'}")
    print(f"SHARED_SCRATCHPAD={'PASS' if proof.passed else 'NOT_PROVEN'}")
    for failure in failures:
        print(f"FAILURE={failure}")
    passed = inference_ok and boundary_ok and proof.passed and not failures
    color = GREEN if passed else RED
    print(
        f"{color}DEEP_RESEARCH_INTEGRATION={'PASS' if passed else 'FAIL'}{RESET}",
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nINTERRUPTED: no integration result was established.", file=sys.stderr)
        raise SystemExit(130)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
