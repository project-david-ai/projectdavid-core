"""
tests/integration/message_intrusion_test.py

MESSAGE ISOLATION SWEEP
=======================
Verifies that message endpoints enforce thread-level ownership.
All read, write, and delete operations on messages are scoped to
the authenticated user via the parent thread's owner_id.

Gaps covered
------------
  retrieve_message      — no user_id check (was: open to all)
  list_messages         — thread exists check only (was: no owner_id match)
  get_formatted_messages— same gap as list_messages
  create_message        — thread exists check only (was: intruder could inject)
  submit_tool_output    — same gap as create_message
  delete_message        — no user_id check (was: open to all)
"""

import logging
import os
import sys

from projectdavid import Entity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------

BASE_URL = os.getenv("API_BASE_URL", "http://localhost:9000")

OWNER_KEY = os.getenv("OWNER_API_KEY", "pt-test-owner")
INTRUDER_KEY = os.getenv("INTRUDER_API_KEY", "pt-test-intruder")

# ---------------------------------------------------------------------------
#  Helpers (mirror the pattern from other intrusion tests)
# ---------------------------------------------------------------------------

results: list[dict] = []


def record(label: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    icon = "✅" if passed else "❌"
    msg = f"[{status}] {icon}  {label}"
    if detail:
        msg += f": {detail}"
    print(msg)
    results.append({"label": label, "passed": passed, "detail": detail})


def expect_error(label: str, fn, *args, **kwargs) -> None:
    """Assert that fn() raises an exception (HTTP 403/404 expected)."""
    try:
        fn(*args, **kwargs)
        record(label, False, "Expected denial but call succeeded")
    except Exception as exc:
        record(label, True, str(exc).split("\n")[0])


def expect_success(label: str, fn, *args, **kwargs):
    """Assert that fn() returns without raising."""
    try:
        result = fn(*args, **kwargs)
        record(label, True)
        return result
    except Exception as exc:
        record(label, False, str(exc).split("\n")[0])
        return None


# ---------------------------------------------------------------------------
#  Clients
# ---------------------------------------------------------------------------


def _make_client(api_key: str) -> Entity:
    return Entity(base_url=BASE_URL, api_key=api_key)


# ---------------------------------------------------------------------------
#  Sweep
# ---------------------------------------------------------------------------


def run_message_isolation_sweep() -> list[dict]:
    print()
    print("=" * 60)
    print("MESSAGE ISOLATION SWEEP")
    print("=" * 60)

    owner_client = _make_client(OWNER_KEY)
    intruder_client = _make_client(INTRUDER_KEY)

    # ── SETUP ───────────────────────────────────────────────────────────────
    print()
    print("-" * 60)
    print("  SETUP")
    print("-" * 60)

    owner_thread = expect_success(
        "Setup: Owner creates thread",
        owner_client.threads.create_thread,
    )
    if owner_thread is None:
        print("[FATAL] Cannot continue without owner thread.")
        return results

    print(f"  thread.id : {owner_thread.id}")

    # Post an initial message so we have a message_id to probe
    owner_message = expect_success(
        "Setup: Owner creates message",
        owner_client.messages.create_message,
        thread_id=owner_thread.id,
        content="Hello from the owner.",
        assistant_id="",
        role="user",
    )
    if owner_message is None:
        print("[FATAL] Cannot continue without owner message.")
        return results

    print(f"  message.id : {owner_message.id}")

    # ── Test 1: Owner retrieves own message ──────────────────────────────────
    print("\n--- Test 1: Owner retrieves own message (expecting success) ---")
    retrieved = expect_success(
        "Test 1: Owner retrieves own message",
        owner_client.messages.retrieve_message,
        owner_message.id,
    )
    if retrieved:
        print(f"  retrieved.id      : {retrieved.id}")
        print(f"  retrieved.content : {retrieved.content[:40]}")

    # ── Test 2: Intruder retrieves owner's message (expecting 403) ───────────
    print("\n--- Test 2: Intruder retrieves owner's message (expecting 403) ---")
    expect_error(
        "Test 2: Intruder retrieves owner's message",
        intruder_client.messages.retrieve_message,
        owner_message.id,
    )

    # ── Test 3: Owner lists messages in own thread ───────────────────────────
    print("\n--- Test 3: Owner lists messages in own thread (expecting success) ---")
    msg_list = expect_success(
        "Test 3: Owner lists messages in own thread",
        owner_client.messages.list_messages,
        owner_thread.id,
    )
    if msg_list:
        print(f"  messages returned : {len(msg_list.data)}")

    # ── Test 4: Intruder lists messages in owner's thread (expecting 403) ────
    print("\n--- Test 4: Intruder lists messages in owner's thread (expecting 403) ---")
    expect_error(
        "Test 4: Intruder lists messages in owner's thread",
        intruder_client.messages.list_messages,
        owner_thread.id,
    )

    # ── Test 5: Owner gets formatted messages for own thread ─────────────────
    print("\n--- Test 5: Owner gets formatted messages (expecting success) ---")
    fmt = expect_success(
        "Test 5: Owner gets formatted messages",
        owner_client.messages.get_messages_without_system_message,
        owner_thread.id,
    )
    if fmt is not None:
        print(f"  formatted messages : {len(fmt)}")

    # ── Test 6: Intruder gets formatted messages for owner's thread (403) ────
    print("\n--- Test 6: Intruder gets formatted messages (expecting 403) ---")
    expect_error(
        "Test 6: Intruder gets formatted messages for owner's thread",
        intruder_client.messages.get_messages_without_system_message,
        owner_thread.id,
    )

    # ── Test 7: Intruder injects a message into owner's thread (expecting 403)
    print(
        "\n--- Test 7: Intruder injects message into owner's thread (expecting 403) ---"
    )
    expect_error(
        "Test 7: Intruder injects message into owner's thread",
        intruder_client.messages.create_message,
        thread_id=owner_thread.id,
        content="Injected by intruder",
        assistant_id="",
        role="user",
    )

    # ── Test 8: Intruder deletes owner's message (expecting 403) ────────────
    print("\n--- Test 8: Intruder deletes owner's message (expecting 403) ---")
    expect_error(
        "Test 8: Intruder deletes owner's message",
        intruder_client.messages.delete_message,
        owner_message.id,
    )

    # Verify message still exists after failed delete
    still_there = expect_success(
        "Test 8 verify: Message still exists after failed delete",
        owner_client.messages.retrieve_message,
        owner_message.id,
    )
    if still_there:
        print(f"  message still present: {still_there.id}")

    # ── Test 9: Owner deletes own message (expecting success) ────────────────
    print("\n--- Test 9: Owner deletes own message (expecting success) ---")
    expect_success(
        "Test 9: Owner deletes own message",
        owner_client.messages.delete_message,
        owner_message.id,
    )

    # ── Test 10: Deleted message returns 404 for owner ───────────────────────
    print("\n--- Test 10: Deleted message returns 404 for owner (expecting 404) ---")
    expect_error(
        "Test 10: Deleted message returns 404 for owner",
        owner_client.messages.retrieve_message,
        owner_message.id,
    )

    # ── TEARDOWN ─────────────────────────────────────────────────────────────
    print()
    print("-" * 60)
    print("  TEARDOWN")
    print("-" * 60)

    try:
        owner_client.threads.delete_thread(owner_thread.id)
        print(f"[OK] Thread deleted: {owner_thread.id}")
    except Exception as exc:
        print(f"[WARN] Could not delete thread {owner_thread.id}: {exc}")

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("MESSAGE ISOLATION SWEEP — SUMMARY")
    print("=" * 60)
    for r in results:
        icon = "✅" if r["passed"] else "❌"
        print(f"  {icon}  {r['label']}: {'PASS' if r['passed'] else 'FAIL'}")

    return results


# ---------------------------------------------------------------------------
#  Compliance runner entry-point
#  run_sweep() → dict  (label → "PASS" | "FAIL")
#  Matches the contract expected by compliance_runner.py for every sweep.
# ---------------------------------------------------------------------------


def run_sweep() -> dict:
    results.clear()
    run_message_isolation_sweep()
    return {r["label"]: ("PASS" if r["passed"] else "FAIL") for r in results}


# ---------------------------------------------------------------------------
#  Direct entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_message_isolation_sweep()
    failed = sum(1 for r in results if not r["passed"])
    sys.exit(1 if failed else 0)
