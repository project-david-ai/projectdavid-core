"""
tests/integration/assistant_intrusion_test.py

Assistant ownership isolation sweep for Project David.
Tests that row-level access control is correctly enforced on all
assistant endpoints.

Fixtures:
  owner    — authenticated via OWNER_API_KEY
  intruder — authenticated via INTRUDER_API_KEY

Run directly:
    python -m tests.integration.assistant_intrusion_test

Or via the compliance runner (exposes run_sweep()).
"""

import os
from typing import Optional

from dotenv import load_dotenv
from projectdavid import Entity

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
BASE_URL = os.getenv("ENTITIES_BASE_URL", "http://localhost:9000")
OWNER_KEY = os.getenv("OWNER_API_KEY")
INTRUDER_KEY = os.getenv("INTRUDER_API_KEY")

if not OWNER_KEY or not INTRUDER_KEY:
    raise RuntimeError(
        "Set OWNER_API_KEY and INTRUDER_API_KEY in your environment or .env file."
    )

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
results: dict = {}


def record(label: str, passed: bool, note: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    tag = "✅" if passed else "❌"
    print(f"[{status}] {tag}  {label}" + (f": {note}" if note else ""))
    results[label] = status


def expect_success(label: str, fn, *args, **kwargs):
    try:
        result = fn(*args, **kwargs)
        record(label, True)
        return result
    except Exception as exc:
        record(label, False, str(exc))
        return None


def expect_error(label: str, expected_codes: tuple, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
        record(label, False, "Expected error but call succeeded")
        return None
    except Exception as exc:
        msg = str(exc)
        matched = any(str(code) in msg for code in expected_codes)
        record(label, matched, msg)
        return exc


# ──────────────────────────────────────────────────────────────────────────────
# Clients
# ──────────────────────────────────────────────────────────────────────────────
owner_client = Entity(base_url=BASE_URL, api_key=OWNER_KEY)
intruder_client = Entity(base_url=BASE_URL, api_key=INTRUDER_KEY)

# ──────────────────────────────────────────────────────────────────────────────
# State
# ──────────────────────────────────────────────────────────────────────────────
owner_assistant_id: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Sweep
# ──────────────────────────────────────────────────────────────────────────────
def run_sweep() -> dict:
    global owner_assistant_id
    results.clear()

    print("\n" + "═" * 60)
    print("ASSISTANT ISOLATION SWEEP")
    print("═" * 60)

    # ── Test 1: Owner creates assistant ──────────────────────────────────────
    print("\n--- Test 1: Owner creates assistant (expecting success) ---")
    assistant = expect_success(
        "Test 1: Owner creates assistant",
        owner_client.assistants.create_assistant,
        name="Isolation Test Assistant",
        model="gpt-oss-120b",
        instructions="You are a test assistant for isolation verification.",
    )
    if assistant:
        owner_assistant_id = assistant.id
        print(f"  assistant.id      : {assistant.id}")
        print(f"  assistant.name    : {assistant.name}")

    # ── Test 2: Owner retrieves own assistant ─────────────────────────────────
    print("\n--- Test 2: Owner retrieves own assistant (expecting success) ---")
    if owner_assistant_id:
        expect_success(
            "Test 2: Owner retrieves own assistant",
            owner_client.assistants.retrieve_assistant,
            owner_assistant_id,
        )

    # ── Test 3: Intruder retrieves owner's assistant ──────────────────────────
    print("\n--- Test 3: Intruder retrieves owner's assistant (expecting 403/404) ---")
    if owner_assistant_id:
        expect_error(
            "Test 3: Intruder retrieves owner's assistant",
            (403, 404),
            intruder_client.assistants.retrieve_assistant,
            owner_assistant_id,
        )

    # ── Test 4: Owner updates own assistant ───────────────────────────────────
    print("\n--- Test 4: Owner updates own assistant (expecting success) ---")
    if owner_assistant_id:
        updated = expect_success(
            "Test 4: Owner updates own assistant",
            owner_client.assistants.update_assistant,
            assistant_id=owner_assistant_id,
            name="Isolation Test Assistant (Updated)",
        )
        if updated:
            print(f"  updated.name : {updated.name}")

    # ── Test 5: Intruder updates owner's assistant ────────────────────────────
    print("\n--- Test 5: Intruder updates owner's assistant (expecting 403/404) ---")
    if owner_assistant_id:
        expect_error(
            "Test 5: Intruder updates owner's assistant",
            (403, 404),
            intruder_client.assistants.update_assistant,
            assistant_id=owner_assistant_id,
            name="Hijacked Name",
        )

    # ── Test 6: Intruder creates run against owner's assistant ────────────────
    print(
        "\n--- Test 6: Intruder creates run against owner's assistant (expecting 403) ---"
    )
    if owner_assistant_id:
        try:
            intruder_thread = (
                owner_client.threads.create_thread()
            )  # intruder uses own thread
            # Re-create using intruder client to properly test the run isolation
            intruder_thread2 = intruder_client.threads.create_thread()
            expect_error(
                "Test 6: Intruder creates run against owner's assistant",
                (403, 404),
                intruder_client.runs.create_run,
                thread_id=intruder_thread2.id,
                assistant_id=owner_assistant_id,
            )
            # Cleanup intruder thread
            try:
                intruder_client.threads.delete_thread(intruder_thread2.id)
            except Exception:
                pass
            try:
                owner_client.threads.delete_thread(intruder_thread.id)
            except Exception:
                pass
        except Exception as exc:
            record(
                "Test 6: Intruder creates run against owner's assistant",
                False,
                str(exc),
            )

    # ── Test 7: Owner lists assistants — no cross-user leakage ───────────────
    print("\n--- Test 7: Owner lists assistants (expecting only own) ---")
    if owner_assistant_id:
        try:
            owner_list = owner_client.assistants.list_assistants()
            ids = [
                a.id
                for a in (
                    owner_list.data if hasattr(owner_list, "data") else owner_list
                )
            ]
            own_present = owner_assistant_id in ids
            record("Test 7: Owner's assistant appears in own list", own_present)
        except Exception as exc:
            record("Test 7: Owner's assistant appears in own list", False, str(exc))

    # ── Test 8: Intruder lists assistants — must not see owner's ─────────────
    print("\n--- Test 8: Intruder lists assistants (must not see owner's) ---")
    if owner_assistant_id:
        try:
            intruder_list = intruder_client.assistants.list_assistants()
            ids = [
                a.id
                for a in (
                    intruder_list.data
                    if hasattr(intruder_list, "data")
                    else intruder_list
                )
            ]
            leaked = owner_assistant_id in ids
            record("Test 8: Owner's assistant not visible to intruder", not leaked)
        except Exception as exc:
            record("Test 8: Owner's assistant not visible to intruder", False, str(exc))

    # ── Test 9: Intruder deletes owner's assistant ────────────────────────────
    print("\n--- Test 9: Intruder deletes owner's assistant (expecting 403/404) ---")
    if owner_assistant_id:
        expect_error(
            "Test 9: Intruder deletes owner's assistant",
            (403, 404),
            intruder_client.assistants.delete_assistant,
            owner_assistant_id,
        )

    # ── Test 10: Owner deletes own assistant ──────────────────────────────────
    print("\n--- Test 10: Owner deletes own assistant (expecting success) ---")
    if owner_assistant_id:
        expect_success(
            "Test 10: Owner deletes own assistant",
            owner_client.assistants.delete_assistant,
            owner_assistant_id,
        )

    # ── Test 11: Deleted assistant returns 404 for owner ─────────────────────
    print("\n--- Test 11: Deleted assistant returns 404 for owner (expecting 404) ---")
    if owner_assistant_id:
        expect_error(
            "Test 11: Deleted assistant returns 404 for owner",
            (404,),
            owner_client.assistants.retrieve_assistant,
            owner_assistant_id,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("ASSISTANT ISOLATION SWEEP — SUMMARY")
    print("═" * 60)
    for label, outcome in results.items():
        tag = "✅" if outcome == "PASS" else "❌"
        print(f"  {tag}  {label}: {outcome}")
    print("═" * 60)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_sweep()
