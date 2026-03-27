"""
tests/integration/thread_intrusion_test.py

Thread ownership isolation sweep for Project David.
Tests that row-level access control is correctly enforced on all
thread endpoints.

Fixtures:
  owner    — OWNER_API_KEY
  intruder — INTRUDER_API_KEY

Run directly:
    python -m tests.integration.thread_intrusion_test

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
# Clients  (module level — one instance per process)
# ──────────────────────────────────────────────────────────────────────────────
owner_client = Entity(base_url=BASE_URL, api_key=OWNER_KEY)
intruder_client = Entity(base_url=BASE_URL, api_key=INTRUDER_KEY)


# ──────────────────────────────────────────────────────────────────────────────
# Sweep
# ──────────────────────────────────────────────────────────────────────────────
def run_sweep() -> dict:
    results.clear()

    owner_thread: Optional[object] = None

    print("\n" + "═" * 60)
    print("THREAD ISOLATION SWEEP")
    print("═" * 60)

    # ── Test 1: Owner creates thread ──────────────────────────────────────────
    print("\n--- Test 1: Owner creates thread (expecting success) ---")
    owner_thread = expect_success(
        "Test 1: Owner creates thread",
        owner_client.threads.create_thread,
    )
    if owner_thread:
        print(f"  thread.id       : {owner_thread.id}")
        print(f"  thread.owner_id : {getattr(owner_thread, 'owner_id', 'N/A')}")

    # ── Test 2: Owner retrieves own thread ────────────────────────────────────
    print("\n--- Test 2: Owner retrieves own thread (expecting success) ---")
    if owner_thread:
        retrieved = expect_success(
            "Test 2: Owner retrieves own thread",
            owner_client.threads.retrieve_thread,
            owner_thread.id,
        )
        if retrieved:
            print(f"  retrieved.id : {retrieved.id}")

    # ── Test 3: Owner updates own thread metadata ─────────────────────────────
    print("\n--- Test 3: Owner updates own thread metadata (expecting success) ---")
    if owner_thread:
        expect_success(
            "Test 3: Owner updates own thread metadata",
            owner_client.threads.update_thread_metadata,
            owner_thread.id,
            {"compliance_sweep": "threads", "test": True},
        )

    # ── Test 4: Intruder updates owner's thread metadata ─────────────────────
    print("\n--- Test 4: Intruder updates owner's thread metadata (expecting 403) ---")
    if owner_thread:
        expect_error(
            "Test 4: Intruder updates owner's thread metadata",
            (403, 404),
            intruder_client.threads.update_thread_metadata,
            owner_thread.id,
            {"injected": "malicious_value"},
        )

    # ── Test 5: Intruder deletes owner's thread ───────────────────────────────
    print("\n--- Test 5: Intruder deletes owner's thread (expecting 403) ---")
    if owner_thread:
        expect_error(
            "Test 5: Intruder deletes owner's thread",
            (403, 404),
            intruder_client.threads.delete_thread,
            owner_thread.id,
        )

    # ── Test 6: Owner deletes own thread ─────────────────────────────────────
    print("\n--- Test 6: Owner deletes own thread (expecting success) ---")
    if owner_thread:
        deleted = expect_success(
            "Test 6: Owner deletes own thread",
            owner_client.threads.delete_thread,
            owner_thread.id,
        )
        if deleted:
            print(f"  deleted.id      : {deleted.id}")
            print(f"  deleted.deleted : {deleted.deleted}")

    # ── Test 7: Retrieve deleted thread returns 404 ───────────────────────────
    print("\n--- Test 7: Retrieve deleted thread (expecting 404) ---")
    if owner_thread:
        expect_error(
            "Test 7: Deleted thread returns 404 for owner",
            (404,),
            owner_client.threads.retrieve_thread,
            owner_thread.id,
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("THREAD ISOLATION SWEEP — SUMMARY")
    print("═" * 60)
    for label, outcome in results.items():
        tag = "✅" if outcome == "PASS" else "❌"
        print(f"  {tag}  {label}: {outcome}")
    print("═" * 60)

    return results


if __name__ == "__main__":
    run_sweep()
