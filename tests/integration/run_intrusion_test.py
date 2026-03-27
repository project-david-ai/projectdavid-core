"""
tests/integration/run_intrusion_test.py

Run ownership isolation sweep for Project David.
Tests that row-level access control is correctly enforced on all
run endpoints.

Fixtures:
  owner    — OWNER_API_KEY
  intruder — INTRUDER_API_KEY

Run directly:
    python -m tests.integration.run_intrusion_test

Or via the compliance runner (exposes run_sweep()).
"""

import os

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

    owner_assistant = None
    owner_thread = None
    owner_run = None

    print("\n" + "═" * 60)
    print("RUN ISOLATION SWEEP")
    print("═" * 60)

    # ── Setup: create fixtures ────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  SETUP")
    print("─" * 60)

    owner_assistant = expect_success(
        "Setup: Owner creates assistant",
        owner_client.assistants.create_assistant,
        name="Run Isolation Test Assistant",
        model="gpt-oss-120b",
        instructions="You are a test assistant for run isolation verification.",
    )

    owner_thread = expect_success(
        "Setup: Owner creates thread",
        owner_client.threads.create_thread,
    )

    if not owner_assistant or not owner_thread:
        record("Setup", False, "Fixtures could not be created — aborting sweep")
        return results

    print(f"  assistant.id : {owner_assistant.id}")
    print(f"  thread.id    : {owner_thread.id}")

    # ── Test 1: Owner creates run ─────────────────────────────────────────────
    print("\n--- Test 1: Owner creates run (expecting success) ---")
    owner_run = expect_success(
        "Test 1: Owner creates run",
        owner_client.runs.create_run,
        thread_id=owner_thread.id,
        assistant_id=owner_assistant.id,
    )
    if owner_run:
        print(f"  run.id      : {owner_run.id}")
        print(f"  run.user_id : {owner_run.user_id}")
        print(f"  run.status  : {owner_run.status}")

    # ── Test 2: Intruder creates run against owner's assistant ────────────────
    print(
        "\n--- Test 2: Intruder creates run against owner's assistant (expecting 403) ---"
    )
    intruder_thread = expect_success(
        "Test 2 setup: Intruder creates own thread",
        intruder_client.threads.create_thread,
    )
    if intruder_thread:
        expect_error(
            "Test 2: Intruder creates run against owner's assistant",
            (403, 404),
            intruder_client.runs.create_run,
            thread_id=intruder_thread.id,
            assistant_id=owner_assistant.id,
        )
        try:
            intruder_client.threads.delete_thread(intruder_thread.id)
        except Exception:
            pass

    # ── Test 3: Owner retrieves own run ───────────────────────────────────────
    print("\n--- Test 3: Owner retrieves own run (expecting success) ---")
    if owner_run:
        retrieved = expect_success(
            "Test 3: Owner retrieves own run",
            owner_client.runs.retrieve_run,
            owner_run.id,
        )
        if retrieved:
            print(f"  retrieved.id      : {retrieved.id}")
            print(f"  retrieved.user_id : {retrieved.user_id}")

    # ── Test 4: Intruder retrieves owner's run ────────────────────────────────
    print("\n--- Test 4: Intruder retrieves owner's run (expecting 403) ---")
    if owner_run:
        expect_error(
            "Test 4: Intruder retrieves owner's run",
            (403, 404),
            intruder_client.runs.retrieve_run,
            owner_run.id,
        )

    # ── Test 5: Intruder cancels owner's run ──────────────────────────────────
    print("\n--- Test 5: Intruder cancels owner's run (expecting 403) ---")
    if owner_run:
        expect_error(
            "Test 5: Intruder cancels owner's run",
            (403, 404),
            intruder_client.runs.cancel_run,
            owner_run.id,
        )

    # ── Test 6: Owner cancels own run ─────────────────────────────────────────
    print("\n--- Test 6: Owner cancels own run (expecting success) ---")
    if owner_run:
        cancelled = expect_success(
            "Test 6: Owner cancels own run",
            owner_client.runs.cancel_run,
            owner_run.id,
        )
        if cancelled:
            cancelled_id = (
                cancelled.get("id") if isinstance(cancelled, dict) else cancelled.id
            )
            cancelled_status = (
                cancelled.get("status")
                if isinstance(cancelled, dict)
                else cancelled.status
            )
            print(f"  cancelled.id     : {cancelled_id}")
            print(f"  cancelled.status : {cancelled_status}")

    # ── Test 7: Owner can retrieve own run; list has no cross-user leakage ─────
    # NOTE: list_all_runs() filters to active (non-terminal) runs only, which
    # excludes queued and cancelled runs.  Verifying positive ownership via the
    # list is therefore unreliable — we use retrieve_run instead, which is the
    # correct API for that assertion.  The list is still exercised to confirm
    # no foreign runs leak into the owner's view.
    print("\n--- Test 7: Owner lists runs (expecting only own runs) ---")

    # 7a — ownership: owner can retrieve the run created in Test 1 by ID
    if owner_run:
        try:
            fetched = owner_client.runs.retrieve_run(owner_run.id)
            record(
                "Test 7: Owner's run retrievable by owner",
                fetched is not None and fetched.id == owner_run.id,
            )
            if fetched:
                print(f"  fetched.id      : {fetched.id}")
                print(f"  fetched.user_id : {fetched.user_id}")
        except Exception as exc:
            record("Test 7: Owner's run retrievable by owner", False, str(exc))

    # 7b — isolation: list_all_runs must not surface foreign user data
    try:
        run_list = owner_client.runs.list_all_runs()
        owner_user_id = owner_run.user_id if owner_run else None
        if owner_user_id:
            leaked = [r for r in run_list.data if r.user_id != owner_user_id]
            record(
                "Test 7: No cross-user leakage in run list",
                len(leaked) == 0,
                f"{len(leaked)} foreign run(s) leaked" if leaked else "",
            )
        else:
            record(
                "Test 7: No cross-user leakage in run list",
                True,
                "no owner_user_id to compare",
            )
    except Exception as exc:
        record("Test 7: No cross-user leakage in run list", False, str(exc))

    # ── Test 8: Intruder lists runs — must not see owner's run ───────────────
    print("\n--- Test 8: Intruder lists runs (must not see owner's run) ---")
    try:
        intruder_list = intruder_client.runs.list_all_runs()
        intruder_run_ids = [r.id for r in intruder_list.data]
        leaked = owner_run and owner_run.id in intruder_run_ids
        record(
            "Test 8: Owner's run not visible to intruder",
            not leaked,
            "owner's run leaked into intruder list!" if leaked else "",
        )
    except Exception as exc:
        record("Test 8: Owner's run not visible to intruder", False, str(exc))

    # ── Teardown ──────────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  TEARDOWN")
    print("─" * 60)

    if owner_thread:
        try:
            owner_client.threads.delete_thread(owner_thread.id)
            print(f"[OK] Thread deleted: {owner_thread.id}")
        except Exception as exc:
            print(f"[WARN] Could not delete thread: {exc}")

    if owner_assistant:
        try:
            owner_client.assistants.delete_assistant(owner_assistant.id)
            print(f"[OK] Assistant deleted: {owner_assistant.id}")
        except Exception as exc:
            print(f"[WARN] Could not delete assistant: {exc}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("RUN ISOLATION SWEEP — SUMMARY")
    print("═" * 60)
    for label, outcome in results.items():
        tag = "✅" if outcome == "PASS" else "❌"
        print(f"  {tag}  {label}: {outcome}")
    print("═" * 60)

    return results


if __name__ == "__main__":
    run_sweep()
