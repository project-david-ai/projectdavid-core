"""
tests/integration/db_integrity_check.py

White-box database integrity checker for Project David.
Connects directly to the database (no API required) and verifies that
the data state itself is clean — no orphans, no ownership violations,
no rows that escaped the access-control guards.

This is the independent verification layer. It proves not just that the
API enforces ownership at request time, but that the database contains
no historical violations from before the guards were in place.

Usage:
    python -m tests.integration.db_integrity_check

Required env vars:
    DATABASE_URL  — e.g. mysql+pymysql://user:pass@localhost:3306/entities_db
"""

import os
import sys
import time

from dotenv import load_dotenv

# Bootstrap the app path so we can import from src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.api.entities_api.db.database import SessionLocal
from src.api.entities_api.models.models import (
    Action,
    ApiKey,
    Assistant,
    File,
    FileStorage,
    Message,
    Run,
    Thread,
    User,
    VectorStore,
    VectorStoreFile,
)

load_dotenv()
WIDTH = 68
results: dict = {}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def banner(text: str) -> None:
    print(f"\n{'═' * WIDTH}")
    print(f"  {text}")
    print(f"{'═' * WIDTH}")


def section(text: str) -> None:
    print(f"\n{'─' * WIDTH}")
    print(f"  {text}")
    print(f"{'─' * WIDTH}")


def record(label: str, violations: list, note: str = "") -> None:
    passed = len(violations) == 0
    tag = "✅" if passed else "❌"
    status = "PASS" if passed else f"FAIL ({len(violations)} violation(s))"
    print(f"  {tag}  {label}: {status}" + (f" — {note}" if note else ""))
    if violations:
        for v in violations[:5]:  # show first 5 to avoid flooding
            print(f"       → {v}")
        if len(violations) > 5:
            print(f"       → ... and {len(violations) - 5} more")
    results[label] = "PASS" if passed else "FAIL"


# ──────────────────────────────────────────────────────────────────────────────
# Checks
# ──────────────────────────────────────────────────────────────────────────────


def check_files_have_user(db) -> None:
    """Every file must have a user_id."""
    rows = db.query(File.id).filter(File.user_id.is_(None)).all()
    record(
        "Files: no NULL user_id",
        [r.id for r in rows],
    )


def check_files_user_exists(db) -> None:
    """Every file's user_id must reference a real user."""
    rows = (
        db.query(File.id, File.user_id)
        .outerjoin(User, User.id == File.user_id)
        .filter(User.id.is_(None))
        .all()
    )
    record(
        "Files: user_id references valid user",
        [f"{r.id} → {r.user_id}" for r in rows],
    )


def check_runs_have_user(db) -> None:
    """Every run should have a user_id (nullable by schema but a gap if missing)."""
    rows = db.query(Run.id).filter(Run.user_id.is_(None)).all()
    record(
        "Runs: no NULL user_id",
        [r.id for r in rows],
        note="nullable by schema — backfill needed if any found",
    )


def check_runs_user_exists(db) -> None:
    """Every run's user_id must reference a real user."""
    rows = (
        db.query(Run.id, Run.user_id)
        .filter(Run.user_id.isnot(None))
        .outerjoin(User, User.id == Run.user_id)
        .filter(User.id.is_(None))
        .all()
    )
    record(
        "Runs: user_id references valid user",
        [f"{r.id} → {r.user_id}" for r in rows],
    )


def check_runs_thread_consistency(db) -> None:
    """
    Each run's thread_id must reference an existing thread.
    (Orphaned runs whose thread was hard-deleted are a data integrity gap.)
    """
    rows = (
        db.query(Run.id, Run.thread_id)
        .outerjoin(Thread, Thread.id == Run.thread_id)
        .filter(Thread.id.is_(None))
        .all()
    )
    record(
        "Runs: thread_id references existing thread",
        [f"{r.id} → {r.thread_id}" for r in rows],
    )


def check_runs_assistant_consistency(db) -> None:
    """Each run's assistant_id must reference an existing (non-deleted) assistant."""
    rows = (
        db.query(Run.id, Run.assistant_id)
        .outerjoin(Assistant, Assistant.id == Run.assistant_id)
        .filter(Assistant.id.is_(None))
        .all()
    )
    record(
        "Runs: assistant_id references existing assistant",
        [f"{r.id} → {r.assistant_id}" for r in rows],
    )


def check_threads_owner_exists(db) -> None:
    """If a thread has an owner_id it must reference a real user."""
    rows = (
        db.query(Thread.id, Thread.owner_id)
        .filter(Thread.owner_id.isnot(None))
        .outerjoin(User, User.id == Thread.owner_id)
        .filter(User.id.is_(None))
        .all()
    )
    record(
        "Threads: owner_id references valid user",
        [f"{r.id} → {r.owner_id}" for r in rows],
    )


def check_assistants_owner_exists(db) -> None:
    """If an assistant has an owner_id it must reference a real user."""
    rows = (
        db.query(Assistant.id, Assistant.owner_id)
        .filter(Assistant.owner_id.isnot(None))
        .outerjoin(User, User.id == Assistant.owner_id)
        .filter(User.id.is_(None))
        .all()
    )
    record(
        "Assistants: owner_id references valid user",
        [f"{r.id} → {r.owner_id}" for r in rows],
    )


def check_vector_stores_have_user(db) -> None:
    """Every vector store must have a user_id."""
    rows = db.query(VectorStore.id).filter(VectorStore.user_id.is_(None)).all()
    record(
        "VectorStores: no NULL user_id",
        [r.id for r in rows],
    )


def check_vector_stores_user_exists(db) -> None:
    """Every vector store's user_id must reference a real user."""
    rows = (
        db.query(VectorStore.id, VectorStore.user_id)
        .outerjoin(User, User.id == VectorStore.user_id)
        .filter(User.id.is_(None))
        .all()
    )
    record(
        "VectorStores: user_id references valid user",
        [f"{r.id} → {r.user_id}" for r in rows],
    )


def check_vector_store_files_parent_exists(db) -> None:
    """Every VectorStoreFile must reference an existing VectorStore."""
    rows = (
        db.query(VectorStoreFile.id, VectorStoreFile.vector_store_id)
        .outerjoin(VectorStore, VectorStore.id == VectorStoreFile.vector_store_id)
        .filter(VectorStore.id.is_(None))
        .all()
    )
    record(
        "VectorStoreFiles: parent vector_store exists",
        [f"{r.id} → {r.vector_store_id}" for r in rows],
    )


def check_file_storage_parent_exists(db) -> None:
    """Every FileStorage record must reference an existing File."""
    rows = (
        db.query(FileStorage.id, FileStorage.file_id)
        .outerjoin(File, File.id == FileStorage.file_id)
        .filter(File.id.is_(None))
        .all()
    )
    record(
        "FileStorage: parent file exists",
        [f"storage_id={r.id} → file_id={r.file_id}" for r in rows],
    )


def check_messages_thread_exists(db) -> None:
    """Every message must reference an existing thread."""
    rows = (
        db.query(Message.id, Message.thread_id)
        .outerjoin(Thread, Thread.id == Message.thread_id)
        .filter(Thread.id.is_(None))
        .all()
    )
    record(
        "Messages: thread_id references existing thread",
        [f"{r.id} → {r.thread_id}" for r in rows],
    )


def check_actions_run_exists(db) -> None:
    """Every action with a run_id must reference an existing run."""
    rows = (
        db.query(Action.id, Action.run_id)
        .filter(Action.run_id.isnot(None))
        .outerjoin(Run, Run.id == Action.run_id)
        .filter(Run.id.is_(None))
        .all()
    )
    record(
        "Actions: run_id references existing run",
        [f"{r.id} → {r.run_id}" for r in rows],
    )


def check_api_keys_user_exists(db) -> None:
    """Every API key must reference an existing user."""
    rows = (
        db.query(ApiKey.id, ApiKey.user_id)
        .outerjoin(User, User.id == ApiKey.user_id)
        .filter(User.id.is_(None))
        .all()
    )
    record(
        "ApiKeys: user_id references valid user",
        [f"key_id={r.id} → user_id={r.user_id}" for r in rows],
    )


def check_cross_user_run_thread(db) -> None:
    """
    Cross-ownership check: a run's user_id should match the thread's owner_id
    when the thread has one set. Mismatches indicate a historical access gap.
    """
    rows = (
        db.query(Run.id, Run.user_id, Thread.owner_id)
        .join(Thread, Thread.id == Run.thread_id)
        .filter(
            Run.user_id.isnot(None),
            Thread.owner_id.isnot(None),
            Run.user_id != Thread.owner_id,
        )
        .all()
    )
    record(
        "Cross-check: run.user_id matches thread.owner_id",
        [f"run={r.id} run.user={r.user_id} thread.owner={r.owner_id}" for r in rows],
        note="mismatches may indicate pre-guard data or shared threads",
    )


def check_duplicate_active_api_keys(db) -> None:
    """
    Sanity check: no user should have more than a reasonable number of
    active API keys (threshold: 20). Outliers may indicate a key leak.
    """
    from sqlalchemy import func

    rows = (
        db.query(ApiKey.user_id, func.count(ApiKey.id).label("cnt"))
        .filter(ApiKey.is_active.is_(True))  # <--- FIXED: Changed from == True
        .group_by(ApiKey.user_id)
        .having(func.count(ApiKey.id) > 20)
        .all()
    )
    record(
        "ApiKeys: no user has excessive active keys (>20)",
        [f"user_id={r.user_id} active_keys={r.cnt}" for r in rows],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def run_checks() -> dict:
    """Run all checks and return results dict. Callable from compliance_runner."""
    banner("PROJECT DAVID — DB INTEGRITY CHECK")
    print(f"  DATABASE_URL : {os.getenv('DATABASE_URL', '(not set)')[:60]}...")
    print(f"  Time         : {time.strftime('%Y-%m-%d %H:%M:%S')}")

    if not os.getenv("DATABASE_URL"):
        print("\n[ABORT] DATABASE_URL is not set.")
        sys.exit(2)

    with SessionLocal() as db:

        section("Files")
        check_files_have_user(db)
        check_files_user_exists(db)

        section("Runs")
        check_runs_have_user(db)
        check_runs_user_exists(db)
        check_runs_thread_consistency(db)
        check_runs_assistant_consistency(db)

        section("Threads")
        check_threads_owner_exists(db)

        section("Assistants")
        check_assistants_owner_exists(db)

        section("Vector Stores")
        check_vector_stores_have_user(db)
        check_vector_stores_user_exists(db)
        check_vector_store_files_parent_exists(db)

        section("File Storage")
        check_file_storage_parent_exists(db)

        section("Messages")
        check_messages_thread_exists(db)

        section("Actions")
        check_actions_run_exists(db)

        section("API Keys")
        check_api_keys_user_exists(db)
        check_duplicate_active_api_keys(db)

        section("Cross-Ownership Checks")
        check_cross_user_run_thread(db)

    return results


def main() -> int:
    check_results = run_checks()

    failed = sum(1 for v in check_results.values() if v == "FAIL")
    passed = sum(1 for v in check_results.values() if v == "PASS")

    banner("DB INTEGRITY CHECK — SUMMARY")
    for label, outcome in check_results.items():
        tag = "✅" if outcome == "PASS" else "❌"
        print(f"  {tag}  {label}: {outcome}")

    print(f"\n  Passed  : {passed}")
    print(f"  Failed  : {failed}")
    print(f"\n{'═' * WIDTH}")

    if failed:
        print("  ❌  INTEGRITY STATUS: VIOLATIONS FOUND")
        print(f"{'═' * WIDTH}\n")
        return 1

    print("  ✅  INTEGRITY STATUS: CLEAN")
    print(f"{'═' * WIDTH}\n")
    return 0


# Expose run_sweep alias so compliance_runner can call it uniformly
run_sweep = run_checks

if __name__ == "__main__":
    sys.exit(main())
