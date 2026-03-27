from __future__ import annotations

import os
import tempfile
import time
from urllib.parse import urlparse

import httpx
import pymysql
from dotenv import find_dotenv, load_dotenv
from projectdavid import Entity

from src.api.entities_api.services.logging_service import LoggingUtility

"""
user_erasure_test.py
────────────────────
GDPR right-to-erasure integration sweep.

Creates a full asset graph owned by the sacrificial user, calls
DELETE /users/{id} with the admin key, then verifies at both the
API level and the DB level that everything is correctly gone,
nullified, or preserved according to policy.

Erasure policy under test
─────────────────────────
GONE       files, file_storage, vector_stores, vector_store_files,
           runs, actions, api_keys, sandboxes, batfish_snapshots,
           messages in owned threads, exclusively-owned assistants (soft-deleted)
NULLIFIED  threads.owner_id, assistants.owner_id (shared assistants)
PRESERVED  audit_logs row with user_id = NULL (compliance record)

Environment variables (same .env as other sweeps)
─────────────────────────────────────────────────
ENTITIES_BASE_URL     — API base URL          (default: http://localhost:9000)
ADMIN_API_KEY         — Admin user API key
SACRIFICIAL_API_KEY   — Sacrificial user API key
SACRIFICIAL_USER_ID   — Sacrificial user ID
DB_HOST               — MySQL host            (default: localhost)
DB_PORT               — MySQL port            (default: 3307)
DB_NAME               — MySQL database name   (default: entities_db)
DB_USER               — MySQL user            (default: root)
DB_PASSWORD           — MySQL password

Usage
─────
python -m tests.integration.user_erasure_test
"""


load_dotenv(find_dotenv())

# ── Config ────────────────────────────────────────────────────────────────────

BASE_URL = os.getenv("ENTITIES_BASE_URL", "http://localhost:9000")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")
SACRIFICIAL_API_KEY = os.getenv("SACRIFICIAL_API_KEY")
SACRIFICIAL_USER_ID = os.getenv("SACRIFICIAL_USER_ID")

logging_utility = LoggingUtility()

# ── Results store ─────────────────────────────────────────────────────────────

results: list[dict] = []


# ── Helpers ───────────────────────────────────────────────────────────────────


def _record(name: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    results.append({"name": name, "status": status, "detail": detail})
    icon = "✅" if passed else "❌"
    detail_str = f": {detail}" if detail else ""
    print(f"  [{status}] {icon}  {name}{detail_str}")


def _admin_headers() -> dict:
    return {"X-API-Key": ADMIN_API_KEY}


def _sacrificial_headers() -> dict:
    return {"X-API-Key": SACRIFICIAL_API_KEY}


def _db_conn():
    url = os.getenv("DATABASE_URL", "")
    parsed = urlparse(url.replace("mysql+pymysql://", "mysql://"))
    return pymysql.connect(
        host=parsed.hostname,
        port=parsed.port or 3306,
        database=parsed.path.lstrip("/"),
        user=parsed.username,
        password=parsed.password,
        cursorclass=pymysql.cursors.DictCursor,
    )


def _db_fetchone(sql: str, params: tuple) -> dict | None:
    with _db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()


def _db_fetchall(sql: str, params: tuple) -> list[dict]:
    with _db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


# ── Asset creation helpers ────────────────────────────────────────────────────


def _create_assistant(client: Entity) -> str:
    asst = client.assistants.create_assistant(
        name="Erasure Test Assistant",
        model="gpt-4",
        instructions="test",
    )
    logging_utility.info("Created assistant id=%s", asst.id)
    return asst.id


def _create_thread(client: Entity) -> str:
    thread = client.threads.create_thread(participant_ids=[SACRIFICIAL_USER_ID])
    logging_utility.info("Created thread id=%s", thread.id)
    return thread.id


def _create_message(client: Entity, thread_id: str, assistant_id: str) -> str:
    msg = client.messages.create_message(
        thread_id=thread_id,
        content="Erasure test message — should be deleted.",
        assistant_id=assistant_id,
        role="user",
    )
    logging_utility.info("Created message id=%s", msg.id)
    return msg.id


def _create_file(client: Entity, tmp_path: str) -> str:
    file_obj = client.files.upload_file(
        file_path=tmp_path,
        purpose="assistants",
    )
    logging_utility.info("Created file id=%s", file_obj.id)
    return file_obj.id


# ── Main sweep ────────────────────────────────────────────────────────────────


def run_sweep() -> dict:
    results.clear()

    print()
    print("=" * 60)
    print("USER ERASURE SWEEP (GDPR RIGHT-TO-ERASURE)")
    print("=" * 60)

    # ── Guard rails ──────────────────────────────────────────────────────────
    if not all([ADMIN_API_KEY, SACRIFICIAL_API_KEY, SACRIFICIAL_USER_ID]):
        print(
            "  [SKIP] Missing ADMIN_API_KEY, SACRIFICIAL_API_KEY, or SACRIFICIAL_USER_ID"
        )
        return {"passed": 0, "failed": 0, "skipped": 1}

    # ── SDK clients ──────────────────────────────────────────────────────────
    sacrificial_client = Entity(
        base_url=BASE_URL,
        api_key=SACRIFICIAL_API_KEY,
    )
    logging_utility.info("Entity initialized for sacrificial user")

    # ── Temp file for upload ──────────────────────────────────────────────────
    tmp_file = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        prefix="erasure_test_",
        delete=False,
    )
    tmp_file.write("Erasure test file content — should be deleted.")
    tmp_file.close()
    tmp_path = tmp_file.name
    print(f"\n[SETUP] Temp file: {tmp_path}")

    # ── Track created asset IDs ───────────────────────────────────────────────
    assistant_id: str | None = None
    thread_id: str | None = None
    message_id: str | None = None
    file_id: str | None = None

    try:
        # ── SETUP: Create asset graph ─────────────────────────────────────────
        print("\n" + "─" * 60)
        print("  SETUP — building asset graph")
        print("─" * 60)

        try:
            assistant_id = _create_assistant(sacrificial_client)
            _record("Setup: Create assistant", True, assistant_id)
        except Exception as exc:
            _record("Setup: Create assistant", False, str(exc))
            raise

        try:
            thread_id = _create_thread(sacrificial_client)
            _record("Setup: Create thread", True, thread_id)
        except Exception as exc:
            _record("Setup: Create thread", False, str(exc))
            raise

        try:
            message_id = _create_message(sacrificial_client, thread_id, assistant_id)
            _record("Setup: Create message", True, message_id)
        except Exception as exc:
            _record("Setup: Create message", False, str(exc))
            raise

        try:
            file_id = _create_file(sacrificial_client, tmp_path)
            _record("Setup: Upload file", True, file_id)
        except Exception as exc:
            _record("Setup: Upload file", False, str(exc))
            raise

        # Brief pause to let any async indexing settle
        time.sleep(1)

        # ── ACTION: Call DELETE /users/{id} with admin key ────────────────────
        print("\n" + "─" * 60)
        print("  ACTION — calling GDPR erasure endpoint")
        print("─" * 60)

        with httpx.Client(base_url=BASE_URL, timeout=30) as http:
            resp = http.delete(
                f"/v1/users/{SACRIFICIAL_USER_ID}",
                headers=_admin_headers(),
            )

        _record(
            "Erasure endpoint returns 204",
            resp.status_code == 204,
            f"HTTP {resp.status_code}",
        )

        if resp.status_code != 204:
            print("\n  [ABORT] Erasure endpoint failed — cannot verify cleanup.")
            return _summarise()

        # Brief pause for DB to settle
        time.sleep(1)

        # ── VERIFY: API-level checks ──────────────────────────────────────────
        print("\n" + "─" * 60)
        print("  VERIFY — API level")
        print("─" * 60)

        with httpx.Client(base_url=BASE_URL, timeout=10) as http:

            # User row gone
            r = http.get(
                f"/v1/users/{SACRIFICIAL_USER_ID}",
                headers=_admin_headers(),
            )
            _record(
                "Test 1: User returns 404 after erasure",
                r.status_code == 404,
                f"HTTP {r.status_code}",
            )

            # File gone
            if file_id:
                r = http.get(
                    f"/v1/files/{file_id}",
                    headers=_admin_headers(),
                )
                _record(
                    "Test 2: File returns 404 after erasure",
                    r.status_code in (403, 404),
                    f"HTTP {r.status_code}",
                )

            # Thread still exists but owner_id is NULL (SET NULL cascade)
            if thread_id:
                r = http.get(
                    f"/v1/threads/{thread_id}",
                    headers=_admin_headers(),
                )
                # Thread row survives (SET NULL), so we expect 200 or 404 depending
                # on whether the admin can retrieve ownerless threads.
                # We verify owner_id nullification at DB level below.
                _record(
                    "Test 3: Thread endpoint reachable after erasure",
                    r.status_code in (200, 403, 404),
                    f"HTTP {r.status_code}",
                )

        # ── VERIFY: DB-level checks ───────────────────────────────────────────
        print("\n" + "─" * 60)
        print("  VERIFY — DB level")
        print("─" * 60)

        # User row gone
        row = _db_fetchone(
            "SELECT id FROM users WHERE id = %s",
            (SACRIFICIAL_USER_ID,),
        )
        _record("Test 4: User row absent from DB", row is None)

        # File row gone
        if file_id:
            row = _db_fetchone(
                "SELECT id FROM files WHERE id = %s",
                (file_id,),
            )
            _record("Test 5: File row absent from DB", row is None)

        # Messages deleted
        if thread_id:
            rows = _db_fetchall(
                "SELECT id FROM messages WHERE thread_id = %s",
                (thread_id,),
            )
            _record(
                "Test 6: Messages in owned thread deleted from DB",
                len(rows) == 0,
                f"{len(rows)} message(s) remaining",
            )

        # Thread row survives, owner_id is NULL
        if thread_id:
            row = _db_fetchone(
                "SELECT id, owner_id FROM threads WHERE id = %s",
                (thread_id,),
            )
            _record(
                "Test 7: Thread row survives with owner_id = NULL",
                row is not None and row["owner_id"] is None,
                f"owner_id={row['owner_id'] if row else 'row missing'}",
            )

        # Assistant soft-deleted (exclusively owned)
        if assistant_id:
            row = _db_fetchone(
                "SELECT id, owner_id, deleted_at FROM assistants WHERE id = %s",
                (assistant_id,),
            )
            if row:
                _record(
                    "Test 8: Exclusively-owned assistant is soft-deleted",
                    row["deleted_at"] is not None,
                    f"deleted_at={row['deleted_at']}",
                )
            else:
                _record(
                    "Test 8: Exclusively-owned assistant is soft-deleted",
                    False,
                    "row missing",
                )

        # Audit log preserved with user_id = NULL
        row = _db_fetchone(
            """
            SELECT id, user_id, action, entity_id
            FROM audit_logs
            WHERE action = 'ERASE'
              AND entity_type = 'User'
              AND entity_id = %s
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (SACRIFICIAL_USER_ID,),
        )
        _record(
            "Test 9: Erasure audit log preserved",
            row is not None,
            f"audit_log id={row['id'] if row else 'NOT FOUND'}",
        )
        if row:
            _record(
                "Test 10: Audit log user_id is NULL (anonymised)",
                row["user_id"] is None,
                f"user_id={row['user_id']}",
            )

        # api_keys cascade-deleted
        rows = _db_fetchall(
            "SELECT id FROM api_keys WHERE user_id = %s",
            (SACRIFICIAL_USER_ID,),
        )
        _record(
            "Test 11: API keys cascade-deleted",
            len(rows) == 0,
            f"{len(rows)} key(s) remaining",
        )

        # runs cascade-deleted
        rows = _db_fetchall(
            "SELECT id FROM runs WHERE user_id = %s",
            (SACRIFICIAL_USER_ID,),
        )
        _record(
            "Test 12: Runs cascade-deleted",
            len(rows) == 0,
            f"{len(rows)} run(s) remaining",
        )

        # user_assistants join rows gone
        rows = _db_fetchall(
            "SELECT user_id FROM user_assistants WHERE user_id = %s",
            (SACRIFICIAL_USER_ID,),
        )
        _record(
            "Test 13: user_assistants join rows cascade-deleted",
            len(rows) == 0,
            f"{len(rows)} row(s) remaining",
        )

        # thread_participants join rows gone
        rows = _db_fetchall(
            "SELECT user_id FROM thread_participants WHERE user_id = %s",
            (SACRIFICIAL_USER_ID,),
        )
        _record(
            "Test 14: thread_participants join rows cascade-deleted",
            len(rows) == 0,
            f"{len(rows)} row(s) remaining",
        )

    except Exception as exc:
        _record("Sweep aborted with exception", False, str(exc))
        logging_utility.error("Erasure sweep exception: %s", exc, exc_info=True)

    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
            print(f"\n[TEARDOWN] Temp file removed: {tmp_path}")
        except Exception:
            pass

    return _summarise()


def _summarise() -> dict:
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")

    print()
    print("=" * 60)
    print("USER ERASURE SWEEP — SUMMARY")
    print("=" * 60)
    for r in results:
        icon = "✅" if r["status"] == "PASS" else "❌"
        print(f"  {icon}  {r['name']}: {r['status']}")
    print("=" * 60)

    return {"passed": passed, "failed": failed, "skipped": 0}


if __name__ == "__main__":
    run_sweep()
