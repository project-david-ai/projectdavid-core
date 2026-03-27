# src/api/entities_api/daemons/purge_orphaned_threads.py
#!/usr/bin/env python3
"""
purge_orphaned_threads.py
─────────────────────────
GDPR / housekeeping daemon — two complementary thread cleanup passes.

Pass 1 — Orphaned threads  (GDPR)
  Threads whose owner was erased: owner_id IS NULL.
  Messages must be deleted manually (no FK cascade on Message.thread_id).
  Retained for ORPHAN_THREAD_RETENTION_DAYS before deletion (default: 30).

Pass 2 — Abandoned threads  (housekeeping)
  Threads whose owner still exists but that were never used:
    owner_id IS NOT NULL
    AND no messages
    AND no runs
  Retained for ABANDONED_THREAD_RETENTION_DAYS before deletion (default: 7).
  thread_participants rows cascade automatically (FK ondelete=CASCADE).
  No messages to delete by definition.

Key schema facts (from models.py)
──────────────────────────────────
  Thread.created_at  — Integer (Unix epoch seconds), NOT a DateTime column.
  Message.thread_id  — plain String, no FK constraint, no DB-level cascade.
  Run.thread_id      — plain String, no FK constraint.
  thread_participants.thread_id — FK with ondelete=CASCADE (auto-deleted).

Deletion order
──────────────
  Orphaned  : DELETE messages → DELETE threads
  Abandoned : DELETE threads only (participants cascade; no messages by definition)

Environment variables  (same .env the API uses)
────────────────────────────────────────────────
  DATABASE_URL                    — resolved automatically via database.py
  ORPHAN_THREAD_RETENTION_DAYS    — orphan retention in days      (default: 30)
  ABANDONED_THREAD_RETENTION_DAYS — abandoned retention in days   (default: 7)
  CHECK_INTERVAL_SECONDS          — daemon polling interval        (default: 3600)
  PURGE_BATCH_SIZE                — rows per batch deletion        (default: 200)
  DRY_RUN                         — "true" to log without deleting (default: false)

Usage
─────
  # Safe dry-run (both passes)
  DRY_RUN=true python .../purge_orphaned_threads.py --once --skip-wait

  # One-shot
  python .../purge_orphaned_threads.py --once

  # Daemon (supervisord)
  python .../purge_orphaned_threads.py

Supervisord stanza
──────────────────
  [program:purge_orphaned_threads]
  command=python /app/src/api/entities_api/daemons/purge_orphaned_threads.py
  directory=/app
  autostart=true
  autorestart=true
  startretries=5
  startsecs=10
  stdout_logfile=/dev/stdout
  stdout_logfile_maxbytes=0
  stderr_logfile=/dev/stderr
  stderr_logfile_maxbytes=0
  environment=PYTHONPATH="/app/src/api"
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from sqlalchemy import text

load_dotenv()

# ── Reuse the centralised engine / session factory ────────────────────────────
from entities_api.db.database import SessionLocal, wait_for_databases

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("purge-orphaned-threads")

# ── Config ────────────────────────────────────────────────────────────────────

ORPHAN_RETENTION_DAYS: int = int(os.getenv("ORPHAN_THREAD_RETENTION_DAYS", "30"))
ABANDONED_RETENTION_DAYS: int = int(os.getenv("ABANDONED_THREAD_RETENTION_DAYS", "7"))
CHECK_INTERVAL: int = int(os.getenv("CHECK_INTERVAL_SECONDS", "3600"))
BATCH_SIZE: int = int(os.getenv("PURGE_BATCH_SIZE", "200"))
DRY_RUN: bool = os.getenv("DRY_RUN", "false").lower() == "true"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _cutoff_epoch(retention_days: int) -> int:
    return int(
        (datetime.now(tz=timezone.utc) - timedelta(days=retention_days)).timestamp()
    )


def _batch_delete(session, batch_ids: list[str], label: str) -> tuple[int, int]:
    """
    Delete one batch of threads (and their messages if orphaned).

    Returns (threads_deleted, messages_deleted).
    """
    id_params = {f"id_{i}": v for i, v in enumerate(batch_ids)}
    in_clause = ", ".join(f":id_{i}" for i in range(len(batch_ids)))

    msg_count = 0

    if label == "orphaned":
        # Message.thread_id has no FK cascade — must delete manually.
        msg_result = session.execute(
            text(
                f"DELETE FROM messages WHERE thread_id IN ({in_clause})"  # nosec B608
            ),  # nosec B608
            id_params,
        )
        msg_count = msg_result.rowcount

    thr_result = session.execute(
        text(f"DELETE FROM threads WHERE id IN ({in_clause})"),  # nosec B608
        id_params,
    )
    # thread_participants cascades automatically on thread deletion.

    session.commit()
    return thr_result.rowcount, msg_count


# ── Pass 1: Orphaned threads (owner erased → owner_id IS NULL) ────────────────


def purge_orphaned_threads(session) -> tuple[int, int]:
    """
    Delete threads with owner_id IS NULL older than ORPHAN_RETENTION_DAYS,
    plus their messages (no FK cascade on Message.thread_id).

    Returns (threads_deleted, messages_deleted).
    """
    cutoff = _cutoff_epoch(ORPHAN_RETENTION_DAYS)

    count_row = session.execute(
        text(
            """
            SELECT COUNT(*) AS cnt
            FROM   threads
            WHERE  owner_id   IS NULL
              AND  created_at  < :cutoff
            """
        ),
        {"cutoff": cutoff},
    ).fetchone()

    total = count_row.cnt if count_row else 0
    log.info(
        "[orphaned] Threads with owner_id=NULL older than %d day(s): %d  (cutoff epoch=%d)",
        ORPHAN_RETENTION_DAYS,
        total,
        cutoff,
    )

    if total == 0:
        log.info("[orphaned] Nothing to purge.")
        return 0, 0

    if DRY_RUN:
        log.info(
            "[orphaned][DRY_RUN] Would delete %d thread(s) and their messages.", total
        )
        return total, 0

    threads_deleted = messages_deleted = 0

    while True:
        rows = session.execute(
            text(
                """
                SELECT id FROM threads
                WHERE  owner_id   IS NULL
                  AND  created_at  < :cutoff
                LIMIT  :batch_size
                """
            ),
            {"cutoff": cutoff, "batch_size": BATCH_SIZE},
        ).fetchall()

        if not rows:
            break

        batch_ids = [r.id for r in rows]
        try:
            t, m = _batch_delete(session, batch_ids, "orphaned")
        except Exception as exc:
            session.rollback()
            log.error("[orphaned] Batch failed — rolled back | %s", exc, exc_info=True)
            raise

        threads_deleted += t
        messages_deleted += m
        log.info(
            "[orphaned] Batch: threads=%d  messages=%d  (running: %d / %d)",
            t,
            m,
            threads_deleted,
            messages_deleted,
        )

        if len(batch_ids) < BATCH_SIZE:
            break
        time.sleep(0.05)

    log.info(
        "[orphaned] Complete — threads: %d  messages: %d",
        threads_deleted,
        messages_deleted,
    )
    return threads_deleted, messages_deleted


# ── Pass 2: Abandoned threads (owner exists, never used) ─────────────────────


def purge_abandoned_threads(session) -> int:
    """
    Delete threads that have a live owner but were never used:
      - owner_id IS NOT NULL
      - no rows in messages for this thread_id
      - no rows in runs    for this thread_id
      - created_at older than ABANDONED_RETENTION_DAYS

    thread_participants rows cascade automatically (FK ondelete=CASCADE).
    There are no messages to delete by definition.

    Returns threads_deleted.
    """
    cutoff = _cutoff_epoch(ABANDONED_RETENTION_DAYS)

    count_row = session.execute(
        text(
            """
            SELECT COUNT(*) AS cnt
            FROM   threads t
            WHERE  t.owner_id   IS NOT NULL
              AND  t.created_at  < :cutoff
              AND  NOT EXISTS (
                       SELECT 1 FROM messages m WHERE m.thread_id = t.id
                   )
              AND  NOT EXISTS (
                       SELECT 1 FROM runs r WHERE r.thread_id = t.id
                   )
            """
        ),
        {"cutoff": cutoff},
    ).fetchone()

    total = count_row.cnt if count_row else 0
    log.info(
        "[abandoned] Threads with no messages/runs older than %d day(s): %d  (cutoff epoch=%d)",
        ABANDONED_RETENTION_DAYS,
        total,
        cutoff,
    )

    if total == 0:
        log.info("[abandoned] Nothing to purge.")
        return 0

    if DRY_RUN:
        log.info("[abandoned][DRY_RUN] Would delete %d abandoned thread(s).", total)
        return total

    threads_deleted = 0

    while True:
        rows = session.execute(
            text(
                """
                SELECT t.id
                FROM   threads t
                WHERE  t.owner_id   IS NOT NULL
                  AND  t.created_at  < :cutoff
                  AND  NOT EXISTS (
                           SELECT 1 FROM messages m WHERE m.thread_id = t.id
                       )
                  AND  NOT EXISTS (
                           SELECT 1 FROM runs r WHERE r.thread_id = t.id
                       )
                LIMIT  :batch_size
                """
            ),
            {"cutoff": cutoff, "batch_size": BATCH_SIZE},
        ).fetchall()

        if not rows:
            break

        batch_ids = [r.id for r in rows]
        id_params = {f"id_{i}": v for i, v in enumerate(batch_ids)}
        in_clause = ", ".join(f":id_{i}" for i in range(len(batch_ids)))

        try:
            result = session.execute(
                text(f"DELETE FROM threads WHERE id IN ({in_clause})"),  # nosec B608
                id_params,
            )
            # thread_participants cascade automatically — no manual step needed.
            session.commit()
        except Exception as exc:
            session.rollback()
            log.error("[abandoned] Batch failed — rolled back | %s", exc, exc_info=True)
            raise

        t = result.rowcount
        threads_deleted += t
        log.info(
            "[abandoned] Batch: threads=%d  (running total: %d)",
            t,
            threads_deleted,
        )

        if len(batch_ids) < BATCH_SIZE:
            break
        time.sleep(0.05)

    log.info("[abandoned] Complete — threads deleted: %d", threads_deleted)
    return threads_deleted


# ── Combined cycle ────────────────────────────────────────────────────────────


def run_cycle() -> None:
    """Run both passes inside a single DB session lifecycle."""
    db = SessionLocal()
    try:
        log.info("── Pass 1: orphaned threads ──────────────────────────────")
        orphan_t, orphan_m = purge_orphaned_threads(db)

        log.info("── Pass 2: abandoned threads ─────────────────────────────")
        abandoned_t = purge_abandoned_threads(db)

        log.info(
            "Cycle summary — orphaned: %d threads / %d messages  |  abandoned: %d threads",
            orphan_t,
            orphan_m,
            abandoned_t,
        )
    finally:
        db.close()


# ── Entry points ──────────────────────────────────────────────────────────────


def run_once() -> None:
    log.info(
        "=== One-shot thread purge"
        " | orphan_retention=%dd | abandoned_retention=%dd | DRY_RUN=%s ===",
        ORPHAN_RETENTION_DAYS,
        ABANDONED_RETENTION_DAYS,
        DRY_RUN,
    )
    run_cycle()
    log.info("=== Done ===")


def run_daemon() -> None:
    log.info(
        "=== Thread purge daemon started"
        " | interval=%ds | orphan_retention=%dd | abandoned_retention=%dd | DRY_RUN=%s ===",
        CHECK_INTERVAL,
        ORPHAN_RETENTION_DAYS,
        ABANDONED_RETENTION_DAYS,
        DRY_RUN,
    )
    while True:
        try:
            run_cycle()
        except Exception as exc:
            log.error("Unexpected error during purge cycle: %s", exc, exc_info=True)
        log.info("Sleeping %d seconds until next cycle…", CHECK_INTERVAL)
        time.sleep(CHECK_INTERVAL)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Orphaned + abandoned thread cleanup daemon"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single pass then exit (default: run as daemon)",
    )
    parser.add_argument(
        "--skip-wait",
        action="store_true",
        help="Skip the DB readiness check (useful if DB is already confirmed up)",
    )
    args = parser.parse_args()

    if not args.skip_wait:
        wait_for_databases()

    if args.once:
        run_once()
    else:
        run_daemon()
