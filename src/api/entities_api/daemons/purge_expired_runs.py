# src/api/entities_api/daemons/purge_expired_runs.py
#!/usr/bin/env python3
"""
purge_expired_runs.py
─────────────────────
Housekeeping daemon — three complementary run cleanup passes.

Purge policy
────────────
  Pass 1 — Expired  (status = 'expired')
    Runs explicitly transitioned to 'expired' by the system.
    Hard-deleted immediately — no age threshold needed.

  Pass 2 — Zombie queued  (status = 'queued', age > STUCK_RUN_MAX_AGE_HOURS)
    Runs that were never picked up by the executor.
    The executor is gone or the message was lost — they will never run.

  Pass 3 — Zombie pending_action  (status = 'pending_action', age > STUCK_RUN_MAX_AGE_HOURS)
    Runs waiting for a function-call result that never came back.
    The client is gone — they will never resolve.

  NOT touched
    completed  — kept under data retention policy
    cancelled  — kept under data retention policy (explicit user action)
    in_progress / cancelling — active or transitioning, never touched here

Key schema facts (from models.py)
──────────────────────────────────
  Run.created_at — Integer (Unix epoch seconds), NOT a DateTime column.
  Run.status     — SAEnum(StatusEnum).
  Action.run_id  — FK runs.id ondelete=CASCADE (auto-deleted with run).

Environment variables  (same .env the API uses)
────────────────────────────────────────────────
  DATABASE_URL             — resolved automatically via database.py
  STUCK_RUN_MAX_AGE_HOURS  — age threshold for zombie runs  (default: 24)
  CHECK_INTERVAL_SECONDS   — daemon polling interval        (default: 3600)
  PURGE_BATCH_SIZE         — rows per batch deletion        (default: 200)
  DRY_RUN                  — "true" to log without deleting (default: false)

Usage
─────
  # Safe dry-run (all three passes)
  DRY_RUN=true python .../purge_expired_runs.py --once --skip-wait

  # One-shot
  python .../purge_expired_runs.py --once

  # Daemon (supervisord)
  python .../purge_expired_runs.py

Supervisord stanza
──────────────────
  [program:purge_expired_runs]
  command=python /app/src/api/entities_api/daemons/purge_expired_runs.py
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
log = logging.getLogger("purge-expired-runs")

# ── Config ────────────────────────────────────────────────────────────────────

STUCK_MAX_AGE_HOURS: int = int(os.getenv("STUCK_RUN_MAX_AGE_HOURS", "24"))
CHECK_INTERVAL: int = int(os.getenv("CHECK_INTERVAL_SECONDS", "3600"))
BATCH_SIZE: int = int(os.getenv("PURGE_BATCH_SIZE", "200"))
DRY_RUN: bool = os.getenv("DRY_RUN", "false").lower() == "true"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _stuck_cutoff_epoch() -> int:
    """Unix epoch seconds for now - STUCK_MAX_AGE_HOURS."""
    return int(
        (
            datetime.now(tz=timezone.utc) - timedelta(hours=STUCK_MAX_AGE_HOURS)
        ).timestamp()
    )


def _batch_delete_runs(session, where_clause: str, params: dict, label: str) -> int:
    """
    Fetch up to BATCH_SIZE run IDs matching where_clause, delete them,
    and repeat until none remain.  Actions cascade automatically.

    Returns total runs deleted.
    """
    count_row = session.execute(
        text(f"SELECT COUNT(*) AS cnt FROM runs WHERE {where_clause}"),  # nosec B608
        params,
    ).fetchone()

    total = count_row.cnt if count_row else 0
    log.info("[%s] Qualifying runs: %d", label, total)

    if total == 0:
        log.info("[%s] Nothing to purge.", label)
        return 0

    if DRY_RUN:
        log.info("[%s][DRY_RUN] Would delete %d run(s).", label, total)
        return total

    runs_deleted = 0

    while True:
        rows = session.execute(
            text(
                f"SELECT id FROM runs WHERE {where_clause} LIMIT :batch_size"  # nosec B608
            ),  # nosec B602
            {**params, "batch_size": BATCH_SIZE},
        ).fetchall()

        if not rows:
            break

        batch_ids = [r.id for r in rows]
        id_params = {f"id_{i}": v for i, v in enumerate(batch_ids)}
        in_clause = ", ".join(f":id_{i}" for i in range(len(batch_ids)))

        try:
            result = session.execute(
                text(f"DELETE FROM runs WHERE id IN ({in_clause})"),  # nosec B608
                id_params,
            )
            session.commit()
        except Exception as exc:
            session.rollback()
            log.error("[%s] Batch failed — rolled back | %s", label, exc, exc_info=True)
            raise

        deleted_in_batch = result.rowcount
        runs_deleted += deleted_in_batch
        log.info(
            "[%s] Batch: %d  (running total: %d)",
            label,
            deleted_in_batch,
            runs_deleted,
        )

        if len(batch_ids) < BATCH_SIZE:
            break

        time.sleep(0.05)

    log.info("[%s] Complete — runs deleted: %d", label, runs_deleted)
    return runs_deleted


# ── Three passes ──────────────────────────────────────────────────────────────


def purge_explicitly_expired(session) -> int:
    """Pass 1 — status='expired', no age threshold."""
    return _batch_delete_runs(
        session,
        where_clause="status = 'expired'",
        params={},
        label="expired",
    )


def purge_zombie_queued(session) -> int:
    """Pass 2 — status='queued', stuck longer than STUCK_MAX_AGE_HOURS."""
    cutoff = _stuck_cutoff_epoch()
    log.info(
        "[queued] Cutoff: runs queued before epoch %d  (>%dh ago)",
        cutoff,
        STUCK_MAX_AGE_HOURS,
    )
    return _batch_delete_runs(
        session,
        where_clause="status = 'queued' AND created_at < :cutoff",
        params={"cutoff": cutoff},
        label="queued",
    )


def purge_zombie_pending_action(session) -> int:
    """Pass 3 — status='pending_action', stuck longer than STUCK_MAX_AGE_HOURS."""
    cutoff = _stuck_cutoff_epoch()
    log.info(
        "[pending_action] Cutoff: runs pending before epoch %d  (>%dh ago)",
        cutoff,
        STUCK_MAX_AGE_HOURS,
    )
    return _batch_delete_runs(
        session,
        where_clause="status = 'pending_action' AND created_at < :cutoff",
        params={"cutoff": cutoff},
        label="pending_action",
    )


# ── Combined cycle ────────────────────────────────────────────────────────────


def run_cycle() -> None:
    db = SessionLocal()
    try:
        log.info("── Pass 1: explicitly expired ────────────────────────────")
        expired = purge_explicitly_expired(db)

        log.info("── Pass 2: zombie queued ─────────────────────────────────")
        queued = purge_zombie_queued(db)

        log.info("── Pass 3: zombie pending_action ─────────────────────────")
        pending = purge_zombie_pending_action(db)

        log.info(
            "Cycle summary — expired: %d  |  zombie queued: %d  |  zombie pending_action: %d",
            expired,
            queued,
            pending,
        )
    finally:
        db.close()


# ── Entry points ──────────────────────────────────────────────────────────────


def run_once() -> None:
    log.info(
        "=== One-shot run purge | stuck_threshold=%dh | DRY_RUN=%s ===",
        STUCK_MAX_AGE_HOURS,
        DRY_RUN,
    )
    run_cycle()
    log.info("=== Done ===")


def run_daemon() -> None:
    log.info(
        "=== Run purge daemon started | interval=%ds | stuck_threshold=%dh | DRY_RUN=%s ===",
        CHECK_INTERVAL,
        STUCK_MAX_AGE_HOURS,
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
        description="Expired and zombie run cleanup daemon"
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
