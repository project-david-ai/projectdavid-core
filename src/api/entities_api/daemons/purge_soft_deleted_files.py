# src/api/entities_api/cli/purge_soft_deleted_files.py
#!/usr/bin/env python3
"""
purge_soft_deleted_files.py
───────────────────────────
Hard-purges File records that have been soft-deleted (deleted_at IS NOT NULL)
and whose grace period has elapsed.

Soft-delete stamps deleted_at but preserves the physical Samba bytes and the
DB row so that a restore is possible within the grace window.  This daemon
is what actually performs the final irreversible destruction once the window
has closed.

Deletion order (safest first):
  a. Remove the physical file from the Samba share
  b. Delete FileStorage row  (or let CASCADE handle it)
  c. Delete File row

Usage
  DRY_RUN=true python purge_soft_deleted_files.py --once
  python purge_soft_deleted_files.py --once
  python purge_soft_deleted_files.py          # daemon

Environment variables:
  SHARED_PATH              — host path mounted as Samba share root (default: ./shared_data)
  CHECK_INTERVAL_SECONDS   — daemon polling interval in seconds    (default: 300)
  SOFT_DELETE_GRACE_HOURS  — hours after deleted_at before purge   (default: 48)
  DRY_RUN                  — "true" to log without deleting        (default: false)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import text

load_dotenv()

from entities_api.db.database import SessionLocal, wait_for_databases

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("soft-delete-file-purge")

# ─── Config ───────────────────────────────────────────────────────────────────

SHARED_PATH: str = os.getenv("SHARED_PATH", "./shared_data")
CHECK_INTERVAL: int = int(os.getenv("CHECK_INTERVAL_SECONDS", "300"))
GRACE_HOURS: int = int(os.getenv("SOFT_DELETE_GRACE_HOURS", "48"))
DRY_RUN: bool = os.getenv("DRY_RUN", "false").lower() == "true"

# ─── Helpers ──────────────────────────────────────────────────────────────────


def samba_path(storage_path: str) -> Path:
    return Path(SHARED_PATH) / storage_path


def delete_physical_file(path: Path) -> bool:
    if not path.exists():
        log.debug("Physical file already absent: %s", path)
        return True
    if DRY_RUN:
        log.info("[DRY_RUN] Would delete physical file: %s", path)
        return True
    try:
        path.unlink()
        log.debug("Deleted: %s", path)
        return True
    except OSError as exc:
        log.error("Failed to delete %s: %s", path, exc)
        return False


# ─── Core logic ───────────────────────────────────────────────────────────────


def purge_soft_deleted(session) -> tuple[int, int]:
    """
    Find all File rows that have been soft-deleted and whose grace period has
    elapsed.  Destroy the physical bytes then hard-delete the DB records.

    deleted_at is stored as a Unix integer timestamp (mirrors Assistant.deleted_at
    and the new File.deleted_at column).

    Returns (files_attempted, files_deleted).
    """
    grace_cutoff = int((datetime.utcnow() - timedelta(hours=GRACE_HOURS)).timestamp())

    rows = session.execute(
        text(
            """
            SELECT f.id             AS file_id,
                   f.filename       AS filename,
                   f.deleted_at     AS deleted_at,
                   fs.storage_path  AS storage_path
            FROM   files f
            JOIN   file_storage fs ON fs.file_id = f.id
            WHERE  f.deleted_at IS NOT NULL
              AND  f.deleted_at < :cutoff
            """
        ),
        {"cutoff": grace_cutoff},
    ).fetchall()

    if not rows:
        log.info("No soft-deleted files past grace period.")
        return 0, 0

    log.info(
        "Found %d soft-deleted file(s) past the %dh grace period.",
        len(rows),
        GRACE_HOURS,
    )

    attempted = 0
    deleted = 0

    for row in rows:
        file_id = row.file_id
        filename = row.filename
        storage_path = row.storage_path
        deleted_at = datetime.utcfromtimestamp(row.deleted_at)

        log.info(
            "Purging | id=%s | name=%s | deleted_at=%s",
            file_id,
            filename,
            deleted_at,
        )
        attempted += 1

        # 1. Physical bytes first — safe even if DB step later fails
        phys_ok = delete_physical_file(samba_path(storage_path))

        # 2. DB hard-delete (FileStorage CASCADE-deletes with File)
        if not DRY_RUN:
            try:
                session.execute(
                    text("DELETE FROM file_storage WHERE file_id = :fid"),
                    {"fid": file_id},
                )
                session.execute(
                    text("DELETE FROM files WHERE id = :fid"),
                    {"fid": file_id},
                )
                session.commit()
                log.info("DB records removed | file_id=%s", file_id)
                if phys_ok:
                    deleted += 1
            except Exception as exc:
                session.rollback()
                log.error("DB deletion failed | file_id=%s | %s", file_id, exc)
        else:
            log.info("[DRY_RUN] Would remove DB records for file_id=%s", file_id)
            deleted += 1

    return attempted, deleted


# ─── Entry points ─────────────────────────────────────────────────────────────


def run_once() -> None:
    log.info(
        "=== One-shot soft-delete purge | grace=%dh | SHARED_PATH=%s | DRY_RUN=%s ===",
        GRACE_HOURS,
        SHARED_PATH,
        DRY_RUN,
    )
    db = SessionLocal()
    try:
        attempted, deleted = purge_soft_deleted(db)
    finally:
        db.close()
    log.info("Done. %d/%d files hard-purged.", deleted, attempted)


def run_daemon() -> None:
    log.info(
        "=== Soft-delete purge daemon started | interval=%ds | grace=%dh | DRY_RUN=%s ===",
        CHECK_INTERVAL,
        GRACE_HOURS,
        DRY_RUN,
    )
    while True:
        try:
            db = SessionLocal()
            try:
                attempted, deleted = purge_soft_deleted(db)
            finally:
                db.close()
            log.info("Cycle complete. %d/%d files hard-purged.", deleted, attempted)
        except Exception as exc:
            log.error("Unexpected error during purge cycle: %s", exc, exc_info=True)
        log.info("Sleeping %ds until next cycle…", CHECK_INTERVAL)
        time.sleep(CHECK_INTERVAL)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Soft-deleted file hard-purge utility")
    parser.add_argument("--once", action="store_true", help="Single pass then exit")
    parser.add_argument(
        "--skip-wait", action="store_true", help="Skip DB readiness check"
    )
    args = parser.parse_args()

    if not args.skip_wait:
        wait_for_databases()

    if args.once:
        run_once()
    else:
        run_daemon()
