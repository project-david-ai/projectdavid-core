# src/api/entities_api/cli/purge_expired_files.py
#!/usr/bin/env python3
"""
purge_expired_files.py
──────────────────────
Deletes files whose signed-URL window has passed.

Strategy
  1. On startup, immediately purge anything already expired.
  2. Loop indefinitely, checking every CHECK_INTERVAL_SECONDS.

Deletion order (safest first):
  a. Remove the physical file from the Samba share (filesystem)
  b. Delete FileStorage row
  c. Delete File row

Usage
  # Safe dry-run — logs what would be deleted, touches nothing
  DRY_RUN=true python purge_expired_files.py --once

  # One-shot (cron / CI)
  python purge_expired_files.py --once

  # Daemon (background service / Docker sidecar)
  python purge_expired_files.py

Environment variables (same .env your API uses):
  DATABASE_URL            — resolved automatically via database.py
  SHARED_PATH             — host path mounted as /samba/share (default: ./shared_data)
  CHECK_INTERVAL_SECONDS  — daemon polling interval in seconds   (default: 300)
  DRY_RUN                 — "true" to log without deleting       (default: false)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import text

load_dotenv()

# ── Reuse the centralised engine / session factory ─────────────────────────
# database.py already handles the running_in_docker() URL resolution so
# 'db' vs 'localhost:3307' is sorted for us automatically.
from entities_api.db.database import SessionLocal, wait_for_databases

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("file-cleanup")

# ─── Config ───────────────────────────────────────────────────────────────────

SHARED_PATH: str = os.getenv("SHARED_PATH", "./shared_data")
CHECK_INTERVAL: int = int(os.getenv("CHECK_INTERVAL_SECONDS", "300"))
DRY_RUN: bool = os.getenv("DRY_RUN", "false").lower() == "true"

# ─── Helpers ──────────────────────────────────────────────────────────────────


def samba_path(storage_path: str) -> Path:
    """
    Resolve a FileStorage.storage_path (relative to the Samba share root)
    to an absolute host path.

    Example:  "file_abc123_report.pdf"  →  ./shared_data/file_abc123_report.pdf
    """
    return Path(SHARED_PATH) / storage_path


def delete_physical_file(path: Path) -> bool:
    """Remove a file from the filesystem. Returns True on success."""
    if not path.exists():
        log.debug("Physical file already absent: %s", path)
        return True  # already gone — that's fine
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


def purge_expired(session) -> tuple[int, int]:
    """
    Find all File rows whose expires_at has passed, delete the physical files,
    then remove the DB rows. Excludes training datasets and models.
    """
    now = datetime.utcnow()

    rows = session.execute(
        text(
            """
            SELECT f.id             AS file_id,
                   f.filename       AS filename,
                   f.expires_at     AS expires_at,
                   f.purpose        AS purpose,
                   fs.storage_path  AS storage_path
            FROM   files f
            JOIN   file_storage fs ON fs.file_id = f.id
            WHERE  f.expires_at IS NOT NULL
              AND  f.expires_at < :now
            """
        ),
        {"now": now},
    ).fetchall()

    if not rows:
        log.info("No expired files found.")
        return 0, 0

    log.info("Found %d expired file(s) to process.", len(rows))

    attempted = 0
    deleted = 0

    for row in rows:
        file_id = row.file_id
        filename = row.filename
        storage_path = row.storage_path
        expires_at = row.expires_at
        purpose = row.purpose

        # ─── PROTECTION GUARD ──────────────────────────────────────────
        # Skip files explicitly marked for training or located in the models dir
        if purpose == "training" or "models/" in (storage_path or ""):
            log.info(
                "Skipping protected file | id=%s | purpose=%s | path=%s",
                file_id,
                purpose,
                storage_path,
            )
            continue
        # ───────────────────────────────────────────────────────────────

        log.info(
            "Processing | id=%s | name=%s | expired=%s",
            file_id,
            filename,
            expires_at,
        )
        attempted += 1

        # 1. Physical deletion first (safe even if DB step fails)
        phys_ok = delete_physical_file(samba_path(storage_path))

        # 2. DB deletion
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
    log.info("=== One-shot cleanup | SHARED_PATH=%s | DRY_RUN=%s ===", SHARED_PATH, DRY_RUN)
    db = SessionLocal()
    try:
        attempted, deleted = purge_expired(db)
    finally:
        db.close()
    log.info("Done. %d/%d files cleaned up.", deleted, attempted)


def run_daemon() -> None:
    log.info(
        "=== Cleanup daemon started | interval=%ds | SHARED_PATH=%s | DRY_RUN=%s ===",
        CHECK_INTERVAL,
        SHARED_PATH,
        DRY_RUN,
    )
    while True:
        try:
            db = SessionLocal()
            try:
                attempted, deleted = purge_expired(db)
            finally:
                db.close()
            log.info("Cycle complete. %d/%d files cleaned up.", deleted, attempted)
        except Exception as exc:
            log.error("Unexpected error during cleanup cycle: %s", exc, exc_info=True)
        log.info("Sleeping %d seconds until next cycle…", CHECK_INTERVAL)
        time.sleep(CHECK_INTERVAL)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Expired file cleanup utility")
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
        # Reuse the same retry/wait logic the API uses on startup
        wait_for_databases()

    if args.once:
        run_once()
    else:
        run_daemon()
