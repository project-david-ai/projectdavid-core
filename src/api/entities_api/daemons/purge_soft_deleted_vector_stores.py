# src/api/entities_api/cli/purge_soft_deleted_vector_stores.py
#!/usr/bin/env python3
"""
purge_soft_deleted_vector_stores.py
────────────────────────────────────
Hard-purges VectorStore records that have been soft-deleted (deleted_at IS NOT NULL)
and whose grace period has elapsed.

Soft-delete stamps deleted_at and sets status=deleted but intentionally preserves
the Qdrant collection so that a restore is possible within the grace window.  This
daemon performs the final irreversible destruction once that window closes.

Deletion order (safest first):
  a. Destroy the Qdrant collection  (vector data)
  b. Hard-delete VectorStoreFile rows
  c. Hard-delete VectorStore row

Usage
  DRY_RUN=true python purge_soft_deleted_vector_stores.py --once
  python purge_soft_deleted_vector_stores.py --once
  python purge_soft_deleted_vector_stores.py          # daemon

Environment variables:
  QDRANT_HOST              — Qdrant service host              (default: localhost)
  QDRANT_PORT              — Qdrant service port              (default: 6333)
  CHECK_INTERVAL_SECONDS   — daemon polling interval seconds  (default: 300)
  SOFT_DELETE_GRACE_HOURS  — hours after deleted_at before purge (default: 48)
  DRY_RUN                  — "true" to log without deleting   (default: false)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta

from dotenv import load_dotenv

load_dotenv()
from entities_api.db.database import SessionLocal, wait_for_databases
from sqlalchemy import text

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("soft-delete-vs-purge")

# ─── Config ───────────────────────────────────────────────────────────────────

QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
CHECK_INTERVAL: int = int(os.getenv("CHECK_INTERVAL_SECONDS", "300"))
GRACE_HOURS: int = int(os.getenv("SOFT_DELETE_GRACE_HOURS", "48"))
DRY_RUN: bool = os.getenv("DRY_RUN", "false").lower() == "true"

# ─── Qdrant helper ────────────────────────────────────────────────────────────


def delete_qdrant_collection(collection_name: str) -> bool:
    """
    Destroy a Qdrant collection.  Returns True on success or if already absent.
    Uses the REST API directly to avoid pulling in the full VectorStoreManager
    dependency chain into a lightweight CLI script.
    """
    import httpx

    url = f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{collection_name}"
    if DRY_RUN:
        log.info("[DRY_RUN] Would DELETE Qdrant collection: %s", collection_name)
        return True
    try:
        resp = httpx.delete(url, timeout=10.0)
        if resp.status_code in (200, 404):
            # 404 = already gone, that's fine
            log.debug("Qdrant collection removed (or absent): %s", collection_name)
            return True
        log.error(
            "Unexpected Qdrant response deleting %s: %d %s",
            collection_name,
            resp.status_code,
            resp.text,
        )
        return False
    except Exception as exc:
        log.error("Failed to delete Qdrant collection %s: %s", collection_name, exc)
        return False


# ─── Core logic ───────────────────────────────────────────────────────────────


def purge_soft_deleted(session) -> tuple[int, int]:
    """
    Find VectorStore rows that are soft-deleted and past the grace period.
    Destroy the Qdrant collection, then hard-delete the DB records.

    Returns (stores_attempted, stores_deleted).
    """
    grace_cutoff = int((datetime.utcnow() - timedelta(hours=GRACE_HOURS)).timestamp())

    rows = session.execute(
        text(
            """
            SELECT id               AS store_id,
                   name             AS name,
                   collection_name  AS collection_name,
                   deleted_at       AS deleted_at
            FROM   vector_stores
            WHERE  deleted_at IS NOT NULL
              AND  deleted_at < :cutoff
            """
        ),
        {"cutoff": grace_cutoff},
    ).fetchall()

    if not rows:
        log.info("No soft-deleted vector stores past grace period.")
        return 0, 0

    log.info(
        "Found %d soft-deleted vector store(s) past the %dh grace period.",
        len(rows),
        GRACE_HOURS,
    )

    attempted = 0
    deleted = 0

    for row in rows:
        store_id = row.store_id
        name = row.name
        collection_name = row.collection_name
        deleted_at = datetime.utcfromtimestamp(row.deleted_at)

        log.info(
            "Purging | id=%s | name=%s | collection=%s | deleted_at=%s",
            store_id,
            name,
            collection_name,
            deleted_at,
        )
        attempted += 1

        # 1. Qdrant collection first — safe even if DB step later fails
        qdrant_ok = delete_qdrant_collection(collection_name)

        # 2. DB hard-delete (VectorStoreFile CASCADE-deletes with VectorStore)
        if not DRY_RUN:
            try:
                session.execute(
                    text("DELETE FROM vector_store_files WHERE vector_store_id = :sid"),
                    {"sid": store_id},
                )
                session.execute(
                    text("DELETE FROM vector_stores WHERE id = :sid"),
                    {"sid": store_id},
                )
                session.commit()
                log.info("DB records removed | store_id=%s", store_id)
                if qdrant_ok:
                    deleted += 1
            except Exception as exc:
                session.rollback()
                log.error("DB deletion failed | store_id=%s | %s", store_id, exc)
        else:
            log.info("[DRY_RUN] Would remove DB records for store_id=%s", store_id)
            deleted += 1

    return attempted, deleted


# ─── Entry points ─────────────────────────────────────────────────────────────


def run_once() -> None:
    log.info(
        "=== One-shot VS purge | grace=%dh | qdrant=%s:%d | DRY_RUN=%s ===",
        GRACE_HOURS,
        QDRANT_HOST,
        QDRANT_PORT,
        DRY_RUN,
    )
    db = SessionLocal()
    try:
        attempted, deleted = purge_soft_deleted(db)
    finally:
        db.close()
    log.info("Done. %d/%d vector stores hard-purged.", deleted, attempted)


def run_daemon() -> None:
    log.info(
        "=== VS soft-delete purge daemon | interval=%ds | grace=%dh | DRY_RUN=%s ===",
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
            log.info(
                "Cycle complete. %d/%d vector stores hard-purged.", deleted, attempted
            )
        except Exception as exc:
            log.error("Unexpected error during purge cycle: %s", exc, exc_info=True)
        log.info("Sleeping %ds until next cycle…", CHECK_INTERVAL)
        time.sleep(CHECK_INTERVAL)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Soft-deleted vector store hard-purge utility"
    )
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
