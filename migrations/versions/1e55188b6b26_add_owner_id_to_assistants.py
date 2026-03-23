"""add owner_id to assistants

Revision ID: 1e55188b6b26
Revises: b50938f6bd99
Create Date: 2026-03-06 05:25:54.596685

Changes
-------
- assistants       : add owner_id (FK → users.id, SET NULL on delete)
- batfish_snapshots: comment-only column updates + rename unique constraint
- messages         : tighten content to NOT NULL; update reasoning type/comment
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

from migrations.utils.safe_ddl import (
    add_column_if_missing,
    create_fk_if_not_exists,
    create_index_if_missing,
    drop_column_if_exists,
    drop_fk_if_exists,
    drop_index_if_exists,
    has_table,
    safe_alter_column,
)

# --- Revision identifiers ---
revision: str = "1e55188b6b26"
down_revision: Union[str, None] = "b50938f6bd99"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _constraint_exists(constraint_name: str, table_name: str) -> bool:
    """
    Return True if a named constraint already exists on the given table.
    Returns False immediately if the table itself does not exist — prevents
    op.drop_constraint / op.create_unique_constraint from being called on a
    missing table in fresh-container scenarios.
    Used for unique constraints, which safe_ddl does not yet cover.
    """
    if not has_table(table_name):
        return False

    bind = op.get_bind()
    result = bind.execute(
        sa.text(
            """
            SELECT COUNT(*) FROM information_schema.TABLE_CONSTRAINTS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME   = :table
              AND CONSTRAINT_NAME = :name
            """
        ),
        {"table": table_name, "name": constraint_name},
    )
    return result.scalar() > 0


# ---------------------------------------------------------------------------
# Upgrade
# ---------------------------------------------------------------------------


def upgrade() -> None:

    # ── assistants: owner_id ────────────────────────────────────────────────
    add_column_if_missing(
        "assistants",
        sa.Column(
            "owner_id",
            sa.String(64),
            nullable=True,
            comment=(
                "Canonical owner of this assistant. "
                "Primary key for row-level access filtering. "
                "Separate from the many-to-many (user_assistants) which handles sharing."
            ),
        ),
    )

    # Safe index + FK creation via helpers — guard against missing table,
    # duplicate index, and duplicate constraint
    create_index_if_missing("ix_assistants_owner_id", "assistants", ["owner_id"])

    create_fk_if_not_exists(
        "fk_assistants_owner_id_users",
        "assistants",
        "users",
        ["owner_id"],
        ["id"],
        ondelete="SET NULL",
    )

    # ── batfish_snapshots: comment-only column updates ──────────────────────
    # All safe_alter_column calls check has_table internally — safe on fresh containers.
    safe_alter_column(
        "batfish_snapshots",
        "id",
        existing_type=mysql.VARCHAR(length=64),
        comment="Opaque snapshot ID returned to caller e.g. snap_abc123",
        existing_nullable=False,
    )
    safe_alter_column(
        "batfish_snapshots",
        "snapshot_name",
        existing_type=mysql.VARCHAR(length=128),
        comment="Caller-supplied label e.g. 'incident_001'",
        existing_nullable=False,
    )
    safe_alter_column(
        "batfish_snapshots",
        "snapshot_key",
        existing_type=mysql.VARCHAR(length=256),
        comment="Namespaced isolation key: {user_id}_{id}",
        existing_nullable=False,
    )
    safe_alter_column(
        "batfish_snapshots",
        "configs_root",
        existing_type=mysql.VARCHAR(length=512),
        comment=None,
        existing_nullable=True,
    )
    safe_alter_column(
        "batfish_snapshots",
        "status",
        existing_type=mysql.ENUM(
            "deleted",
            "active",
            "queued",
            "in_progress",
            "pending_action",
            "completed",
            "failed",
            "cancelling",
            "cancelled",
            "pending",
            "processing",
            "expired",
            "retrying",
        ),
        comment=None,
        existing_nullable=False,
    )
    safe_alter_column(
        "batfish_snapshots",
        "last_ingested_at",
        existing_type=mysql.BIGINT(),
        comment=None,
        existing_nullable=True,
    )

    # ── batfish_snapshots: rename unique constraint ─────────────────────────
    # _constraint_exists now returns False when the table is absent, so both
    # branches below are safe on a fresh container (neither op will fire).
    if _constraint_exists("uq_batfish_user_snapshot", "batfish_snapshots"):
        op.drop_constraint("uq_batfish_user_snapshot", "batfish_snapshots", type_="unique")

    if not _constraint_exists("uq_batfish_user_snapshot_name", "batfish_snapshots"):
        # Only attempt to create if the table actually exists; the has_table
        # check inside _constraint_exists already gates this when the table is
        # missing (it returns False, making `not False` = True, which would
        # incorrectly enter this branch on a missing table).
        if has_table("batfish_snapshots"):
            op.create_unique_constraint(
                "uq_batfish_user_snapshot_name",
                "batfish_snapshots",
                ["user_id", "snapshot_name"],
            )

    # ── messages ────────────────────────────────────────────────────────────
    safe_alter_column(
        "messages",
        "content",
        existing_type=mysql.TEXT(),
        nullable=False,
    )
    safe_alter_column(
        "messages",
        "reasoning",
        existing_type=mysql.LONGTEXT(),
        type_=sa.Text(length=4294967295),
        comment="Stores the internal 'thinking' or reasoning tokens from the model.",
        existing_nullable=True,
    )


# ---------------------------------------------------------------------------
# Downgrade
# ---------------------------------------------------------------------------


def downgrade() -> None:

    # ── messages ────────────────────────────────────────────────────────────
    safe_alter_column(
        "messages",
        "reasoning",
        existing_type=sa.Text(length=4294967295),
        type_=mysql.LONGTEXT(),
        comment=None,
        existing_nullable=True,
    )
    safe_alter_column(
        "messages",
        "content",
        existing_type=mysql.TEXT(),
        nullable=True,
    )

    # ── batfish_snapshots: restore unique constraint name ───────────────────
    if _constraint_exists("uq_batfish_user_snapshot_name", "batfish_snapshots"):
        op.drop_constraint("uq_batfish_user_snapshot_name", "batfish_snapshots", type_="unique")

    if not _constraint_exists("uq_batfish_user_snapshot", "batfish_snapshots"):
        if has_table("batfish_snapshots"):
            op.create_index(
                "uq_batfish_user_snapshot",
                "batfish_snapshots",
                ["user_id", "snapshot_name"],
                unique=True,
            )

    # ── batfish_snapshots: restore original comments ────────────────────────
    safe_alter_column(
        "batfish_snapshots",
        "last_ingested_at",
        existing_type=mysql.BIGINT(),
        comment="Unix timestamp of last successful config ingest",
        existing_nullable=True,
    )
    safe_alter_column(
        "batfish_snapshots",
        "status",
        existing_type=mysql.ENUM(
            "deleted",
            "active",
            "queued",
            "in_progress",
            "pending_action",
            "completed",
            "failed",
            "cancelling",
            "cancelled",
            "pending",
            "processing",
            "expired",
            "retrying",
        ),
        comment="pending | loading | ready | failed | deleted",
        existing_nullable=False,
    )
    safe_alter_column(
        "batfish_snapshots",
        "configs_root",
        existing_type=mysql.VARCHAR(length=512),
        comment="Server-side path used during last ingest",
        existing_nullable=True,
    )
    safe_alter_column(
        "batfish_snapshots",
        "snapshot_key",
        existing_type=mysql.VARCHAR(length=256),
        comment="Namespaced isolation key: {user_id}_{snapshot_name}",
        existing_nullable=False,
    )
    safe_alter_column(
        "batfish_snapshots",
        "snapshot_name",
        existing_type=mysql.VARCHAR(length=128),
        comment="Caller-supplied incident/tenant label",
        existing_nullable=False,
    )
    safe_alter_column(
        "batfish_snapshots",
        "id",
        existing_type=mysql.VARCHAR(length=64),
        comment="Shared ID — also used as the Batfish network name",
        existing_nullable=False,
    )

    # ── assistants: owner_id ────────────────────────────────────────────────
    drop_fk_if_exists("assistants", "fk_assistants_owner_id_users")
    drop_index_if_exists("ix_assistants_owner_id", "assistants")
    drop_column_if_exists("assistants", "owner_id")
