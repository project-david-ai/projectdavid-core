"""add BatfishSnapshot to models

Revision ID: b50938f6bd99
Revises: ce0a8a7e9d41
Create Date: 2026-02-27 03:24:20.687063
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

from migrations.utils.safe_ddl import (create_fk_if_not_exists,
                                       drop_fk_if_exists, has_table,
                                       safe_alter_column)

# revision identifiers, used by Alembic.
revision: str = "b50938f6bd99"
down_revision: Union[str, None] = "ce0a8a7e9d41"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema safely."""

    # --- Table: batfish_snapshots ---
    # The FK to users is intentionally NOT declared inline in create_table.
    # MySQL InnoDB enforces FK references at CREATE TABLE time — if users doesn't
    # exist yet in this session (fresh container), the statement fails with
    # ER_NO_REFERENCED_ROW (1824). The FK is added separately below via
    # create_fk_if_not_exists, which checks has_table("users") before acting.
    if not has_table("batfish_snapshots"):
        op.create_table(
            "batfish_snapshots",
            sa.Column(
                "id",
                sa.String(length=64),
                nullable=False,
                comment="Shared ID — also used as the Batfish network name",
            ),
            sa.Column(
                "snapshot_name",
                sa.String(length=128),
                nullable=False,
                comment="Caller-supplied incident/tenant label",
            ),
            sa.Column(
                "snapshot_key",
                sa.String(length=256),
                nullable=False,
                comment="Namespaced isolation key: {user_id}_{snapshot_name}",
            ),
            sa.Column("user_id", sa.String(length=64), nullable=False),
            sa.Column(
                "configs_root",
                sa.String(length=512),
                nullable=True,
                comment="Server-side path used during last ingest",
            ),
            sa.Column("device_count", sa.Integer(), nullable=False),
            sa.Column(
                "devices",
                sa.JSON(),
                nullable=False,
                comment="List of hostnames ingested into this snapshot",
            ),
            sa.Column(
                "status",
                sa.Enum(
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
                    name="statusenum",
                ),
                nullable=False,
                comment="pending | loading | ready | failed | deleted",
            ),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.Column("created_at", sa.BigInteger(), nullable=False),
            sa.Column("updated_at", sa.BigInteger(), nullable=False),
            sa.Column(
                "last_ingested_at",
                sa.BigInteger(),
                nullable=True,
                comment="Unix timestamp of last successful config ingest",
            ),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "user_id",
                "snapshot_name",
                name="uq_batfish_user_snapshot",
            ),
        )

        op.create_index("idx_batfish_status", "batfish_snapshots", ["status"], unique=False)
        op.create_index("idx_batfish_user_id", "batfish_snapshots", ["user_id"], unique=False)
        op.create_index(op.f("ix_batfish_snapshots_id"), "batfish_snapshots", ["id"], unique=False)
        op.create_index(
            op.f("ix_batfish_snapshots_snapshot_key"),
            "batfish_snapshots",
            ["snapshot_key"],
            unique=True,
        )
        op.create_index(
            op.f("ix_batfish_snapshots_user_id"), "batfish_snapshots", ["user_id"], unique=False
        )

    # Add FK to users as a separate deferred step — safe on fresh containers
    # where users may not exist yet, and idempotent on existing databases.
    create_fk_if_not_exists(
        "batfish_snapshots_ibfk_1",
        "batfish_snapshots",
        "users",
        ["user_id"],
        ["id"],
        ondelete="CASCADE",
    )

    # --- Table: messages ---
    safe_alter_column("messages", "content", existing_type=mysql.TEXT(), nullable=False)
    safe_alter_column(
        "messages",
        "reasoning",
        existing_type=mysql.LONGTEXT(),
        type_=sa.Text(length=4294967295),
        nullable=True,
    )


def downgrade() -> None:
    """Downgrade schema safely."""

    # --- Table: messages ---
    safe_alter_column(
        "messages",
        "reasoning",
        existing_type=sa.Text(length=4294967295),
        type_=mysql.LONGTEXT(),
        nullable=True,
    )
    safe_alter_column("messages", "content", existing_type=mysql.TEXT(), nullable=True)

    # --- Table: batfish_snapshots ---
    # Drop FK before indexes and table — MySQL requires this ordering.
    # drop_fk_if_exists checks has_table internally.
    drop_fk_if_exists("batfish_snapshots", "batfish_snapshots_ibfk_1")

    if has_table("batfish_snapshots"):
        op.drop_index(op.f("ix_batfish_snapshots_user_id"), table_name="batfish_snapshots")
        op.drop_index(op.f("ix_batfish_snapshots_snapshot_key"), table_name="batfish_snapshots")
        op.drop_index(op.f("ix_batfish_snapshots_id"), table_name="batfish_snapshots")
        op.drop_index("idx_batfish_user_id", table_name="batfish_snapshots")
        op.drop_index("idx_batfish_status", table_name="batfish_snapshots")
        op.drop_table("batfish_snapshots")
