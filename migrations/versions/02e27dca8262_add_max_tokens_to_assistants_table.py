"""Add max_tokens to assistants table

Revision ID: 02e27dca8262
Revises: db67202be996
Create Date: 2026-03-29 03:05:02.355939

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

from migrations.utils.safe_ddl import (
    add_column_if_missing,
    drop_fk_if_exists,
    drop_index_if_exists,
    drop_table_if_exists,
    has_table,
    safe_alter_column,
)

# revision identifiers, used by Alembic.
revision: str = "02e27dca8262"
down_revision: Union[str, None] = "db67202be996"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# ---------------------------------------------------------------------------
# Shared ENUM definition
# ---------------------------------------------------------------------------
_STATUS_ENUM = mysql.ENUM(
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
    "offline",
)


def upgrade() -> None:
    """Upgrade schema."""

    # -------------------------------------------------------------------------
    # 1. Remove batfish_snapshots
    #    Order matters: FK → indexes → table
    # -------------------------------------------------------------------------
    if has_table("batfish_snapshots"):
        drop_fk_if_exists("batfish_snapshots", "batfish_snapshots_ibfk_1")
        for idx in [
            "uq_batfish_user_snapshot_name",
            "idx_batfish_status",
            "idx_batfish_user_id",
            "ix_batfish_snapshots_id",
            "ix_batfish_snapshots_snapshot_key",
            "ix_batfish_snapshots_user_id",
        ]:
            drop_index_if_exists(idx, "batfish_snapshots")
        drop_table_if_exists("batfish_snapshots")

    # -------------------------------------------------------------------------
    # 2. api_keys
    # -------------------------------------------------------------------------
    safe_alter_column(
        "api_keys",
        "is_active",
        existing_type=mysql.TINYINT(display_width=1),
        nullable=False,
    )

    # -------------------------------------------------------------------------
    # 3. assistants
    # -------------------------------------------------------------------------
    add_column_if_missing(
        "assistants",
        sa.Column(
            "max_tokens",
            sa.Integer(),
            server_default="2048",
            nullable=True,
            comment="Maximum tokens to generate per inference pass. Overrides provider defaults at runtime.",
        ),
    )
    safe_alter_column(
        "assistants",
        "top_p",
        existing_type=mysql.INTEGER(),
        type_=sa.Float(),
        existing_nullable=True,
    )
    safe_alter_column(
        "assistants",
        "temperature",
        existing_type=mysql.INTEGER(),
        type_=sa.Float(),
        existing_nullable=True,
    )
    safe_alter_column(
        "assistants",
        "agent_mode",
        existing_type=mysql.TINYINT(display_width=1),
        comment="False = Standard (Level 2), True = Autonomous (Level 3).",
        existing_nullable=False,
        existing_server_default=sa.text("'0'"),
    )
    safe_alter_column(
        "assistants",
        "web_access",
        existing_type=mysql.TINYINT(display_width=1),
        comment="Enable live web search and browsing capabilities.",
        existing_nullable=False,
        existing_server_default=sa.text("'0'"),
    )
    safe_alter_column(
        "assistants",
        "deep_research",
        existing_type=mysql.TINYINT(display_width=1),
        comment="Enable deep research capabilities.",
        existing_nullable=False,
        existing_server_default=sa.text("'0'"),
    )
    safe_alter_column(
        "assistants",
        "engineer",
        existing_type=mysql.TINYINT(display_width=1),
        comment="Enable network engineering capabilities and inventory map access.",
        existing_nullable=False,
        existing_server_default=sa.text("'0'"),
    )
    safe_alter_column(
        "assistants",
        "decision_telemetry",
        existing_type=mysql.TINYINT(display_width=1),
        comment="If True, captures detailed reasoning payloads and confidence scores.",
        existing_nullable=False,
        existing_server_default=sa.text("'0'"),
    )

    # -------------------------------------------------------------------------
    # 4. datasets
    # -------------------------------------------------------------------------
    safe_alter_column(
        "datasets",
        "status",
        existing_type=_STATUS_ENUM,
        nullable=False,
    )

    # -------------------------------------------------------------------------
    # 5. file_storage
    # -------------------------------------------------------------------------
    safe_alter_column(
        "file_storage",
        "is_primary",
        existing_type=mysql.TINYINT(display_width=1),
        comment="Indicates if this is the primary storage location",
        existing_nullable=True,
    )

    # -------------------------------------------------------------------------
    # 6. fine_tuned_models
    # -------------------------------------------------------------------------
    safe_alter_column(
        "fine_tuned_models",
        "is_active",
        existing_type=mysql.TINYINT(display_width=1),
        nullable=False,
    )
    safe_alter_column(
        "fine_tuned_models",
        "status",
        existing_type=_STATUS_ENUM,
        nullable=False,
    )

    # -------------------------------------------------------------------------
    # 7. messages
    # -------------------------------------------------------------------------
    safe_alter_column(
        "messages",
        "content",
        existing_type=mysql.LONGTEXT(),
        type_=sa.Text(length=4294967295),
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

    # -------------------------------------------------------------------------
    # 8. runs
    # -------------------------------------------------------------------------
    safe_alter_column(
        "runs",
        "status",
        existing_type=_STATUS_ENUM,
        nullable=False,
    )
    safe_alter_column(
        "runs",
        "temperature",
        existing_type=mysql.INTEGER(),
        type_=sa.Float(),
        existing_nullable=True,
    )
    safe_alter_column(
        "runs",
        "top_p",
        existing_type=sa.Float(),
        type_=mysql.INTEGER(),
        existing_nullable=True,
    )

    # -------------------------------------------------------------------------
    # 9. training_jobs
    # -------------------------------------------------------------------------
    safe_alter_column(
        "training_jobs",
        "status",
        existing_type=_STATUS_ENUM,
        nullable=False,
    )

    # -------------------------------------------------------------------------
    # 10. users
    # -------------------------------------------------------------------------
    safe_alter_column(
        "users",
        "is_admin",
        existing_type=mysql.TINYINT(display_width=1),
        comment="Flag indicating administrative privileges",
        existing_nullable=False,
        existing_server_default=sa.text("'0'"),
    )
    safe_alter_column(
        "users",
        "email_verified",
        existing_type=mysql.TINYINT(display_width=1),
        comment="Whether the email address has been verified",
        existing_nullable=True,
    )

    # -------------------------------------------------------------------------
    # 11. vector_stores
    # -------------------------------------------------------------------------
    safe_alter_column(
        "vector_stores",
        "status",
        existing_type=_STATUS_ENUM,
        nullable=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    safe_alter_column(
        "vector_stores",
        "status",
        existing_type=_STATUS_ENUM,
        nullable=True,
    )
    safe_alter_column(
        "users",
        "email_verified",
        existing_type=mysql.TINYINT(display_width=1),
        comment=None,
        existing_comment="Whether the email address has been verified",
        existing_nullable=True,
    )
    safe_alter_column(
        "users",
        "is_admin",
        existing_type=mysql.TINYINT(display_width=1),
        comment=None,
        existing_comment="Flag indicating administrative privileges",
        existing_nullable=False,
        existing_server_default=sa.text("'0'"),
    )
    safe_alter_column(
        "training_jobs",
        "status",
        existing_type=_STATUS_ENUM,
        nullable=True,
    )
    safe_alter_column(
        "runs",
        "top_p",
        existing_type=sa.Float(),
        type_=mysql.INTEGER(),
        existing_nullable=True,
    )
    safe_alter_column(
        "runs",
        "temperature",
        existing_type=sa.Float(),
        type_=mysql.INTEGER(),
        existing_nullable=True,
    )
    safe_alter_column(
        "runs",
        "status",
        existing_type=_STATUS_ENUM,
        nullable=True,
    )
    safe_alter_column(
        "messages",
        "reasoning",
        existing_type=sa.Text(length=4294967295),
        type_=mysql.LONGTEXT(),
        comment=None,
        existing_comment="Stores the internal 'thinking' or reasoning tokens from the model.",
        existing_nullable=True,
    )
    safe_alter_column(
        "messages",
        "content",
        existing_type=sa.Text(length=4294967295),
        type_=mysql.LONGTEXT(),
        nullable=True,
    )
    safe_alter_column(
        "fine_tuned_models",
        "status",
        existing_type=_STATUS_ENUM,
        nullable=True,
    )
    safe_alter_column(
        "fine_tuned_models",
        "is_active",
        existing_type=mysql.TINYINT(display_width=1),
        nullable=True,
    )
    safe_alter_column(
        "file_storage",
        "is_primary",
        existing_type=mysql.TINYINT(display_width=1),
        comment=None,
        existing_comment="Indicates if this is the primary storage location",
        existing_nullable=True,
    )
    safe_alter_column(
        "datasets",
        "status",
        existing_type=_STATUS_ENUM,
        nullable=True,
    )
    safe_alter_column(
        "assistants",
        "decision_telemetry",
        existing_type=mysql.TINYINT(display_width=1),
        comment=None,
        existing_comment="If True, captures detailed reasoning payloads and confidence scores.",
        existing_nullable=False,
        existing_server_default=sa.text("'0'"),
    )
    safe_alter_column(
        "assistants",
        "engineer",
        existing_type=mysql.TINYINT(display_width=1),
        comment=None,
        existing_comment="Enable network engineering capabilities and inventory map access.",
        existing_nullable=False,
        existing_server_default=sa.text("'0'"),
    )
    safe_alter_column(
        "assistants",
        "deep_research",
        existing_type=mysql.TINYINT(display_width=1),
        comment=None,
        existing_comment="Enable deep research capabilities.",
        existing_nullable=False,
        existing_server_default=sa.text("'0'"),
    )
    safe_alter_column(
        "assistants",
        "web_access",
        existing_type=mysql.TINYINT(display_width=1),
        comment=None,
        existing_comment="Enable live web search and browsing capabilities.",
        existing_nullable=False,
        existing_server_default=sa.text("'0'"),
    )
    safe_alter_column(
        "assistants",
        "agent_mode",
        existing_type=mysql.TINYINT(display_width=1),
        comment=None,
        existing_comment="False = Standard (Level 2), True = Autonomous (Level 3).",
        existing_nullable=False,
        existing_server_default=sa.text("'0'"),
    )
    safe_alter_column(
        "assistants",
        "temperature",
        existing_type=sa.Float(),
        type_=mysql.INTEGER(),
        existing_nullable=True,
    )
    safe_alter_column(
        "assistants",
        "top_p",
        existing_type=sa.Float(),
        type_=mysql.INTEGER(),
        existing_nullable=True,
    )
    op.drop_column("assistants", "max_tokens")
    safe_alter_column(
        "api_keys",
        "is_active",
        existing_type=mysql.TINYINT(display_width=1),
        nullable=True,
    )
    op.create_table(
        "batfish_snapshots",
        sa.Column(
            "id",
            mysql.VARCHAR(length=64),
            nullable=False,
            comment="Opaque snapshot ID returned to caller e.g. snap_abc123",
        ),
        sa.Column(
            "snapshot_name",
            mysql.VARCHAR(length=128),
            nullable=False,
            comment="Caller-supplied label e.g. 'incident_001'",
        ),
        sa.Column(
            "snapshot_key",
            mysql.VARCHAR(length=256),
            nullable=False,
            comment="Namespaced isolation key: {user_id}_{id}",
        ),
        sa.Column("user_id", mysql.VARCHAR(length=64), nullable=False),
        sa.Column("configs_root", mysql.VARCHAR(length=512), nullable=True),
        sa.Column("device_count", mysql.INTEGER(), autoincrement=False, nullable=False),
        sa.Column(
            "devices",
            mysql.JSON(),
            nullable=False,
            comment="List of hostnames ingested into this snapshot",
        ),
        sa.Column("status", _STATUS_ENUM, nullable=True),
        sa.Column("error_message", mysql.TEXT(), nullable=True),
        sa.Column("created_at", mysql.BIGINT(), autoincrement=False, nullable=False),
        sa.Column("updated_at", mysql.BIGINT(), autoincrement=False, nullable=False),
        sa.Column(
            "last_ingested_at", mysql.BIGINT(), autoincrement=False, nullable=True
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="batfish_snapshots_ibfk_1",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        mysql_collate="utf8mb4_0900_ai_ci",
        mysql_default_charset="utf8mb4",
        mysql_engine="InnoDB",
    )
    op.create_index(
        "uq_batfish_user_snapshot_name",
        "batfish_snapshots",
        ["user_id", "snapshot_name"],
        unique=True,
    )
    op.create_index(
        "ix_batfish_snapshots_user_id", "batfish_snapshots", ["user_id"], unique=False
    )
    op.create_index(
        "ix_batfish_snapshots_snapshot_key",
        "batfish_snapshots",
        ["snapshot_key"],
        unique=True,
    )
    op.create_index(
        "ix_batfish_snapshots_id", "batfish_snapshots", ["id"], unique=False
    )
    op.create_index(
        "idx_batfish_user_id", "batfish_snapshots", ["user_id"], unique=False
    )
    op.create_index("idx_batfish_status", "batfish_snapshots", ["status"], unique=False)
