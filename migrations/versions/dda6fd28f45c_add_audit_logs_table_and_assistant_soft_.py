"""Add audit_logs table and assistant soft_delete

Revision ID: dda6fd28f45c
Revises: b697008df93a
Create Date: 2026-02-14 00:52:59.555724

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

# Import the safe DDL helpers
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

# revision identifiers, used by Alembic.
revision: str = "dda6fd28f45c"
down_revision: Union[str, None] = "b697008df93a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema safely."""

    # 1. Create Audit Logs Table (idempotent)
    # The FK to users is intentionally NOT declared inline in create_table.
    # MySQL InnoDB enforces FK references at CREATE TABLE time, so if users
    # doesn't exist yet in this session (fresh container), the statement fails
    # with ER_NO_REFERENCED_ROW (1824). Instead we create the table without the
    # FK and add it separately below via create_fk_if_not_exists, which checks
    # has_table("users") before attempting anything.
    if not has_table("audit_logs"):
        op.create_table(
            "audit_logs",
            sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
            sa.Column("user_id", sa.String(length=64), nullable=True),
            sa.Column(
                "action",
                sa.String(length=32),
                nullable=False,
                comment="e.g. CREATE, UPDATE, DELETE, HARD_DELETE",
            ),
            sa.Column(
                "entity_type",
                sa.String(length=64),
                nullable=False,
                comment="e.g. Assistant, User, Thread",
            ),
            sa.Column("entity_id", sa.String(length=64), nullable=False),
            sa.Column("timestamp", sa.DateTime(), nullable=False),
            sa.Column("ip_address", sa.String(length=45), nullable=True),
            sa.Column(
                "details",
                sa.JSON(),
                nullable=True,
                comment="Stores before/after state or reasoning for action",
            ),
            sa.PrimaryKeyConstraint("id"),
        )
        print("[Alembic-safeDDL] ✅ Created table: audit_logs")
    else:
        print("[Alembic-safeDDL] ⚠️ Skipped – table already exists: audit_logs")

    # Add the FK to users as a separate deferred step.
    # create_fk_if_not_exists checks has_table for both source and referent tables
    # before acting — no-op on fresh containers where users isn't created yet,
    # and idempotent on existing databases where the FK is already present.
    # NOTE: if users doesn't exist at this point in the chain on a fresh container,
    # the FK will simply be absent until the migration that creates users runs.
    # The 222cafa3baac GDPR migration later calls replace_fk("audit_logs_ibfk_1",...)
    # which will add it at that point if it was skipped here.
    create_fk_if_not_exists(
        "audit_logs_ibfk_1",
        "audit_logs",
        "users",
        ["user_id"],
        ["id"],
    )

    # Create indexes via helper — guards against missing table and duplicate indexes.
    # Safe whether the table was just created above or already existed.
    create_index_if_missing("ix_audit_logs_action", "audit_logs", ["action"])
    create_index_if_missing("ix_audit_logs_entity_id", "audit_logs", ["entity_id"])
    create_index_if_missing("ix_audit_logs_entity_type", "audit_logs", ["entity_type"])
    create_index_if_missing("ix_audit_logs_id", "audit_logs", ["id"])
    create_index_if_missing("ix_audit_logs_timestamp", "audit_logs", ["timestamp"])
    create_index_if_missing("ix_audit_logs_user_id", "audit_logs", ["user_id"])

    # 2. Add deleted_at to assistants
    # add_column_if_missing checks has_table internally — no outer guard needed.
    add_column_if_missing(
        "assistants",
        sa.Column(
            "deleted_at",
            sa.Integer(),
            nullable=True,
            comment="Unix timestamp of soft-deletion. If present, entity is in 'Recycle Bin'.",
        ),
    )

    # 3. Message table updates (type hardening)
    # safe_alter_column checks has_table internally — no outer guard needed.
    safe_alter_column("messages", "content", existing_type=mysql.TEXT(), nullable=False)
    safe_alter_column(
        "messages",
        "reasoning",
        existing_type=mysql.LONGTEXT(),
        type_=sa.Text(length=4294967295),
        existing_comment="Stores the internal 'thinking' or reasoning tokens from the model.",
        existing_nullable=True,
    )


def downgrade() -> None:
    """Downgrade schema safely."""

    # 1. Revert message types
    # safe_alter_column checks has_table internally — no outer guard needed.
    safe_alter_column(
        "messages",
        "reasoning",
        existing_type=sa.Text(length=4294967295),
        type_=mysql.LONGTEXT(),
        existing_comment="Stores the internal 'thinking' or reasoning tokens from the model.",
        existing_nullable=True,
    )
    safe_alter_column("messages", "content", existing_type=mysql.TEXT(), nullable=True)

    # 2. Drop deleted_at column
    # drop_column_if_exists checks has_table internally — no outer guard needed.
    drop_column_if_exists("assistants", "deleted_at")

    # 3. Drop audit_logs indexes, FK, then table
    # drop_index_if_exists and drop_fk_if_exists check has_table internally.
    drop_index_if_exists("ix_audit_logs_user_id", "audit_logs")
    drop_index_if_exists("ix_audit_logs_timestamp", "audit_logs")
    drop_index_if_exists("ix_audit_logs_id", "audit_logs")
    drop_index_if_exists("ix_audit_logs_entity_type", "audit_logs")
    drop_index_if_exists("ix_audit_logs_entity_id", "audit_logs")
    drop_index_if_exists("ix_audit_logs_action", "audit_logs")
    drop_fk_if_exists("audit_logs", "audit_logs_ibfk_1")

    if has_table("audit_logs"):
        op.drop_table("audit_logs")
        print("[Alembic-safeDDL] ✅ Dropped table: audit_logs")
    else:
        print("[Alembic-safeDDL] ⚠️ Skipped – table does not exist: audit_logs")
