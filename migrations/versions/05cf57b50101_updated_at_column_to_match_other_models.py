"""updated_at column to match other models

Revision ID: 05cf57b50101
Revises: 02ce09f95a67
Create Date: 2026-03-19 20:12:26.477236

"""

import time
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
    drop_index_if_exists,
    has_table,
)

# revision identifiers, used by Alembic.
revision: str = '05cf57b50101'
down_revision: Union[str, None] = '02ce09f95a67'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Status options derived from your model's StatusEnum
STATUS_ENUM_VALUES = [
    'deleted',
    'active',
    'queued',
    'in_progress',
    'pending_action',
    'completed',
    'failed',
    'cancelling',
    'cancelled',
    'pending',
    'processing',
    'expired',
    'retrying',
]


def upgrade() -> None:
    """Upgrade schema safely for training_jobs."""

    # 1. Create the table from scratch if it doesn't exist at all
    if not has_table("training_jobs"):
        op.create_table(
            "training_jobs",
            sa.Column(
                "id",
                sa.String(64),
                primary_key=True,
                index=True,
                comment="Opaque job ID e.g. tj_abc123",
            ),
            sa.Column("user_id", sa.String(64), nullable=False, index=True),
            sa.Column("dataset_id", sa.String(64), nullable=True, index=True),
            sa.Column("base_model", sa.String(256), nullable=False),
            sa.Column("framework", sa.String(32), nullable=False, server_default="axolotl"),
            sa.Column("config", sa.JSON(), nullable=True),
            sa.Column(
                "status",
                sa.Enum(*STATUS_ENUM_VALUES, name="status_enum"),
                nullable=False,
                server_default="queued",
            ),
            sa.Column("created_at", sa.BigInteger(), nullable=False),
            sa.Column("started_at", sa.BigInteger(), nullable=True),
            sa.Column("updated_at", sa.BigInteger(), nullable=False),
            sa.Column("completed_at", sa.BigInteger(), nullable=True),
            sa.Column("failed_at", sa.BigInteger(), nullable=True),
            sa.Column("last_error", sa.Text(), nullable=True),
            sa.Column("metrics", sa.JSON(), nullable=True),
            sa.Column("output_path", sa.String(512), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )

    # 2. If the table exists but is missing specific columns (like updated_at), add them
    add_column_if_missing(
        "training_jobs",
        sa.Column(
            "updated_at", sa.BigInteger(), nullable=False, server_default=str(int(time.time()))
        ),
    )
    add_column_if_missing("training_jobs", sa.Column("metrics", sa.JSON(), nullable=True))
    add_column_if_missing("training_jobs", sa.Column("output_path", sa.String(512), nullable=True))

    # 3. Ensure Foreign Keys exist
    create_fk_if_not_exists(
        "fk_training_jobs_user_id",
        "training_jobs",
        "users",
        ["user_id"],
        ["id"],
        ondelete="CASCADE",
    )
    create_fk_if_not_exists(
        "fk_training_jobs_dataset_id",
        "training_jobs",
        "datasets",
        ["dataset_id"],
        ["id"],
        ondelete="SET NULL",
    )

    # 4. Ensure Indexes exist
    create_index_if_missing("idx_trainingjob_user_id", "training_jobs", ["user_id"])
    create_index_if_missing("idx_trainingjob_status", "training_jobs", ["status"])
    create_index_if_missing("idx_trainingjob_dataset_id", "training_jobs", ["dataset_id"])


def downgrade() -> None:
    """Downgrade schema safely."""
    # Only remove the columns specifically added in this revision if necessary,
    # or drop the table if this migration was the one that created it.
    drop_index_if_exists("idx_trainingjob_dataset_id", "training_jobs")
    drop_index_if_exists("idx_trainingjob_status", "training_jobs")
    drop_index_if_exists("idx_trainingjob_user_id", "training_jobs")

    # Normally we don't drop the whole table in production downgrades to prevent data loss,
    # but we can remove the specific new columns:
    drop_column_if_exists("training_jobs", "updated_at")
