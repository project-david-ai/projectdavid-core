"""Add fine tuning tables

Revision ID: 005820173bc4
Revises: ba35b4620058
Create Date: 2026-03-17 01:36:41.012316
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

# Use your official Organization-wide SafeDDL helpers
from migrations.utils.safe_ddl import \
    drop_table_if_exists  # Optional: for cleaner downgrades
from migrations.utils.safe_ddl import (create_fk_if_not_exists,
                                       create_index_if_missing,
                                       drop_index_if_exists, has_table,
                                       safe_alter_column)

# revision identifiers, used by Alembic.
revision: str = "005820173bc4"
down_revision: Union[str, None] = "ba35b4620058"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = "9351530d20ab"


def upgrade() -> None:
    """Upgrade schema safely using SafeDDL patterns."""

    # 1. TABLE: datasets
    # Note: Foreign Keys are REMOVED from the create_table block to prevent Error 1824
    if not has_table("datasets"):
        op.create_table(
            "datasets",
            sa.Column("id", sa.String(length=64), primary_key=True, index=True),
            sa.Column("user_id", sa.String(length=64), nullable=False),
            sa.Column("name", sa.String(length=128), nullable=False),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("format", sa.String(length=32), nullable=False),
            sa.Column("storage_path", sa.String(length=512), nullable=False),
            sa.Column("train_samples", sa.Integer(), nullable=True),
            sa.Column("eval_samples", sa.Integer(), nullable=True),
            sa.Column("config", sa.JSON(), nullable=True),
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
            ),
            sa.Column("created_at", sa.BigInteger(), nullable=False),
            sa.Column("updated_at", sa.BigInteger(), nullable=False),
            sa.Column("deleted_at", sa.Integer(), nullable=True),
        )
        print("[alembic.safe_ddl] ✅ Created table: datasets")

    # 2. TABLE: training_jobs
    if not has_table("training_jobs"):
        op.create_table(
            "training_jobs",
            sa.Column("id", sa.String(length=64), primary_key=True, index=True),
            sa.Column("user_id", sa.String(length=64), nullable=False),
            sa.Column("dataset_id", sa.String(length=64), nullable=True),
            sa.Column("base_model", sa.String(length=256), nullable=False),
            sa.Column("framework", sa.String(length=32), nullable=False),
            sa.Column("config", sa.JSON(), nullable=True),
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
            ),
            sa.Column("created_at", sa.BigInteger(), nullable=False),
            sa.Column("started_at", sa.BigInteger(), nullable=True),
            sa.Column("completed_at", sa.BigInteger(), nullable=True),
            sa.Column("failed_at", sa.BigInteger(), nullable=True),
            sa.Column("last_error", sa.Text(), nullable=True),
            sa.Column("metrics", sa.JSON(), nullable=True),
            sa.Column("output_path", sa.String(length=512), nullable=True),
        )
        print("[alembic.safe_ddl] ✅ Created table: training_jobs")

    # 3. TABLE: fine_tuned_models
    if not has_table("fine_tuned_models"):
        op.create_table(
            "fine_tuned_models",
            sa.Column("id", sa.String(length=64), primary_key=True, index=True),
            sa.Column("user_id", sa.String(length=64), nullable=False),
            sa.Column("training_job_id", sa.String(length=64), nullable=True),
            sa.Column("name", sa.String(length=128), nullable=False),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("base_model", sa.String(length=256), nullable=False),
            sa.Column("hf_repo", sa.String(length=256), nullable=True),
            sa.Column("storage_path", sa.String(length=512), nullable=True),
            sa.Column("is_active", sa.Boolean(), nullable=False),
            sa.Column("vllm_model_id", sa.String(length=256), nullable=True),
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
            ),
            sa.Column("created_at", sa.BigInteger(), nullable=False),
            sa.Column("updated_at", sa.BigInteger(), nullable=False),
            sa.Column("deleted_at", sa.Integer(), nullable=True),
        )
        print("[alembic.safe_ddl] ✅ Created table: fine_tuned_models")

    # --- 4. ADD FOREIGN KEYS (Using SafeDDL helpers) ---
    # These helpers guard against Error 1824 by checking if the parent table exists first.
    create_fk_if_not_exists(
        "fk_datasets_user_id", "datasets", "users", ["user_id"], ["id"], ondelete="CASCADE"
    )
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
    create_fk_if_not_exists(
        "fk_ftm_user_id", "fine_tuned_models", "users", ["user_id"], ["id"], ondelete="CASCADE"
    )
    create_fk_if_not_exists(
        "fk_ftm_job_id",
        "fine_tuned_models",
        "training_jobs",
        ["training_job_id"],
        ["id"],
        ondelete="SET NULL",
    )

    # Audit Logs & Snapshots
    create_fk_if_not_exists(
        "fk_audit_logs_user_id", "audit_logs", "users", ["user_id"], ["id"], ondelete="SET NULL"
    )
    create_fk_if_not_exists(
        "fk_batfish_snapshots_user_id",
        "batfish_snapshots",
        "users",
        ["user_id"],
        ["id"],
        ondelete="CASCADE",
    )

    # --- 5. ADD INDEXES ---
    create_index_if_missing("idx_dataset_status", "datasets", ["status"])
    create_index_if_missing("idx_trainingjob_status", "training_jobs", ["status"])
    create_index_if_missing("idx_finetunedmodel_status", "fine_tuned_models", ["status"])

    # --- 6. SAFE ALTERS ---
    safe_alter_column(
        "audit_logs",
        "action",
        existing_type=mysql.VARCHAR(length=32),
        comment="e.g. CREATE, UPDATE, DELETE, HARD_DELETE, ERASE",
    )
    safe_alter_column("messages", "content", existing_type=mysql.TEXT(), nullable=False)
    safe_alter_column(
        "messages",
        "reasoning",
        existing_type=mysql.LONGTEXT(),
        type_=sa.Text(length=4294967295),
        existing_nullable=True,
    )


def downgrade() -> None:
    """Safe downgrade logic."""
    # Note: In a production environment, usually we just drop the tables we created
    # but the SafeDDL policy usually prefers avoiding massive drops in prod.
    # For dev resetting, dropping is fine.
    if has_table("fine_tuned_models"):
        op.drop_table("fine_tuned_models")
    if has_table("training_jobs"):
        op.drop_table("training_jobs")
    if has_table("datasets"):
        op.drop_table("datasets")
