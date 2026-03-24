"""Add fine tuning tables

Revision ID: 005820173bc4
Revises: ba35b4620058
Create Date: 2026-03-17 01:36:41.012316
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

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
    """Standardized Upgrade with Emergency Foundation Rescue."""

    # ──────────────────────────────────────────────────────────────────
    # PART 0: EMERGENCY FOUNDATION RESCUE
    # ──────────────────────────────────────────────────────────────────
    # The background purge daemons are crashing because these tables are
    # missing. We build them here if they don't already exist to stop the
    # crash loops immediately.

    # 1. Runs (crashing purge_expired_runs)
    if not has_table("runs"):
        op.create_table(
            "runs",
            sa.Column("id", sa.String(64), primary_key=True),
            sa.Column("user_id", sa.String(64), nullable=True),
            sa.Column("thread_id", sa.String(64), nullable=False),
            sa.Column("assistant_id", sa.String(64), nullable=False),
            sa.Column("status", sa.String(32), nullable=False),
            sa.Column("created_at", sa.Integer()),
        )
        print("[alembic.safe_ddl] 🚑 RESCUE: Created missing table: runs")

    # 2. Messages (crashing purge_orphaned_threads)
    if not has_table("messages"):
        op.create_table(
            "messages",
            sa.Column("id", sa.String(64), primary_key=True),
            sa.Column("thread_id", sa.String(64), nullable=False),
            sa.Column("role", sa.String(32), nullable=False),
            sa.Column("content", sa.Text(), nullable=False),
            sa.Column("created_at", sa.Integer()),
        )
        print("[alembic.safe_ddl] 🚑 RESCUE: Created missing table: messages")

    # 3. Files (crashing purge_expired_files)
    if not has_table("files"):
        op.create_table(
            "files",
            sa.Column("id", sa.String(64), primary_key=True),
            sa.Column("user_id", sa.String(64), nullable=False),
            sa.Column("filename", sa.String(256), nullable=False),
            sa.Column("purpose", sa.String(64), nullable=False),
            sa.Column("expires_at", sa.DateTime()),
            sa.Column("deleted_at", sa.Integer()),
            sa.Column("created_at", sa.DateTime()),
        )
        print("[alembic.safe_ddl] 🚑 RESCUE: Created missing table: files")

    # 4. Vector Stores (crashing vs_soft_delete_purge)
    if not has_table("vector_stores"):
        op.create_table(
            "vector_stores",
            sa.Column("id", sa.String(64), primary_key=True),
            sa.Column("user_id", sa.String(64), nullable=False),
            sa.Column("name", sa.String(128), nullable=False),
            sa.Column("status", sa.String(32), nullable=False),
            sa.Column("deleted_at", sa.Integer()),
            sa.Column("created_at", sa.BigInteger()),
        )
        print("[alembic.safe_ddl] 🚑 RESCUE: Created missing table: vector_stores")

    # ──────────────────────────────────────────────────────────────────
    # PART 1: ADD FINE TUNING TABLES (Original logic)
    # ──────────────────────────────────────────────────────────────────

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

    # ──────────────────────────────────────────────────────────────────
    # PART 2: DECOUPLED FOREIGN KEYS
    # ──────────────────────────────────────────────────────────────────
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

    # Indexes & Alters
    create_index_if_missing("idx_dataset_status", "datasets", ["status"])
    create_index_if_missing("idx_trainingjob_status", "training_jobs", ["status"])
    create_index_if_missing("idx_finetunedmodel_status", "fine_tuned_models", ["status"])

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
    if has_table("fine_tuned_models"):
        op.drop_table("fine_tuned_models")
    if has_table("training_jobs"):
        op.drop_table("training_jobs")
    if has_table("datasets"):
        op.drop_table("datasets")
