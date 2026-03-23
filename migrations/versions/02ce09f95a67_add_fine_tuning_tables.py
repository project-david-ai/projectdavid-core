"""Add fine tuning tables

Revision ID: 02ce09f95a67
Revises: d98d34517e5f
Create Date: 2026-03-17 20:07:55.349216

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

from migrations.utils.safe_ddl import (create_fk_if_not_exists,
                                       create_index_if_missing,
                                       drop_fk_if_exists, drop_index_if_exists,
                                       has_table)

# revision identifiers, used by Alembic.
revision: str = "02ce09f95a67"
down_revision: Union[str, None] = "d98d34517e5f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema — create fine-tuning tables if they do not exist."""

    # ------------------------------------------------------------------ #
    # 1. datasets                                                          #
    # ------------------------------------------------------------------ #
    if not has_table("datasets"):
        op.create_table(
            "datasets",
            sa.Column(
                "id",
                sa.String(64),
                primary_key=True,
                index=True,
                comment="Opaque dataset ID e.g. ds_abc123",
            ),
            sa.Column("user_id", sa.String(64), nullable=False, index=True),
            sa.Column("name", sa.String(128), nullable=False),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column(
                "format",
                sa.String(32),
                nullable=False,
                comment="Training format: chatml | alpaca | sharegpt | jsonl",
            ),
            sa.Column(
                "file_id",
                sa.String(64),
                nullable=False,
                index=True,
                comment="Reference to the file_id in the core API files table.",
            ),
            sa.Column(
                "storage_path",
                sa.String(512),
                nullable=True,
                comment="Resolved Samba path — populated by worker at training time.",
            ),
            sa.Column("train_samples", sa.Integer(), nullable=True),
            sa.Column("eval_samples", sa.Integer(), nullable=True),
            sa.Column("config", sa.JSON(), nullable=True),
            sa.Column(
                "status",
                sa.Enum(
                    "pending",
                    "processing",
                    "active",
                    "queued",
                    "in_progress",
                    "completed",
                    "failed",
                    "cancelled",
                    name="statusenum",
                ),
                nullable=False,
                server_default="pending",
            ),
            sa.Column("created_at", sa.BigInteger(), nullable=False),
            sa.Column("updated_at", sa.BigInteger(), nullable=False),
            sa.Column(
                "deleted_at",
                sa.Integer(),
                nullable=True,
                comment="Unix timestamp of soft-deletion.",
            ),
        )
        print("[alembic.safe_ddl] ✅  Created table: datasets", flush=True)
    else:
        print("[alembic.safe_ddl] ⚠️  Skipped – table already exists: datasets", flush=True)

    create_fk_if_not_exists(
        "fk_datasets_user_id",
        "datasets",
        "users",
        ["user_id"],
        ["id"],
        ondelete="CASCADE",
    )
    create_index_if_missing("idx_dataset_user_id", "datasets", ["user_id"])
    create_index_if_missing("idx_dataset_status", "datasets", ["status"])

    # ------------------------------------------------------------------ #
    # 2. training_jobs                                                     #
    # ------------------------------------------------------------------ #
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
            sa.Column(
                "dataset_id",
                sa.String(64),
                nullable=True,
                index=True,
                comment="Source dataset. SET NULL if dataset is deleted.",
            ),
            sa.Column(
                "base_model",
                sa.String(256),
                nullable=False,
                comment="Base model identifier e.g. Qwen/Qwen2.5-7B-Instruct",
            ),
            sa.Column(
                "framework",
                sa.String(32),
                nullable=False,
                server_default="axolotl",
                comment="Training framework: axolotl | unsloth",
            ),
            sa.Column(
                "config",
                sa.JSON(),
                nullable=True,
                comment="Complete training configuration passed to the training container.",
            ),
            sa.Column(
                "status",
                sa.Enum(
                    "pending",
                    "processing",
                    "active",
                    "queued",
                    "in_progress",
                    "completed",
                    "failed",
                    "cancelled",
                    name="statusenum",
                ),
                nullable=False,
                server_default="queued",
            ),
            sa.Column("created_at", sa.BigInteger(), nullable=False),
            sa.Column("started_at", sa.BigInteger(), nullable=True),
            sa.Column("completed_at", sa.BigInteger(), nullable=True),
            sa.Column("failed_at", sa.BigInteger(), nullable=True),
            sa.Column("last_error", sa.Text(), nullable=True),
            sa.Column(
                "metrics",
                sa.JSON(),
                nullable=True,
                comment="Final training metrics: loss, eval_loss, perplexity etc.",
            ),
            sa.Column(
                "output_path",
                sa.String(512),
                nullable=True,
                comment="Samba path to the training output checkpoint.",
            ),
        )
        print("[alembic.safe_ddl] ✅  Created table: training_jobs", flush=True)
    else:
        print("[alembic.safe_ddl] ⚠️  Skipped – table already exists: training_jobs", flush=True)

    create_fk_if_not_exists(
        "fk_trainingjob_user_id",
        "training_jobs",
        "users",
        ["user_id"],
        ["id"],
        ondelete="CASCADE",
    )
    create_fk_if_not_exists(
        "fk_trainingjob_dataset_id",
        "training_jobs",
        "datasets",
        ["dataset_id"],
        ["id"],
        ondelete="SET NULL",
    )
    create_index_if_missing("idx_trainingjob_user_id", "training_jobs", ["user_id"])
    create_index_if_missing("idx_trainingjob_status", "training_jobs", ["status"])
    create_index_if_missing("idx_trainingjob_dataset_id", "training_jobs", ["dataset_id"])

    # ------------------------------------------------------------------ #
    # 3. fine_tuned_models                                                 #
    # ------------------------------------------------------------------ #
    if not has_table("fine_tuned_models"):
        op.create_table(
            "fine_tuned_models",
            sa.Column(
                "id",
                sa.String(64),
                primary_key=True,
                index=True,
                comment="Opaque model ID e.g. ftm_abc123",
            ),
            sa.Column("user_id", sa.String(64), nullable=False, index=True),
            sa.Column(
                "training_job_id",
                sa.String(64),
                nullable=True,
                index=True,
                comment="Source training job. SET NULL if job is deleted.",
            ),
            sa.Column("name", sa.String(128), nullable=False),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column(
                "base_model",
                sa.String(256),
                nullable=False,
                comment="Base model this was fine-tuned from.",
            ),
            sa.Column(
                "hf_repo",
                sa.String(256),
                nullable=True,
                comment="HuggingFace repository ID e.g. your-org/your-model",
            ),
            sa.Column(
                "storage_path",
                sa.String(512),
                nullable=True,
                comment="Local Samba path to model weights.",
            ),
            sa.Column(
                "is_active",
                sa.Boolean(),
                nullable=False,
                server_default="0",
                comment="True when this model is currently loaded in vLLM.",
            ),
            sa.Column(
                "vllm_model_id",
                sa.String(256),
                nullable=True,
                comment="The VLLM_MODEL value used to serve this model.",
            ),
            sa.Column(
                "status",
                sa.Enum(
                    "pending",
                    "processing",
                    "active",
                    "queued",
                    "in_progress",
                    "completed",
                    "failed",
                    "cancelled",
                    name="statusenum",
                ),
                nullable=False,
                server_default="processing",
            ),
            sa.Column("created_at", sa.BigInteger(), nullable=False),
            sa.Column("updated_at", sa.BigInteger(), nullable=False),
            sa.Column(
                "deleted_at",
                sa.Integer(),
                nullable=True,
                comment="Unix timestamp of soft-deletion.",
            ),
        )
        print("[alembic.safe_ddl] ✅  Created table: fine_tuned_models", flush=True)
    else:
        print("[alembic.safe_ddl] ⚠️  Skipped – table already exists: fine_tuned_models", flush=True)

    create_fk_if_not_exists(
        "fk_finetunedmodel_user_id",
        "fine_tuned_models",
        "users",
        ["user_id"],
        ["id"],
        ondelete="CASCADE",
    )
    create_fk_if_not_exists(
        "fk_finetunedmodel_training_job_id",
        "fine_tuned_models",
        "training_jobs",
        ["training_job_id"],
        ["id"],
        ondelete="SET NULL",
    )
    create_index_if_missing("idx_finetunedmodel_user_id", "fine_tuned_models", ["user_id"])
    create_index_if_missing("idx_finetunedmodel_status", "fine_tuned_models", ["status"])
    create_index_if_missing("idx_finetunedmodel_is_active", "fine_tuned_models", ["is_active"])


def downgrade() -> None:
    """Downgrade schema — drop fine-tuning tables in reverse dependency order."""

    # fine_tuned_models depends on training_jobs — drop it first
    drop_fk_if_exists("fine_tuned_models", "fk_finetunedmodel_training_job_id")
    drop_fk_if_exists("fine_tuned_models", "fk_finetunedmodel_user_id")
    drop_index_if_exists("idx_finetunedmodel_is_active", "fine_tuned_models")
    drop_index_if_exists("idx_finetunedmodel_status", "fine_tuned_models")
    drop_index_if_exists("idx_finetunedmodel_user_id", "fine_tuned_models")
    if has_table("fine_tuned_models"):
        op.drop_table("fine_tuned_models")
        print("[alembic.safe_ddl] 🗑️  Dropped table: fine_tuned_models", flush=True)

    # training_jobs depends on datasets — drop it second
    drop_fk_if_exists("training_jobs", "fk_trainingjob_dataset_id")
    drop_fk_if_exists("training_jobs", "fk_trainingjob_user_id")
    drop_index_if_exists("idx_trainingjob_dataset_id", "training_jobs")
    drop_index_if_exists("idx_trainingjob_status", "training_jobs")
    drop_index_if_exists("idx_trainingjob_user_id", "training_jobs")
    if has_table("training_jobs"):
        op.drop_table("training_jobs")
        print("[alembic.safe_ddl] 🗑️  Dropped table: training_jobs", flush=True)

    # datasets last
    drop_fk_if_exists("datasets", "fk_datasets_user_id")
    drop_index_if_exists("idx_dataset_status", "datasets")
    drop_index_if_exists("idx_dataset_user_id", "datasets")
    if has_table("datasets"):
        op.drop_table("datasets")
        print("[alembic.safe_ddl] 🗑️  Dropped table: datasets", flush=True)
