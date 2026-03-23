"""Add training data to training.models2

Revision ID: d98d34517e5f
Revises: 66b1d150d350
Create Date: 2026-03-17 18:43:59.120874
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# Application Enums
from projectdavid_common.schemas.enums import StatusEnum
from sqlalchemy.dialects import mysql

# SafeDDL helpers
from migrations.utils.safe_ddl import (
    create_fk_if_not_exists,
    create_index_if_missing,
    has_table,
    safe_alter_column,
)

# revision identifiers, used by Alembic.
revision: str = 'd98d34517e5f'
down_revision: Union[str, None] = '66b1d150d350'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema safely."""

    # --- Table: messages ---
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
        existing_comment="Stores the internal 'thinking' or reasoning tokens from the model.",
        existing_nullable=True,
    )

    # --- Table: datasets ---
    if not has_table("datasets"):
        op.create_table(
            "datasets",
            sa.Column(
                "id",
                sa.String(length=64),
                nullable=False,
                comment="Opaque dataset ID e.g. ds_abc123",
            ),
            sa.Column("user_id", sa.String(length=64), nullable=False),
            sa.Column("name", sa.String(length=128), nullable=False),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column(
                "format",
                sa.String(length=32),
                nullable=False,
                comment="Training format: chatml | alpaca | sharegpt | jsonl",
            ),
            sa.Column(
                "file_id",
                sa.String(length=64),
                nullable=False,
                comment="Reference to the file_id in the core API files table.",
            ),
            sa.Column(
                "storage_path",
                sa.String(length=512),
                nullable=True,
                comment="Resolved Samba path — populated by worker at training time.",
            ),
            sa.Column("train_samples", sa.Integer(), nullable=True),
            sa.Column("eval_samples", sa.Integer(), nullable=True),
            sa.Column("config", sa.JSON(), nullable=True),
            sa.Column(
                "status",
                sa.Enum(StatusEnum),
                nullable=False,
                comment="pending → processing → active → failed",
            ),
            sa.Column("created_at", sa.BigInteger(), nullable=False),
            sa.Column("updated_at", sa.BigInteger(), nullable=False),
            sa.Column(
                "deleted_at",
                sa.Integer(),
                nullable=True,
                comment="Unix timestamp of soft-deletion.",
            ),
            sa.PrimaryKeyConstraint("id"),
        )

    create_index_if_missing("ix_datasets_id", "datasets", ["id"])
    create_index_if_missing("ix_datasets_user_id", "datasets", ["user_id"])
    create_index_if_missing("ix_datasets_file_id", "datasets", ["file_id"])
    create_index_if_missing("ix_datasets_deleted_at", "datasets", ["deleted_at"])
    create_index_if_missing("idx_dataset_user_id", "datasets", ["user_id"])
    create_index_if_missing("idx_dataset_status", "datasets", ["status"])

    create_fk_if_not_exists(
        "fk_datasets_user_id", "datasets", "users", ["user_id"], ["id"], ondelete="CASCADE"
    )

    # --- Table: training_jobs ---
    if not has_table("training_jobs"):
        op.create_table(
            "training_jobs",
            sa.Column(
                "id", sa.String(length=64), nullable=False, comment="Opaque job ID e.g. tj_abc123"
            ),
            sa.Column("user_id", sa.String(length=64), nullable=False),
            sa.Column(
                "dataset_id",
                sa.String(length=64),
                nullable=True,
                comment="Source dataset. SET NULL if dataset is deleted — job record is preserved.",
            ),
            sa.Column(
                "base_model",
                sa.String(length=256),
                nullable=False,
                comment="Base model identifier e.g. Qwen/Qwen2.5-7B-Instruct",
            ),
            sa.Column(
                "framework",
                sa.String(length=32),
                nullable=False,
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
                sa.Enum(StatusEnum),
                nullable=False,
                comment="queued → in_progress → completed | failed | cancelled",
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
                sa.String(length=512),
                nullable=True,
                comment="Samba path to the training output checkpoint.",
            ),
            sa.PrimaryKeyConstraint("id"),
        )

    create_index_if_missing("ix_training_jobs_id", "training_jobs", ["id"])
    create_index_if_missing("ix_training_jobs_user_id", "training_jobs", ["user_id"])
    create_index_if_missing("ix_training_jobs_dataset_id", "training_jobs", ["dataset_id"])
    create_index_if_missing("idx_trainingjob_user_id", "training_jobs", ["user_id"])
    create_index_if_missing("idx_trainingjob_status", "training_jobs", ["status"])
    create_index_if_missing("idx_trainingjob_dataset_id", "training_jobs", ["dataset_id"])

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

    # --- Table: fine_tuned_models ---
    if not has_table("fine_tuned_models"):
        op.create_table(
            "fine_tuned_models",
            sa.Column(
                "id",
                sa.String(length=64),
                nullable=False,
                comment="Opaque model ID e.g. ftm_abc123",
            ),
            sa.Column("user_id", sa.String(length=64), nullable=False),
            sa.Column(
                "training_job_id",
                sa.String(length=64),
                nullable=True,
                comment="Source training job. SET NULL if job is deleted — model record is preserved.",
            ),
            sa.Column("name", sa.String(length=128), nullable=False),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column(
                "base_model",
                sa.String(length=256),
                nullable=False,
                comment="Base model this was fine-tuned from.",
            ),
            sa.Column(
                "hf_repo",
                sa.String(length=256),
                nullable=True,
                comment="HuggingFace repository ID e.g. your-org/your-model",
            ),
            sa.Column(
                "storage_path",
                sa.String(length=512),
                nullable=True,
                comment="Local Samba path to model weights.",
            ),
            sa.Column(
                "is_active",
                sa.Boolean(),
                nullable=False,
                comment="True when this model is currently loaded in vLLM.",
            ),
            sa.Column(
                "vllm_model_id",
                sa.String(length=256),
                nullable=True,
                comment="The VLLM_MODEL value used to serve this model.",
            ),
            sa.Column(
                "status",
                sa.Enum(StatusEnum),
                nullable=False,
                comment="processing → active → failed",
            ),
            sa.Column("created_at", sa.BigInteger(), nullable=False),
            sa.Column("updated_at", sa.BigInteger(), nullable=False),
            sa.Column(
                "deleted_at",
                sa.Integer(),
                nullable=True,
                comment="Unix timestamp of soft-deletion.",
            ),
            sa.PrimaryKeyConstraint("id"),
        )

    create_index_if_missing("ix_fine_tuned_models_id", "fine_tuned_models", ["id"])
    create_index_if_missing("ix_fine_tuned_models_user_id", "fine_tuned_models", ["user_id"])
    create_index_if_missing(
        "ix_fine_tuned_models_training_job_id", "fine_tuned_models", ["training_job_id"]
    )
    create_index_if_missing("ix_fine_tuned_models_deleted_at", "fine_tuned_models", ["deleted_at"])
    create_index_if_missing("idx_finetunedmodel_user_id", "fine_tuned_models", ["user_id"])
    create_index_if_missing("idx_finetunedmodel_status", "fine_tuned_models", ["status"])
    create_index_if_missing("idx_finetunedmodel_is_active", "fine_tuned_models", ["is_active"])

    create_fk_if_not_exists(
        "fk_fine_tuned_models_user_id",
        "fine_tuned_models",
        "users",
        ["user_id"],
        ["id"],
        ondelete="CASCADE",
    )
    create_fk_if_not_exists(
        "fk_fine_tuned_models_training_job_id",
        "fine_tuned_models",
        "training_jobs",
        ["training_job_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    """Downgrade schema safely."""

    # --- Cascading table drops ---
    # (Dropping a table safely eliminates its contained indexes and keys within MySQL)
    if has_table("fine_tuned_models"):
        op.drop_table("fine_tuned_models")

    if has_table("training_jobs"):
        op.drop_table("training_jobs")

    if has_table("datasets"):
        op.drop_table("datasets")

    # --- Table: messages ---
    safe_alter_column(
        "messages",
        "reasoning",
        existing_type=sa.Text(length=4294967295),
        type_=mysql.LONGTEXT(),
        existing_comment="Stores the internal 'thinking' or reasoning tokens from the model.",
        existing_nullable=True,
    )

    safe_alter_column(
        "messages",
        "content",
        existing_type=mysql.TEXT(),
        nullable=True,
    )
