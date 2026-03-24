"""Add fine tuning tables

Revision ID: 005820173bc4
Revises: ba35b4620058
Create Date: 2026-03-17 01:36:41.012316
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

# SafeDDL helpers
from migrations.utils.safe_ddl import (add_column_if_missing,
                                       drop_column_if_exists, has_column,
                                       has_table, safe_alter_column)

# revision identifiers, used by Alembic.
revision: str = "005820173bc4"
down_revision: Union[str, None] = "ba35b4620058"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = "9351530d20ab"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _index_exists(index_name: str, table_name: str) -> bool:
    """Return True if *index_name* already exists on *table_name*."""
    bind = op.get_bind()
    result = bind.execute(
        sa.text(
            "SELECT COUNT(*) FROM information_schema.statistics "
            "WHERE table_schema = DATABASE() "
            "  AND table_name   = :tbl "
            "  AND index_name   = :idx"
        ),
        {"tbl": table_name, "idx": index_name},
    )
    return result.scalar() > 0


def _create_index_if_missing(
    index_name: str, table_name: str, columns: list, unique: bool = False
) -> None:
    if not _index_exists(index_name, table_name):
        op.create_index(index_name, table_name, columns, unique=unique)


def _drop_index_if_exists(index_name: str, table_name: str) -> None:
    if _index_exists(index_name, table_name):
        op.drop_index(index_name, table_name=table_name)


def _fk_exists(table_name: str, fk_name: str) -> bool:
    """Return True if a named foreign key constraint exists."""
    bind = op.get_bind()
    result = bind.execute(
        sa.text(
            "SELECT COUNT(*) FROM information_schema.table_constraints "
            "WHERE table_schema    = DATABASE() "
            "  AND table_name      = :tbl "
            "  AND constraint_name = :fk "
            "  AND constraint_type = 'FOREIGN KEY'"
        ),
        {"tbl": table_name, "fk": fk_name},
    )
    return result.scalar() > 0


def _fk_exists_by_columns(from_table: str, from_col: str, to_table: str, to_col: str) -> bool:
    """Return True if any FK from *from_table.from_col* → *to_table.to_col* exists."""
    bind = op.get_bind()
    result = bind.execute(
        sa.text(
            "SELECT COUNT(*) "
            "FROM information_schema.key_column_usage kcu "
            "JOIN information_schema.table_constraints tc "
            "  ON tc.constraint_name = kcu.constraint_name "
            " AND tc.table_schema    = kcu.table_schema "
            "WHERE kcu.table_schema        = DATABASE() "
            "  AND kcu.table_name          = :from_tbl "
            "  AND kcu.column_name         = :from_col "
            "  AND kcu.referenced_table_name = :to_tbl "
            "  AND kcu.referenced_column_name = :to_col "
            "  AND tc.constraint_type      = 'FOREIGN KEY'"
        ),
        {"from_tbl": from_table, "from_col": from_col, "to_tbl": to_table, "to_col": to_col},
    )
    return result.scalar() > 0


def _create_fk_if_missing(
    from_table: str, from_col: str, to_table: str, to_col: str, ondelete: str
) -> None:
    if not _fk_exists_by_columns(from_table, from_col, to_table, to_col):
        op.create_foreign_key(None, from_table, to_table, [from_col], [to_col], ondelete=ondelete)


def _drop_fk_by_columns_if_exists(
    from_table: str, from_col: str, to_table: str, to_col: str
) -> None:
    """Find and drop the FK constraint matching the given column pair, if present."""
    bind = op.get_bind()
    result = bind.execute(
        sa.text(
            "SELECT kcu.constraint_name "
            "FROM information_schema.key_column_usage kcu "
            "JOIN information_schema.table_constraints tc "
            "  ON tc.constraint_name = kcu.constraint_name "
            " AND tc.table_schema    = kcu.table_schema "
            "WHERE kcu.table_schema          = DATABASE() "
            "  AND kcu.table_name            = :from_tbl "
            "  AND kcu.column_name           = :from_col "
            "  AND kcu.referenced_table_name = :to_tbl "
            "  AND kcu.referenced_column_name = :to_col "
            "  AND tc.constraint_type        = 'FOREIGN KEY' "
            "LIMIT 1"
        ),
        {"from_tbl": from_table, "from_col": from_col, "to_tbl": to_table, "to_col": to_col},
    )
    row = result.fetchone()
    if row:
        op.drop_constraint(row[0], from_table, type_="foreignkey")


# ---------------------------------------------------------------------------
# upgrade
# ---------------------------------------------------------------------------


def upgrade() -> None:
    """Upgrade schema safely."""

    # ------------------------------------------------------------------ #
    # Table: datasets                                                       #
    # ------------------------------------------------------------------ #
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
                "storage_path",
                sa.String(length=512),
                nullable=False,
                comment="Path to the dataset file(s) on Samba, relative to the share root.",
            ),
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
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )

    _create_index_if_missing("idx_dataset_status", "datasets", ["status"])
    _create_index_if_missing("idx_dataset_user_id", "datasets", ["user_id"])
    _create_index_if_missing("ix_datasets_deleted_at", "datasets", ["deleted_at"])
    _create_index_if_missing("ix_datasets_id", "datasets", ["id"])
    _create_index_if_missing("ix_datasets_user_id", "datasets", ["user_id"])

    # ------------------------------------------------------------------ #
    # Table: training_jobs                                                  #
    # ------------------------------------------------------------------ #
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
            sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="SET NULL"),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )

    _create_index_if_missing("idx_trainingjob_dataset_id", "training_jobs", ["dataset_id"])
    _create_index_if_missing("idx_trainingjob_status", "training_jobs", ["status"])
    _create_index_if_missing("idx_trainingjob_user_id", "training_jobs", ["user_id"])
    _create_index_if_missing("ix_training_jobs_dataset_id", "training_jobs", ["dataset_id"])
    _create_index_if_missing("ix_training_jobs_id", "training_jobs", ["id"])
    _create_index_if_missing("ix_training_jobs_user_id", "training_jobs", ["user_id"])

    # ------------------------------------------------------------------ #
    # Table: fine_tuned_models                                              #
    # ------------------------------------------------------------------ #
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
                comment="Local Samba path to model weights (used when not pushed to HF).",
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
            sa.ForeignKeyConstraint(["training_job_id"], ["training_jobs.id"], ondelete="SET NULL"),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )

    _create_index_if_missing("idx_finetunedmodel_is_active", "fine_tuned_models", ["is_active"])
    _create_index_if_missing("idx_finetunedmodel_status", "fine_tuned_models", ["status"])
    _create_index_if_missing("idx_finetunedmodel_user_id", "fine_tuned_models", ["user_id"])
    _create_index_if_missing("ix_fine_tuned_models_deleted_at", "fine_tuned_models", ["deleted_at"])
    _create_index_if_missing("ix_fine_tuned_models_id", "fine_tuned_models", ["id"])
    _create_index_if_missing(
        "ix_fine_tuned_models_training_job_id", "fine_tuned_models", ["training_job_id"]
    )
    _create_index_if_missing("ix_fine_tuned_models_user_id", "fine_tuned_models", ["user_id"])

    # ------------------------------------------------------------------ #
    # Table: audit_logs                                                     #
    # ------------------------------------------------------------------ #
    safe_alter_column(
        "audit_logs",
        "action",
        existing_type=mysql.VARCHAR(length=32),
        comment="e.g. CREATE, UPDATE, DELETE, HARD_DELETE, ERASE",
        existing_nullable=False,
    )
    _create_fk_if_missing("audit_logs", "user_id", "users", "id", ondelete="SET NULL")

    # ------------------------------------------------------------------ #
    # Table: batfish_snapshots                                              #
    # ------------------------------------------------------------------ #
    _create_fk_if_missing("batfish_snapshots", "user_id", "users", "id", ondelete="CASCADE")

    # ------------------------------------------------------------------ #
    # Table: messages                                                       #
    # ------------------------------------------------------------------ #
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


# ---------------------------------------------------------------------------
# downgrade
# ---------------------------------------------------------------------------


def downgrade() -> None:
    """Downgrade schema safely."""

    # ------------------------------------------------------------------ #
    # Table: messages                                                       #
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # Table: batfish_snapshots                                              #
    # ------------------------------------------------------------------ #
    _drop_fk_by_columns_if_exists("batfish_snapshots", "user_id", "users", "id")

    # ------------------------------------------------------------------ #
    # Table: audit_logs                                                     #
    # ------------------------------------------------------------------ #
    _drop_fk_by_columns_if_exists("audit_logs", "user_id", "users", "id")
    safe_alter_column(
        "audit_logs",
        "action",
        existing_type=mysql.VARCHAR(length=32),
        comment="e.g. CREATE, UPDATE, DELETE, HARD_DELETE",
        existing_nullable=False,
    )

    # ------------------------------------------------------------------ #
    # Table: fine_tuned_models (indexes then table)                         #
    # ------------------------------------------------------------------ #
    _drop_index_if_exists("ix_fine_tuned_models_user_id", "fine_tuned_models")
    _drop_index_if_exists("ix_fine_tuned_models_training_job_id", "fine_tuned_models")
    _drop_index_if_exists("ix_fine_tuned_models_id", "fine_tuned_models")
    _drop_index_if_exists("ix_fine_tuned_models_deleted_at", "fine_tuned_models")
    _drop_index_if_exists("idx_finetunedmodel_user_id", "fine_tuned_models")
    _drop_index_if_exists("idx_finetunedmodel_status", "fine_tuned_models")
    _drop_index_if_exists("idx_finetunedmodel_is_active", "fine_tuned_models")
    if has_table("fine_tuned_models"):
        op.drop_table("fine_tuned_models")

    # ------------------------------------------------------------------ #
    # Table: training_jobs (indexes then table)                             #
    # ------------------------------------------------------------------ #
    _drop_index_if_exists("ix_training_jobs_user_id", "training_jobs")
    _drop_index_if_exists("ix_training_jobs_id", "training_jobs")
    _drop_index_if_exists("ix_training_jobs_dataset_id", "training_jobs")
    _drop_index_if_exists("idx_trainingjob_user_id", "training_jobs")
    _drop_index_if_exists("idx_trainingjob_status", "training_jobs")
    _drop_index_if_exists("idx_trainingjob_dataset_id", "training_jobs")
    if has_table("training_jobs"):
        op.drop_table("training_jobs")

    # ------------------------------------------------------------------ #
    # Table: datasets (indexes then table)                                  #
    # ------------------------------------------------------------------ #
    _drop_index_if_exists("ix_datasets_user_id", "datasets")
    _drop_index_if_exists("ix_datasets_id", "datasets")
    _drop_index_if_exists("ix_datasets_deleted_at", "datasets")
    _drop_index_if_exists("idx_dataset_user_id", "datasets")
    _drop_index_if_exists("idx_dataset_status", "datasets")
    if has_table("datasets"):
        op.drop_table("datasets")
