"""
Add split model tables for Cluster, Resource Management, and Inference.

Revision ID: 81522f7beef3
Revises: a4d6ae115898
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# Import the safe_ddl helpers
from migrations.utils.safe_ddl import (add_column_if_missing,
                                       create_fk_if_not_exists,
                                       create_index_if_missing,
                                       drop_column_if_exists,
                                       drop_fk_if_exists, drop_index_if_exists,
                                       has_table)

# Optional: If you strictly enforce database ENUM constraints, import your Python Enum here.
# Otherwise, we default to String(64) in the migration to prevent standard Alembic type parsing issues.
# from your_app_path.models import StatusEnum

# revision identifiers, used by Alembic.
revision: str = '81522f7beef3'
down_revision: Union[str, None] = 'a4d6ae115898'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ---------------------------------------------------------------------------
    # 1. compute_nodes
    # ---------------------------------------------------------------------------
    if not has_table("compute_nodes"):
        op.create_table(
            "compute_nodes",
            sa.Column("id", sa.String(length=64), nullable=False),
            sa.Column("hostname", sa.String(length=128), nullable=False),
            sa.Column("ip_address", sa.String(length=45), nullable=True),
            sa.Column("gpu_model", sa.String(length=128), nullable=True),
            sa.Column("total_vram_gb", sa.Float(), nullable=True),
            sa.Column("current_vram_usage_gb", sa.Float(), default=0.0),
            sa.Column("status", sa.String(length=64), nullable=True),  # Or sa.Enum(StatusEnum)
            sa.Column("last_heartbeat", sa.BigInteger(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
    create_index_if_missing("ix_compute_nodes_id", "compute_nodes", ["id"])

    # ---------------------------------------------------------------------------
    # 2. base_models
    # ---------------------------------------------------------------------------
    if not has_table("base_models"):
        op.create_table(
            "base_models",
            sa.Column("id", sa.String(length=128), nullable=False),
            sa.Column("name", sa.String(length=128), nullable=False),
            sa.Column("family", sa.String(length=64), nullable=True),
            sa.Column("parameter_count", sa.String(length=32), nullable=True),
            sa.Column("is_multimodal", sa.Boolean(), default=False),
            sa.Column("created_at", sa.BigInteger(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
    create_index_if_missing("ix_base_models_id", "base_models", ["id"])

    # ---------------------------------------------------------------------------
    # 3. Add node_id linking to existing tables
    # ---------------------------------------------------------------------------
    # --> training_jobs
    add_column_if_missing(
        "training_jobs", sa.Column("node_id", sa.String(length=64), nullable=True)
    )
    create_fk_if_not_exists(
        "fk_training_jobs_node_id", "training_jobs", "compute_nodes", ["node_id"], ["id"]
    )

    # --> fine_tuned_models
    add_column_if_missing(
        "fine_tuned_models", sa.Column("node_id", sa.String(length=64), nullable=True)
    )
    create_fk_if_not_exists(
        "fk_fine_tuned_models_node_id", "fine_tuned_models", "compute_nodes", ["node_id"], ["id"]
    )

    # ---------------------------------------------------------------------------
    # 4. gpu_allocations
    # ---------------------------------------------------------------------------
    if not has_table("gpu_allocations"):
        op.create_table(
            "gpu_allocations",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("node_id", sa.String(length=64), nullable=True),
            sa.Column("job_id", sa.String(length=64), nullable=True),
            sa.Column("model_id", sa.String(length=64), nullable=True),
            sa.Column("vram_reserved_gb", sa.Float(), nullable=False),
            sa.Column("created_at", sa.BigInteger(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
    create_fk_if_not_exists(
        "fk_gpu_allocations_node_id",
        "gpu_allocations",
        "compute_nodes",
        ["node_id"],
        ["id"],
        ondelete="CASCADE",
    )
    create_fk_if_not_exists(
        "fk_gpu_allocations_job_id",
        "gpu_allocations",
        "training_jobs",
        ["job_id"],
        ["id"],
        ondelete="CASCADE",
    )
    create_fk_if_not_exists(
        "fk_gpu_allocations_model_id",
        "gpu_allocations",
        "fine_tuned_models",
        ["model_id"],
        ["id"],
        ondelete="CASCADE",
    )

    # ---------------------------------------------------------------------------
    # 5. inference_deployments
    # ---------------------------------------------------------------------------
    if not has_table("inference_deployments"):
        op.create_table(
            "inference_deployments",
            sa.Column("id", sa.String(length=64), nullable=False),
            sa.Column("node_id", sa.String(length=64), nullable=True),
            sa.Column("base_model_id", sa.String(length=128), nullable=True),
            sa.Column("fine_tuned_model_id", sa.String(length=64), nullable=True),
            sa.Column("port", sa.Integer(), nullable=True),
            sa.Column("status", sa.String(length=64), nullable=True),
            sa.Column("current_throughput", sa.Float(), nullable=True),
            sa.Column("last_seen", sa.BigInteger(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("node_id", "port", name="uq_node_port_deployment"),
        )
    create_index_if_missing("ix_inference_deployments_id", "inference_deployments", ["id"])

    create_fk_if_not_exists(
        "fk_inference_deployments_node_id",
        "inference_deployments",
        "compute_nodes",
        ["node_id"],
        ["id"],
    )
    create_fk_if_not_exists(
        "fk_inference_deployments_base_model_id",
        "inference_deployments",
        "base_models",
        ["base_model_id"],
        ["id"],
    )
    create_fk_if_not_exists(
        "fk_inference_deployments_ft_model_id",
        "inference_deployments",
        "fine_tuned_models",
        ["fine_tuned_model_id"],
        ["id"],
    )


def downgrade() -> None:
    # Revert 5. inference_deployments
    drop_fk_if_exists("inference_deployments", "fk_inference_deployments_ft_model_id")
    drop_fk_if_exists("inference_deployments", "fk_inference_deployments_base_model_id")
    drop_fk_if_exists("inference_deployments", "fk_inference_deployments_node_id")
    drop_index_if_exists("ix_inference_deployments_id", "inference_deployments")
    if has_table("inference_deployments"):
        op.drop_table("inference_deployments")

    # Revert 4. gpu_allocations
    drop_fk_if_exists("gpu_allocations", "fk_gpu_allocations_model_id")
    drop_fk_if_exists("gpu_allocations", "fk_gpu_allocations_job_id")
    drop_fk_if_exists("gpu_allocations", "fk_gpu_allocations_node_id")
    if has_table("gpu_allocations"):
        op.drop_table("gpu_allocations")

    # Revert 3. existing tables modified
    drop_fk_if_exists("fine_tuned_models", "fk_fine_tuned_models_node_id")
    drop_column_if_exists("fine_tuned_models", "node_id")

    drop_fk_if_exists("training_jobs", "fk_training_jobs_node_id")
    drop_column_if_exists("training_jobs", "node_id")

    # Revert 2. base_models
    drop_index_if_exists("ix_base_models_id", "base_models")
    if has_table("base_models"):
        op.drop_table("base_models")

    # Revert 1. compute_nodes
    drop_index_if_exists("ix_compute_nodes_id", "compute_nodes")
    if has_table("compute_nodes"):
        op.drop_table("compute_nodes")
