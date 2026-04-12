"""add mm_processor_kwargs to inference_deployments

Revision ID: 75a8f1e1f907
Revises: a1b2c3d4e5f6
Create Date: 2026-04-12 14:43:42.860517
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

from migrations.utils.safe_ddl import (
    add_column_if_missing,
    drop_column_if_exists,
    safe_alter_column,
)

revision: str = "75a8f1e1f907"
down_revision: Union[str, None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── New column ────────────────────────────────────────────────────────────
    add_column_if_missing(
        "inference_deployments",
        sa.Column(
            "mm_processor_kwargs",
            sa.JSON(),
            nullable=True,
            comment=(
                "Processor-level kwargs passed to vLLM multimodal processor at engine init. "
                "Overrides family registry defaults (_VISION_FAMILY_CONFIGS) when set. "
                'Examples: {"min_pixels": 784, "max_pixels": 50176} for Qwen2.5-VL, '
                '{"num_crops": 4} for Phi-3.5-Vision. '
                "None = fall back to inference_worker.py family registry defaults."
            ),
        ),
    )

    # ── Comment-only updates (idempotent) ─────────────────────────────────────
    safe_alter_column(
        "inference_deployments",
        "tensor_parallel_size",
        existing_type=mysql.INTEGER(),
        comment="Number of GPUs for tensor parallelism. 1 = single GPU.",
        existing_nullable=False,
    )
    safe_alter_column(
        "inference_deployments",
        "limit_mm_per_prompt",
        existing_type=mysql.JSON(),
        comment='Per-modality token cap per request. e.g. {"image": 2, "video": 0}. None = family registry default.',
        existing_nullable=True,
    )

    # ── Type corrections carried forward from previous migration ──────────────
    safe_alter_column(
        "messages",
        "content",
        existing_type=mysql.LONGTEXT(),
        type_=sa.Text(length=4294967295),
        existing_nullable=False,
    )
    safe_alter_column(
        "messages",
        "reasoning",
        existing_type=mysql.LONGTEXT(),
        type_=sa.Text(length=4294967295),
        existing_comment="Stores the internal 'thinking' or reasoning tokens from the model.",
        existing_nullable=True,
    )
    safe_alter_column(
        "runs",
        "top_p",
        existing_type=mysql.INTEGER(),
        type_=sa.Float(),
        existing_nullable=True,
    )


def downgrade() -> None:
    # ── Reverse type corrections ──────────────────────────────────────────────
    safe_alter_column(
        "runs",
        "top_p",
        existing_type=sa.Float(),
        type_=mysql.INTEGER(),
        existing_nullable=True,
    )
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
        existing_type=sa.Text(length=4294967295),
        type_=mysql.LONGTEXT(),
        existing_nullable=False,
    )

    # ── Revert comment updates ────────────────────────────────────────────────
    safe_alter_column(
        "inference_deployments",
        "limit_mm_per_prompt",
        existing_type=mysql.JSON(),
        comment='Per-modality token cap per request. e.g. {"image": 2, "video": 0}. None = vLLM default.',
        existing_nullable=True,
    )
    safe_alter_column(
        "inference_deployments",
        "tensor_parallel_size",
        existing_type=mysql.INTEGER(),
        comment=None,
        existing_nullable=False,
    )

    # ── Drop new column ───────────────────────────────────────────────────────
    drop_column_if_exists("inference_deployments", "mm_processor_kwargs")
