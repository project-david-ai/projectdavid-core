"""Add vLLM hyperparam columns to inference_deployments

Revision ID: a1b2c3d4e5f6
Revises: 02e27dca8262
Create Date: 2026-04-12

Adds per-deployment vLLM engine hyperparam columns to inference_deployments.
Env vars (VLLM_DEFAULT_*) remain as fallbacks when columns are NULL.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from migrations.utils.safe_ddl import add_column_if_missing, drop_column_if_exists

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "02e27dca8262"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "inference_deployments"


def upgrade() -> None:
    # --- Memory & Context ---
    add_column_if_missing(
        TABLE,
        sa.Column(
            "gpu_memory_utilization",
            sa.Float(),
            nullable=False,
            server_default="0.90",
            comment="Fraction of GPU VRAM vLLM may allocate. Overrides VLLM_DEFAULT_GPU_MEM_UTIL.",
        ),
    )
    add_column_if_missing(
        TABLE,
        sa.Column(
            "max_model_len",
            sa.Integer(),
            nullable=True,
            comment="Max sequence length in tokens. Overrides VLLM_DEFAULT_MAX_MODEL_LEN. None = env default.",
        ),
    )
    add_column_if_missing(
        TABLE,
        sa.Column(
            "max_num_seqs",
            sa.Integer(),
            nullable=True,
            comment="Max concurrent sequences. Critical for vision — each image eats slots. None = vLLM default.",
        ),
    )

    # --- Quantization & Precision ---
    add_column_if_missing(
        TABLE,
        sa.Column(
            "quantization",
            sa.String(32),
            nullable=True,
            comment="Quantization scheme: 'awq', 'awq_marlin', 'gptq', 'bitsandbytes', or None for full precision.",
        ),
    )
    add_column_if_missing(
        TABLE,
        sa.Column(
            "dtype",
            sa.String(16),
            nullable=True,
            comment="Compute dtype: 'float16', 'bfloat16', 'auto', or None to let vLLM decide.",
        ),
    )

    # --- Runtime Behaviour ---
    add_column_if_missing(
        TABLE,
        sa.Column(
            "enforce_eager",
            sa.Boolean(),
            nullable=False,
            server_default="0",
            comment="Disable CUDA graphs. Slower but useful for debugging OOM crashes.",
        ),
    )

    # --- Multimodal Limits ---
    add_column_if_missing(
        TABLE,
        sa.Column(
            "limit_mm_per_prompt",
            sa.JSON(),
            nullable=True,
            comment='Per-modality token cap per request. e.g. {"image": 2, "video": 0}. None = vLLM default.',
        ),
    )


def downgrade() -> None:
    drop_column_if_exists(TABLE, "limit_mm_per_prompt")
    drop_column_if_exists(TABLE, "enforce_eager")
    drop_column_if_exists(TABLE, "dtype")
    drop_column_if_exists(TABLE, "quantization")
    drop_column_if_exists(TABLE, "max_num_seqs")
    drop_column_if_exists(TABLE, "max_model_len")
    drop_column_if_exists(TABLE, "gpu_memory_utilization")
