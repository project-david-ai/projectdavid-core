"""allow nullable inference gpu memory utilization

Revision ID: 371df74151b3
Revises: f893ede7b752
Create Date: 2026-08-27 07:34:24.969790

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "371df74151b3"
down_revision: Union[str, None] = "f893ede7b752"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_OLD_COMMENT = (
    "Fraction of GPU VRAM vLLM may allocate. " "Overrides VLLM_DEFAULT_GPU_MEM_UTIL."
)

_NEW_COMMENT = (
    "Fraction of GPU VRAM vLLM may allocate. "
    "None = fall back to VLLM_DEFAULT_GPU_MEM_UTIL."
)


def upgrade() -> None:
    """Allow runtime GPU-memory defaults to remain unset."""

    op.alter_column(
        "inference_deployments",
        "gpu_memory_utilization",
        existing_type=sa.Float(),
        existing_nullable=False,
        existing_server_default=sa.text("0.90"),
        existing_comment=_OLD_COMMENT,
        nullable=True,
        server_default=None,
        comment=_NEW_COMMENT,
    )


def downgrade() -> None:
    """Restore the historical mandatory 0.90 GPU-memory default."""

    # NULL is valid in the upgraded schema but cannot survive the old
    # NOT NULL contract. Preserve downgrade compatibility by converting
    # inherited/default values back to the historical 0.90 value.
    op.execute(
        sa.text(
            """
            UPDATE inference_deployments
            SET gpu_memory_utilization = 0.90
            WHERE gpu_memory_utilization IS NULL
            """
        )
    )

    op.alter_column(
        "inference_deployments",
        "gpu_memory_utilization",
        existing_type=sa.Float(),
        existing_nullable=True,
        existing_server_default=None,
        existing_comment=_NEW_COMMENT,
        nullable=False,
        server_default=sa.text("0.90"),
        comment=_OLD_COMMENT,
    )
