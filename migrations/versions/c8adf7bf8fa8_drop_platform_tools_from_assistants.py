"""drop platform_tools from assistants

Revision ID: c8adf7bf8fa8
Revises: 0f2bbee38b41
Create Date: 2025‑06‑29 22:00:51.633891
"""

from typing import Sequence, Union

import sqlalchemy as sa

from migrations.utils.safe_ddl import (add_column_if_missing,
                                       drop_column_if_exists)

# Alembic identifiers
revision: str = "c8adf7bf8fa8"
down_revision: Union[str, None] = "0f2bbee38b41"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema (safe drop)."""
    drop_column_if_exists("assistants", "platform_tools")


def downgrade() -> None:
    """Downgrade schema (safe re‑add)."""
    add_column_if_missing(
        "assistants",
        sa.Column("platform_tools", sa.JSON(), nullable=True),
    )
