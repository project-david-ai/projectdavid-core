"""Add soft delete to Files

Revision ID: 3e16915ae60f
Revises: 52be510eb9c8
Create Date: 2026-03-09 17:13:31.024726

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from migrations.utils.safe_ddl import (
    add_column_if_missing,
    create_index_if_missing,
    drop_column_if_exists,
    drop_index_if_exists,
)

# revision identifiers, used by Alembic.
revision: str = '3e16915ae60f'
down_revision: Union[str, None] = '52be510eb9c8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    add_column_if_missing(
        "files",
        sa.Column(
            "deleted_at",
            sa.Integer(),
            nullable=True,
            comment="Unix timestamp of soft-deletion. Non-null = file is in Recycle Bin.",
        ),
    )

    # Safe index creation — guards against missing table and duplicate index
    create_index_if_missing("ix_files_deleted_at", "files", ["deleted_at"])


def downgrade() -> None:
    # Safe index drop — guards against missing table and absent index
    drop_index_if_exists("ix_files_deleted_at", "files")

    drop_column_if_exists("files", "deleted_at")
