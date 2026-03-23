"""Add soft delete to VectorStore

Revision ID: ba35b4620058
Revises: 3e16915ae60f
Create Date: 2026-03-09 17:42:20.789627

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from migrations.utils.safe_ddl import (add_column_if_missing,
                                       create_index_if_missing,
                                       drop_column_if_exists,
                                       drop_index_if_exists)

# revision identifiers, used by Alembic.
revision: str = "ba35b4620058"
down_revision: Union[str, None] = "3e16915ae60f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    add_column_if_missing(
        "vector_stores",
        sa.Column(
            "deleted_at",
            sa.Integer(),
            nullable=True,
            comment="Unix timestamp of soft-deletion. Non-null = store is in Recycle Bin.",
        ),
    )

    create_index_if_missing(
        "ix_vector_stores_deleted_at",
        "vector_stores",
        ["deleted_at"],
    )


def downgrade() -> None:
    drop_index_if_exists("ix_vector_stores_deleted_at", "vector_stores")
    drop_column_if_exists("vector_stores", "deleted_at")
