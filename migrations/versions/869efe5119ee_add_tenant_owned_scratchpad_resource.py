"""add tenant-owned scratchpad resource

Revision ID: 869efe5119ee
Revises: 6f4c2d8a91b7
Create Date: 2026-09-29 09:32:27.317119

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "869efe5119ee"
down_revision: Union[str, None] = "6f4c2d8a91b7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the tenant-owned Scratchpad resource."""

    op.create_table(
        "scratchpads",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("owner_id", sa.String(length=64), nullable=False),
        sa.Column("thread_id", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("meta_data", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(
            ["owner_id"],
            ["users.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["thread_id"],
            ["threads.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "thread_id",
            name="uq_scratchpads_thread_id",
        ),
    )

    op.create_index(
        "idx_scratchpads_owner_id",
        "scratchpads",
        ["owner_id"],
        unique=False,
    )

    op.create_index(
        op.f("ix_scratchpads_id"),
        "scratchpads",
        ["id"],
        unique=False,
    )


def downgrade() -> None:
    """Remove the tenant-owned Scratchpad resource."""

    op.drop_table("scratchpads")
