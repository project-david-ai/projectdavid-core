"""add assistant is_protected

Revision ID: db4f8a3bb8e2
Revises: c48bd2a910ef
Create Date: 2026-11-21 15:02:15.384378

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "db4f8a3bb8e2"
down_revision: Union[str, None] = "c48bd2a910ef"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add protection flag to assistants."""
    op.add_column(
        "assistants",
        sa.Column(
            "is_protected",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )


def downgrade() -> None:
    """Remove protection flag from assistants."""
    op.drop_column("assistants", "is_protected")
