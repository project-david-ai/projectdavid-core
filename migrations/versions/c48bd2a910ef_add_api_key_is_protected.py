"""Add protection flag to API keys.

Revision ID: c48bd2a910ef
Revises: b14a6c9d2e70
"""

import sqlalchemy as sa
from alembic import op

revision = "c48bd2a910ef"
down_revision = "b14a6c9d2e70"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "api_keys",
        sa.Column(
            "is_protected",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("0"),
        ),
    )


def downgrade() -> None:
    connection = op.get_bind()

    protected_count = connection.execute(
        sa.text("SELECT COUNT(*) FROM api_keys WHERE is_protected = 1")
    ).scalar_one()

    if protected_count:
        raise RuntimeError("Cannot downgrade while protected API keys exist.")

    op.drop_column("api_keys", "is_protected")
