"""Durable account erasure completion receipts and quarantine."""

import sqlalchemy as sa
from alembic import op

revision = "ef183ba079a1"
down_revision = "e371fe9ac009"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "account_erasures",
        sa.Column("subject_hash", sa.String(64), primary_key=True),
        sa.Column("state", sa.String(24), nullable=False),
        sa.Column("created_at", sa.BigInteger(), nullable=False),
        sa.Column("completed_at", sa.BigInteger(), nullable=True),
        mysql_engine="InnoDB",
    )


def downgrade():
    # Dropping receipts would turn confirmed erasures into ambiguous 404s and
    # dropping pending operations would restore access. Never do this silently.
    count = (
        op.get_bind().execute(sa.text("SELECT COUNT(*) FROM account_erasures")).scalar()
    )
    if count:
        raise RuntimeError("Cannot remove nonempty account erasure state")
    op.drop_table("account_erasures")
