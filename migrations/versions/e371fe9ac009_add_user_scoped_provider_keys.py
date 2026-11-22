"""add user scoped provider keys

Revision ID: e371fe9ac009
Revises: db4f8a3bb8e2
Create Date: 2026-11-22 11:58:23.181982

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e371fe9ac009"
down_revision: Union[str, None] = "db4f8a3bb8e2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Required for the composite ownership foreign key.
    op.create_unique_constraint(
        "uq_credentials_id_owner",
        "credentials",
        ["id", "owner_id"],
    )

    op.create_table(
        "provider_keys",
        sa.Column("id", sa.String(64), nullable=False),
        sa.Column("owner_id", sa.String(64), nullable=False),
        sa.Column("credential_id", sa.String(64), nullable=False),
        sa.Column("provider", sa.String(64), nullable=False),
        sa.Column("name", sa.String(128), nullable=True),
        sa.Column("fingerprint", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("revoked_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["owner_id"],
            ["users.id"],
            ondelete="CASCADE",
        ),
        # Database-enforced same-user credential binding.
        sa.ForeignKeyConstraint(
            ["credential_id", "owner_id"],
            ["credentials.id", "credentials.owner_id"],
            name="fk_provider_keys_credential_owner",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "credential_id",
            name="uq_provider_keys_credential",
        ),
        # Prevent concurrent duplicate active registrations.
        sa.UniqueConstraint(
            "owner_id",
            "provider",
            "fingerprint",
            name="uq_provider_keys_owner_provider_fingerprint",
        ),
        sa.CheckConstraint(
            "(revoked_at IS NULL AND fingerprint IS NOT NULL) OR "
            "(revoked_at IS NOT NULL AND fingerprint IS NULL)",
            name="ck_provider_keys_revocation",
        ),
    )

    op.create_index(
        "idx_provider_keys_owner_provider",
        "provider_keys",
        ["owner_id", "provider"],
    )


def downgrade() -> None:
    op.drop_index(
        "idx_provider_keys_owner_provider",
        table_name="provider_keys",
    )

    op.drop_table("provider_keys")

    op.drop_constraint(
        "uq_credentials_id_owner",
        "credentials",
        type_="unique",
    )
