"""add tenant-owned encrypted credential storage

Revision ID: a4d7e6b9c2f1
Revises: 8c1e9d4a5b72
Create Date: 2026-09-27

Auth-1A introduces the generic reversible secret-reference foundation.
Credential plaintext is never stored in this table.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from migrations.utils.safe_ddl import (
    create_index_if_not_exists,
    drop_fk_if_exists,
    drop_index_if_exists,
    drop_table_if_exists,
    has_table,
)

revision: str = "a4d7e6b9c2f1"
down_revision: str | None = "8c1e9d4a5b72"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_OWNER_FK_NAME = "fk_credentials_owner_id"


def upgrade() -> None:
    if not has_table("credentials"):
        op.create_table(
            "credentials",
            sa.Column(
                "id",
                sa.String(length=64),
                nullable=False,
            ),
            sa.Column(
                "owner_id",
                sa.String(length=64),
                nullable=False,
            ),
            sa.Column(
                "kind",
                sa.String(length=32),
                nullable=False,
            ),
            sa.Column(
                "encrypted_payload",
                sa.Text(),
                nullable=False,
            ),
            sa.Column(
                "encryption_version",
                sa.Integer(),
                server_default="1",
                nullable=False,
            ),
            sa.Column(
                "created_at",
                sa.DateTime(),
                server_default=sa.text("CURRENT_TIMESTAMP"),
                nullable=False,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(),
                server_default=sa.text("CURRENT_TIMESTAMP"),
                nullable=False,
            ),
            sa.ForeignKeyConstraint(
                ["owner_id"],
                ["users.id"],
                name=_OWNER_FK_NAME,
                ondelete="CASCADE",
            ),
            sa.PrimaryKeyConstraint("id"),
        )

    create_index_if_not_exists(
        op.f("ix_credentials_id"),
        "credentials",
        ["id"],
        unique=False,
    )

    create_index_if_not_exists(
        op.f("ix_credentials_owner_id"),
        "credentials",
        ["owner_id"],
        unique=False,
    )

    create_index_if_not_exists(
        "idx_credentials_owner_kind",
        "credentials",
        ["owner_id", "kind"],
        unique=False,
    )


def downgrade() -> None:
    drop_fk_if_exists(
        "credentials",
        _OWNER_FK_NAME,
    )

    drop_index_if_exists(
        "idx_credentials_owner_kind",
        "credentials",
    )

    drop_index_if_exists(
        "ix_credentials_owner_id",
        "credentials",
    )

    drop_index_if_exists(
        "ix_credentials_id",
        "credentials",
    )

    drop_table_if_exists("credentials")
