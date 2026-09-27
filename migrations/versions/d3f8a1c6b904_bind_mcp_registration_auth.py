"""bind MCP registrations to encrypted credentials

Revision ID: d3f8a1c6b904
Revises: a4d7e6b9c2f1
Create Date: 2026-09-27

Auth-1B adds authentication metadata and an indirect credential reference.
No raw third-party credential material is stored on MCP registrations.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from migrations.utils.safe_ddl import (
    add_column_if_missing,
    create_index_if_not_exists,
    drop_column_if_exists,
    drop_fk_if_exists,
    drop_index_if_exists,
)

revision: str = "d3f8a1c6b904"
down_revision: str | None = "a4d7e6b9c2f1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_FK_NAME = "fk_mcp_registration_credential_id"


def _has_fk(table_name: str, constraint_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())

    return any(
        foreign_key.get("name") == constraint_name
        for foreign_key in inspector.get_foreign_keys(table_name)
    )


def upgrade() -> None:
    add_column_if_missing(
        "mcp_server_registrations",
        sa.Column(
            "auth_type",
            sa.String(length=32),
            server_default="none",
            nullable=False,
        ),
    )

    add_column_if_missing(
        "mcp_server_registrations",
        sa.Column(
            "credential_id",
            sa.String(length=64),
            nullable=True,
        ),
    )

    if not _has_fk(
        "mcp_server_registrations",
        _FK_NAME,
    ):
        op.create_foreign_key(
            _FK_NAME,
            "mcp_server_registrations",
            "credentials",
            ["credential_id"],
            ["id"],
            ondelete="RESTRICT",
        )

    create_index_if_not_exists(
        op.f("ix_mcp_server_registrations_credential_id"),
        "mcp_server_registrations",
        ["credential_id"],
        unique=False,
    )


def downgrade() -> None:
    drop_fk_if_exists(
        "mcp_server_registrations",
        _FK_NAME,
    )

    drop_index_if_exists(
        op.f("ix_mcp_server_registrations_credential_id"),
        "mcp_server_registrations",
    )

    drop_column_if_exists(
        "mcp_server_registrations",
        "credential_id",
    )

    drop_column_if_exists(
        "mcp_server_registrations",
        "auth_type",
    )
