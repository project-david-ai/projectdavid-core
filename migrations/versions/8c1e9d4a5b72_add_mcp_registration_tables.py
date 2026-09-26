"""add MCP registration and assistant tool provenance tables

Revision ID: 8c1e9d4a5b72
Revises: 371df74151b3
Create Date: 2026-09-26

MCP-5A persistence foundation.

This migration intentionally stores no third-party credentials. Project David
does not yet have an approved reversible secret-reference facility.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from migrations.utils.safe_ddl import (
    create_index_if_not_exists,
    drop_index_if_exists,
    drop_table_if_exists,
    has_table,
)

revision: str = "8c1e9d4a5b72"
down_revision: str | None = "371df74151b3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    if not has_table("mcp_server_registrations"):
        op.create_table(
            "mcp_server_registrations",
            sa.Column("id", sa.String(length=64), nullable=False),
            sa.Column("owner_id", sa.String(length=64), nullable=False),
            sa.Column("name", sa.String(length=128), nullable=False),
            sa.Column("url", sa.Text(), nullable=False),
            sa.Column("normalized_url", sa.Text(), nullable=False),
            sa.Column(
                "identity_key",
                sa.String(length=64),
                nullable=False,
                comment=(
                    "Stable SHA-256 identity derived from transport + normalized URL. "
                    "Combined with owner_id to make registration idempotent."
                ),
            ),
            sa.Column(
                "transport",
                sa.String(length=32),
                server_default="streamable_http",
                nullable=False,
            ),
            sa.Column(
                "timeout_seconds",
                sa.Float(),
                server_default="30",
                nullable=False,
            ),
            sa.Column(
                "enabled",
                sa.Boolean(),
                server_default=sa.text("1"),
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
                ondelete="CASCADE",
            ),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "owner_id",
                "identity_key",
                name="uq_mcp_registration_owner_identity",
            ),
        )

    create_index_if_not_exists(
        op.f("ix_mcp_server_registrations_id"),
        "mcp_server_registrations",
        ["id"],
        unique=False,
    )
    create_index_if_not_exists(
        op.f("ix_mcp_server_registrations_owner_id"),
        "mcp_server_registrations",
        ["owner_id"],
        unique=False,
    )
    create_index_if_not_exists(
        "idx_mcp_registration_owner_enabled",
        "mcp_server_registrations",
        ["owner_id", "enabled"],
        unique=False,
    )

    if not has_table("assistant_mcp_tools"):
        op.create_table(
            "assistant_mcp_tools",
            sa.Column("id", sa.String(length=64), nullable=False),
            sa.Column("assistant_id", sa.String(length=64), nullable=False),
            sa.Column("registration_id", sa.String(length=64), nullable=False),
            sa.Column("remote_name", sa.String(length=255), nullable=False),
            sa.Column("canonical_id", sa.String(length=512), nullable=False),
            sa.Column("provider_name", sa.String(length=64), nullable=False),
            sa.Column(
                "enabled",
                sa.Boolean(),
                server_default=sa.text("1"),
                nullable=False,
            ),
            sa.Column(
                "created_at",
                sa.DateTime(),
                server_default=sa.text("CURRENT_TIMESTAMP"),
                nullable=False,
            ),
            sa.ForeignKeyConstraint(
                ["assistant_id"],
                ["assistants.id"],
                ondelete="CASCADE",
            ),
            sa.ForeignKeyConstraint(
                ["registration_id"],
                ["mcp_server_registrations.id"],
                ondelete="CASCADE",
            ),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "assistant_id",
                "registration_id",
                "remote_name",
                name="uq_assistant_mcp_remote_tool",
            ),
            sa.UniqueConstraint(
                "assistant_id",
                "provider_name",
                name="uq_assistant_mcp_provider_name",
            ),
        )

    create_index_if_not_exists(
        op.f("ix_assistant_mcp_tools_id"),
        "assistant_mcp_tools",
        ["id"],
        unique=False,
    )
    create_index_if_not_exists(
        op.f("ix_assistant_mcp_tools_assistant_id"),
        "assistant_mcp_tools",
        ["assistant_id"],
        unique=False,
    )
    create_index_if_not_exists(
        op.f("ix_assistant_mcp_tools_registration_id"),
        "assistant_mcp_tools",
        ["registration_id"],
        unique=False,
    )
    create_index_if_not_exists(
        "idx_assistant_mcp_registration",
        "assistant_mcp_tools",
        ["registration_id"],
        unique=False,
    )


def downgrade() -> None:
    drop_index_if_exists(
        "idx_assistant_mcp_registration",
        "assistant_mcp_tools",
    )
    drop_index_if_exists(
        "ix_assistant_mcp_tools_registration_id",
        "assistant_mcp_tools",
    )
    drop_index_if_exists(
        "ix_assistant_mcp_tools_assistant_id",
        "assistant_mcp_tools",
    )
    drop_index_if_exists(
        "ix_assistant_mcp_tools_id",
        "assistant_mcp_tools",
    )
    drop_table_if_exists("assistant_mcp_tools")

    drop_index_if_exists(
        "idx_mcp_registration_owner_enabled",
        "mcp_server_registrations",
    )
    drop_index_if_exists(
        "ix_mcp_server_registrations_owner_id",
        "mcp_server_registrations",
    )
    drop_index_if_exists(
        "ix_mcp_server_registrations_id",
        "mcp_server_registrations",
    )
    drop_table_if_exists("mcp_server_registrations")
