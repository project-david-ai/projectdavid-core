"""add engineer column to assistants

Revision ID: ce0a8a7e9d41
Revises: dda6fd28f45c
Create Date: 2026-02-24 08:06:25.334076

"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import mysql

from migrations.utils.safe_ddl import (add_column_if_missing,
                                       drop_column_if_exists,
                                       safe_alter_column)

# revision identifiers, used by Alembic.
revision: str = "ce0a8a7e9d41"
down_revision: Union[str, None] = "dda6fd28f45c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""

    # 1. Add engineer column to assistants
    add_column_if_missing(
        "assistants",
        sa.Column(
            "engineer",
            sa.Boolean(),
            server_default="0",
            nullable=False,
            comment="Enable network engineering capabilities and inventory map access.",
        ),
    )

    # 2. Alter deep_research comment in assistants
    safe_alter_column(
        "assistants",
        "deep_research",
        existing_type=mysql.TINYINT(display_width=1),
        comment="Enable deep research capabilities.",
        existing_comment="Enable live web search and browsing capabilities.",
        existing_nullable=False,
        existing_server_default=sa.text("'0'"),
    )

    # 3. Make messages.content not nullable
    safe_alter_column("messages", "content", existing_type=mysql.TEXT(), nullable=False)

    # 4. Change messages.reasoning type to sa.Text(length=4294967295)
    safe_alter_column(
        "messages",
        "reasoning",
        existing_type=mysql.LONGTEXT(),
        type_=sa.Text(length=4294967295),
        existing_comment="Stores the internal 'thinking' or reasoning tokens from the model.",
        existing_nullable=True,
    )


def downgrade() -> None:
    """Downgrade schema."""

    # 1. Revert messages.reasoning type back to LONGTEXT
    safe_alter_column(
        "messages",
        "reasoning",
        existing_type=sa.Text(length=4294967295),
        type_=mysql.LONGTEXT(),
        existing_comment="Stores the internal 'thinking' or reasoning tokens from the model.",
        existing_nullable=True,
    )

    # 2. Revert messages.content to be nullable
    safe_alter_column("messages", "content", existing_type=mysql.TEXT(), nullable=True)

    # 3. Revert deep_research comment in assistants
    safe_alter_column(
        "assistants",
        "deep_research",
        existing_type=mysql.TINYINT(display_width=1),
        comment="Enable live web search and browsing capabilities.",
        existing_comment="Enable deep research capabilities.",
        existing_nullable=False,
        existing_server_default=sa.text("'0'"),
    )

    # 4. Drop engineer column from assistants
    drop_column_if_exists("assistants", "engineer")
