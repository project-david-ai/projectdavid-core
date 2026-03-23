"""add deep_research to assistants

Revision ID: b697008df93a
Revises: 04607cbac68b
Create Date: 2026-02-11 19:11:17.619961

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

# Import the safe DDL helpers
from migrations.utils.safe_ddl import (add_column_if_missing,
                                       drop_column_if_exists,
                                       safe_alter_column)

# revision identifiers, used by Alembic.
revision: str = "b697008df93a"
down_revision: Union[str, None] = "04607cbac68b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema safely."""

    # --- Table: assistants ---

    # 1. Add the deep_research column
    add_column_if_missing(
        "assistants",
        sa.Column(
            "deep_research",
            sa.Boolean(),
            server_default=sa.text("0"),
            nullable=False,
            comment="Enable live web search and browsing capabilities.",
        ),
    )

    # --- Table: messages ---

    # 2. Alter 'content' to be non-nullable
    # Safe alter handles the modification if the column exists
    safe_alter_column("messages", "content", existing_type=mysql.TEXT(), nullable=False)

    # 3. Alter 'reasoning' type definition
    # Ensures the column uses the correct LongText length
    safe_alter_column(
        "messages",
        "reasoning",
        existing_type=mysql.LONGTEXT(),
        type_=sa.Text(length=4294967295),
        existing_comment="Stores the internal 'thinking' or reasoning tokens from the model.",
        existing_nullable=True,
    )


def downgrade() -> None:
    """Downgrade schema safely."""

    # --- Table: messages ---

    # 1. Revert 'reasoning' type definition
    safe_alter_column(
        "messages",
        "reasoning",
        existing_type=sa.Text(length=4294967295),
        type_=mysql.LONGTEXT(),
        existing_comment="Stores the internal 'thinking' or reasoning tokens from the model.",
        existing_nullable=True,
    )

    # 2. Revert 'content' to allow NULLs
    safe_alter_column("messages", "content", existing_type=mysql.TEXT(), nullable=True)

    # --- Table: assistants ---

    # 3. Drop the deep_research column
    drop_column_if_exists("assistants", "deep_research")
