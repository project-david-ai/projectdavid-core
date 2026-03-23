"""add_tool_name_to_actions

Revision ID: 4c257388838e
Revises: 3b439a3fa9f6
Create Date: 2026-01-26 14:40:14.590781

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

# Import the safe DDL helpers
from migrations.utils.safe_ddl import (
    add_column_if_missing,
    drop_column_if_exists,
    safe_alter_column,
)

# revision identifiers, used by Alembic.
revision: str = "4c257388838e"
down_revision: Union[str, None] = "3b439a3fa9f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema safely."""

    # --- Table: actions ---
    # 1. Add the missing tool_name column
    add_column_if_missing(
        "actions",
        sa.Column(
            "tool_name",
            sa.String(length=64),
            nullable=True,
            comment="The name of the function/tool to be executed.",
        ),
    )

    # 2. Update comments and enforce metadata standards for existing columns
    safe_alter_column(
        "actions",
        "tool_call_id",
        existing_type=mysql.VARCHAR(length=64),
        comment="The unique ID linking this action to a specific LLM tool request.",
        nullable=True,
    )

    safe_alter_column(
        "actions",
        "turn_index",
        existing_type=mysql.INTEGER(),
        comment="The iteration of the autonomous loop.",
        nullable=True,
    )

    # --- Table: messages ---
    # 3. Ensure content is non-nullable (Enforcing data integrity for the UI)
    safe_alter_column("messages", "content", existing_type=mysql.TEXT(), nullable=False)

    # 4. Standardize reasoning field across environments
    safe_alter_column(
        "messages",
        "reasoning",
        existing_type=mysql.LONGTEXT(),
        type_=sa.Text(length=4294967295),
        comment="Stores the internal 'thinking' or reasoning tokens from the model.",
        nullable=True,
    )


def downgrade() -> None:
    """Downgrade schema safely."""

    # --- Table: messages ---
    safe_alter_column(
        "messages",
        "reasoning",
        existing_type=sa.Text(length=4294967295),
        type_=mysql.LONGTEXT(),
        nullable=True,
    )

    safe_alter_column("messages", "content", existing_type=mysql.TEXT(), nullable=True)

    # --- Table: actions ---
    safe_alter_column(
        "actions",
        "turn_index",
        existing_type=mysql.INTEGER(),
        comment="The iteration of the autonomous loop in which this action was triggered.",
        nullable=True,
    )

    safe_alter_column(
        "actions",
        "tool_call_id",
        existing_type=mysql.VARCHAR(length=64),
        comment="The unique ID linking this action to a specific LLM tool request (native or generated).",
        nullable=True,
    )

    # Remove the column added in upgrade
    drop_column_if_exists("actions", "tool_name")
