"""add handlers + tool_resources

Revision ID: 0f2bbee38b41
Revises: e844e0ceaba2
Create Date: 2025-05-01 12:59:08.713259
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

from migrations.utils.safe_ddl import (add_column_if_missing,
                                       drop_column_if_exists,
                                       safe_alter_column)

# revision identifiers, used by Alembic.
revision: str = "0f2bbee38b41"
down_revision: Union[str, None] = "e844e0ceaba2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""

    add_column_if_missing(
        "assistants",
        sa.Column(
            "tool_resources",
            sa.JSON(),
            nullable=True,
            comment='Resource map keyed by tool type, e.g. {"file_search": {"vector_store_ids": ["vs_123","vs_456"]}}',
        ),
    )

    safe_alter_column(
        "assistants",
        "handlers",
        existing_type=mysql.JSON(),
        comment='Optional array of inline tool specs, e.g. [{"type": "file_search", "vector_store_ids": ["vs_123"]}]',
        existing_comment='Optional array of inline tool specs, e.g. [{"type": "file_search", "vector_store_ids": ["..."]}]',
        existing_nullable=True,
    )

    safe_alter_column(
        "messages",
        "function_call",
        existing_type=mysql.TEXT(),
        nullable=False,
    )


def downgrade() -> None:
    """Downgrade schema."""

    safe_alter_column(
        "messages",
        "function_call",
        existing_type=mysql.TEXT(),
        nullable=True,
    )

    safe_alter_column(
        "assistants",
        "handlers",
        existing_type=mysql.JSON(),
        comment='Optional array of inline tool specs, e.g. [{"type": "file_search", "vector_store_ids": ["..."]}]',
        existing_comment='Optional array of inline tool specs, e.g. [{"type": "file_search", "vector_store_ids": ["vs_123"]}]',
        existing_nullable=True,
    )

    # Replaced manual INFORMATION_SCHEMA fallback with safe_ddl helper
    drop_column_if_exists("assistants", "tool_resources")
