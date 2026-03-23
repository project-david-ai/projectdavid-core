"""Add file_id to datasets make storage_path nullable

Revision ID: 33111f6ac0b8
Revises: c6d0aaad984f
Create Date: 2026-03-17 18:00:44.521819
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

from migrations.utils.safe_ddl import (
    add_column_if_missing,
    drop_column_if_exists,
    safe_alter_column,
)

revision: str = "33111f6ac0b8"
down_revision: Union[str, None] = "c6d0aaad984f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema safely."""

    # --- Table: datasets ---
    add_column_if_missing(
        "datasets",
        sa.Column(
            "file_id",
            sa.String(length=64),
            nullable=True,
            comment="Reference to the file_id in the core API files table.",
        ),
    )

    safe_alter_column(
        "datasets",
        "storage_path",
        existing_type=mysql.VARCHAR(length=512),
        nullable=True,
        existing_comment="Path to the dataset file(s) on Samba, relative to the share root.",
    )

    # --- Table: messages ---
    # These are already applied by earlier migrations but safe_alter_column
    # is idempotent so repeated runs are harmless.
    safe_alter_column(
        "messages",
        "content",
        existing_type=mysql.TEXT(),
        nullable=False,
    )

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
    safe_alter_column(
        "messages",
        "reasoning",
        existing_type=sa.Text(length=4294967295),
        type_=mysql.LONGTEXT(),
        existing_comment="Stores the internal 'thinking' or reasoning tokens from the model.",
        existing_nullable=True,
    )

    safe_alter_column(
        "messages",
        "content",
        existing_type=mysql.TEXT(),
        nullable=True,
    )

    # --- Table: datasets ---
    safe_alter_column(
        "datasets",
        "storage_path",
        existing_type=mysql.VARCHAR(length=512),
        nullable=False,
        existing_comment="Path to the dataset file(s) on Samba, relative to the share root.",
    )

    drop_column_if_exists("datasets", "file_id")
