"""convert timestamps to epoch time

Revision ID: ffbf9622eece
Revises: c8adf7bf8fa8
Create Date: 2025-08-18 16:30:56.469086
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

from migrations.utils.safe_ddl import (add_column_if_missing,
                                       drop_column_if_exists, has_column,
                                       has_table, safe_alter_column)

# revision identifiers, used by Alembic.
revision: str = "ffbf9622eece"
down_revision: Union[str, None] = "c8adf7bf8fa8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _column_is_integer(table: str, col: str) -> bool:
    """
    Return True if the DB column is already an integer-like type.
    Uses safe_ddl.has_table() for the existence check — consistent with the
    rest of the codebase and avoids the raw inspect(bind) pattern.
    """
    if not has_table(table):
        return False
    bind = op.get_bind()
    insp = inspect(bind)
    for c in insp.get_columns(table):
        if c["name"] == col:
            # Compare by SQLAlchemy type class name to avoid dialect quirks.
            # Matches Integer, BIGINT, INTEGER, etc. but not INTERVAL.
            tname = type(c["type"]).__name__.lower()
            return "int" in tname and "interval" not in tname
    return False


def _change_to_integer_if_needed(table: str, col: str, nullable: bool = True) -> None:
    """
    Convert a column to Integer only if it exists and is not already integer-like.
    Uses safe_alter_column underneath — safe on fresh containers.
    """
    if not has_table(table) or not has_column(table, col):
        return
    if _column_is_integer(table, col):
        print(f"[alembic.safe_ddl] ⏭️  Skip alter: {table}.{col} already integer")
        return
    # We don't pass existing_type to avoid mismatches across environments.
    safe_alter_column(table, col, type_=sa.Integer(), nullable=nullable)


# ─────────────────────────────────────────────────────────────────────────────
# Upgrade / Downgrade
# ─────────────────────────────────────────────────────────────────────────────


def upgrade() -> None:
    """Upgrade schema."""

    # messages.function_call → NOT NULL
    # safe_alter_column checks has_table internally — no outer guard needed.
    safe_alter_column("messages", "function_call", existing_type=sa.Text(), nullable=False)

    # runs.* timestamps → Integer epoch seconds (conditionally)
    # _change_to_integer_if_needed checks has_table and has_column internally.
    _change_to_integer_if_needed("runs", "cancelled_at", nullable=True)
    _change_to_integer_if_needed("runs", "completed_at", nullable=True)
    _change_to_integer_if_needed("runs", "failed_at", nullable=True)
    _change_to_integer_if_needed("runs", "started_at", nullable=True)

    # Drop columns — drop_column_if_exists checks has_table internally.
    drop_column_if_exists("runs", "temperature")
    drop_column_if_exists("runs", "top_p")


def downgrade() -> None:
    """Downgrade schema."""

    # Re-add dropped columns — add_column_if_missing checks has_table internally.
    add_column_if_missing("runs", sa.Column("temperature", sa.Integer(), nullable=True))
    add_column_if_missing("runs", sa.Column("top_p", sa.Integer(), nullable=True))

    # Convert timestamps back to DATETIME — safe_alter_column + has_column guard.
    # The has_column checks here are intentional: they protect against the edge
    # case where a column was absent before this migration ran (e.g. partial schema).
    for col in ("started_at", "failed_at", "completed_at", "cancelled_at"):
        if has_column("runs", col):
            safe_alter_column("runs", col, type_=sa.DateTime(), nullable=True)

    # messages.function_call → back to nullable
    # safe_alter_column checks has_table internally — no outer guard needed.
    safe_alter_column("messages", "function_call", existing_type=sa.Text(), nullable=True)
