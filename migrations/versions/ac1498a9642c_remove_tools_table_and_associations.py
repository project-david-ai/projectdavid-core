"""remove_tools_table_and_associations

Revision ID: ac1498a9642c
Revises: 9314d4058f78
Create Date: 2026-01-25 02:09:10.112629

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

from migrations.utils.safe_ddl import (
    add_column_if_missing,
    create_fk_if_not_exists,
    drop_column_if_exists,
    drop_fk_if_exists,
    has_table,
    safe_alter_column,
)

# revision identifiers, used by Alembic.
revision: str = "ac1498a9642c"
down_revision: Union[str, None] = "9314d4058f78"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema safely."""

    # 1. Drop Foreign Keys on 'actions' (if they exist)
    # drop_fk_if_exists checks has_table internally — safe on fresh containers.
    # Check both possible constraint names MySQL may have assigned.
    drop_fk_if_exists("actions", "actions_ibfk_2")
    drop_fk_if_exists("actions", "actions_tool_id_fkey")

    # 2. Drop Column 'tool_id' from 'actions'
    drop_column_if_exists("actions", "tool_id")

    # 3. Drop Association Table 'assistant_tools'
    if has_table("assistant_tools"):
        op.drop_table("assistant_tools")
        print("[Alembic-safeDDL] ✅ Dropped table: assistant_tools")
    else:
        print("[Alembic-safeDDL] ⚠️ Skipped – table not found: assistant_tools")

    # 4. Drop 'tools' table
    # (Dropping the table automatically drops 'ix_tools_id')
    if has_table("tools"):
        op.drop_table("tools")
        print("[Alembic-safeDDL] ✅ Dropped table: tools")
    else:
        print("[Alembic-safeDDL] ⚠️ Skipped – table not found: tools")

    # 5. Alter 'messages' columns
    safe_alter_column("messages", "function_call", existing_type=mysql.TEXT(), nullable=False)
    safe_alter_column(
        "messages",
        "reasoning",
        existing_type=mysql.LONGTEXT(),
        type_=sa.Text(length=4294967295),
        existing_comment="Stores the internal 'thinking' or reasoning tokens from the model.",
        nullable=True,
    )


def downgrade() -> None:
    """Downgrade schema safely."""

    # 1. Revert 'messages' columns
    safe_alter_column(
        "messages",
        "reasoning",
        existing_type=sa.Text(length=4294967295),
        type_=mysql.LONGTEXT(),
        existing_comment="Stores the internal 'thinking' or reasoning tokens from the model.",
        nullable=True,
    )
    safe_alter_column("messages", "function_call", existing_type=mysql.TEXT(), nullable=True)

    # 2. Recreate 'tools' table — no FKs on this table, safe to create inline.
    if not has_table("tools"):
        op.create_table(
            "tools",
            sa.Column("id", mysql.VARCHAR(length=64), nullable=False),
            sa.Column("name", mysql.VARCHAR(length=128), nullable=False),
            sa.Column("type", mysql.VARCHAR(length=64), nullable=False),
            sa.Column("function", mysql.JSON(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
            mysql_collate="utf8mb4_0900_ai_ci",
            mysql_default_charset="utf8mb4",
            mysql_engine="InnoDB",
        )
        op.create_index("ix_tools_id", "tools", ["id"], unique=False)
        print("[Alembic-safeDDL] ✅ Re-created table: tools")

    # 3. Recreate 'assistant_tools' without inline FKs.
    # 'assistants' is a base table that may not exist on a fresh container at
    # this point in the downgrade chain — FKs are deferred below.
    if not has_table("assistant_tools"):
        op.create_table(
            "assistant_tools",
            sa.Column("assistant_id", mysql.VARCHAR(length=64), nullable=True),
            sa.Column("tool_id", mysql.VARCHAR(length=64), nullable=True),
            mysql_collate="utf8mb4_0900_ai_ci",
            mysql_default_charset="utf8mb4",
            mysql_engine="InnoDB",
        )
        print("[Alembic-safeDDL] ✅ Re-created table: assistant_tools")

    # Deferred FK creation — checks has_table on both source and referent.
    # Safe no-op if either table is absent, idempotent if already present.
    create_fk_if_not_exists(
        "assistant_tools_ibfk_1",
        "assistant_tools",
        "assistants",
        ["assistant_id"],
        ["id"],
    )
    create_fk_if_not_exists(
        "assistant_tools_ibfk_2",
        "assistant_tools",
        "tools",
        ["tool_id"],
        ["id"],
    )

    # 4. Add 'tool_id' back to 'actions'
    add_column_if_missing("actions", sa.Column("tool_id", mysql.VARCHAR(length=64), nullable=True))

    # 5. Re-add FK to 'actions'
    create_fk_if_not_exists(
        "actions_ibfk_2",
        "actions",
        "tools",
        ["tool_id"],
        ["id"],
    )
