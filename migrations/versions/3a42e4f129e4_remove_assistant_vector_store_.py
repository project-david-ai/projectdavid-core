"""Remove assistant ---> vector_store relationship

Revision ID: 3a42e4f129e4
Revises: 1c9784351972
Create Date: 2026-03-08 02:36:07.356463

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

from migrations.utils.safe_ddl import (create_fk_if_not_exists, has_table,
                                       safe_alter_column)

# revision identifiers, used by Alembic.
revision: str = "3a42e4f129e4"
down_revision: Union[str, None] = "1c9784351972"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_JOIN_TABLE = "vector_store_assistants"


def upgrade() -> None:
    # ── 1. Drop the M2M join table if it still exists ──────────────────────
    if has_table(_JOIN_TABLE):
        op.drop_table(_JOIN_TABLE)

    # ── 2. messages.content → NOT NULL ─────────────────────────────────────
    safe_alter_column(
        "messages",
        "content",
        existing_type=mysql.TEXT(),
        nullable=False,
    )

    # ── 3. messages.reasoning → LONGTEXT (explicit length form) ────────────
    safe_alter_column(
        "messages",
        "reasoning",
        existing_type=mysql.LONGTEXT(),
        type_=sa.Text(length=4294967295),
        existing_comment="Stores the internal 'thinking' or reasoning tokens from the model.",
        existing_nullable=True,
    )


def downgrade() -> None:
    # ── 3. Restore messages.reasoning type ─────────────────────────────────
    safe_alter_column(
        "messages",
        "reasoning",
        existing_type=sa.Text(length=4294967295),
        type_=mysql.LONGTEXT(),
        existing_comment="Stores the internal 'thinking' or reasoning tokens from the model.",
        existing_nullable=True,
    )

    # ── 2. messages.content → nullable again ───────────────────────────────
    safe_alter_column(
        "messages",
        "content",
        existing_type=mysql.TEXT(),
        nullable=True,
    )

    # ── 1. Recreate the join table without inline FKs ──────────────────────
    # 'vector_stores' and 'assistants' are base tables that may not exist yet
    # on a fresh container at this point in the downgrade chain. Inline FK
    # declarations would cause MySQL error 1824. FKs are added separately via
    # create_fk_if_not_exists, which checks has_table on both tables first.
    if not has_table(_JOIN_TABLE):
        op.create_table(
            _JOIN_TABLE,
            sa.Column("vector_store_id", sa.String(64), primary_key=True, nullable=False),
            sa.Column("assistant_id", sa.String(64), primary_key=True, nullable=False),
        )

    create_fk_if_not_exists(
        "vector_store_assistants_ibfk_1",
        _JOIN_TABLE,
        "vector_stores",
        ["vector_store_id"],
        ["id"],
        ondelete="CASCADE",
    )
    create_fk_if_not_exists(
        "vector_store_assistants_ibfk_2",
        _JOIN_TABLE,
        "assistants",
        ["assistant_id"],
        ["id"],
        ondelete="CASCADE",
    )
