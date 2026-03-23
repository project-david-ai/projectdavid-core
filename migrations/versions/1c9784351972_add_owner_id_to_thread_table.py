"""add owner_id to Thread table

Revision ID: 1c9784351972
Revises: 1e55188b6b26
Create Date: 2026-03-07 21:32:27.033637

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

from migrations.utils.safe_ddl import (
    add_column_if_missing,
    create_index_if_missing,
    drop_column_if_exists,
    drop_index_if_exists,
    safe_alter_column,
)

# revision identifiers, used by Alembic.
revision: str = "1c9784351972"
down_revision: Union[str, None] = "1e55188b6b26"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── messages.content ────────────────────────────────────────────────────
    # Tighten nullability — content must always be present.
    safe_alter_column(
        "messages",
        "content",
        existing_type=mysql.TEXT(),
        nullable=False,
    )

    # ── messages.reasoning ──────────────────────────────────────────────────
    # Promote storage type to LONGTEXT (4 GiB) to accommodate large
    # reasoning/thinking payloads; column remains optional.
    safe_alter_column(
        "messages",
        "reasoning",
        existing_type=mysql.LONGTEXT(),
        type_=sa.Text(length=4294967295),
        existing_comment="Stores the internal 'thinking' or reasoning tokens from the model.",
        existing_nullable=True,
    )

    # ── threads.owner_id ────────────────────────────────────────────────────
    # Canonical ownership column — mirrors the pattern introduced for
    # assistants.owner_id. Nullable during the back-fill window; tighten
    # to NOT NULL once every existing thread row has been back-filled.
    add_column_if_missing(
        "threads",
        sa.Column(
            "owner_id",
            sa.String(length=64),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
            # index=True is intentionally omitted: it is a metadata-level hint
            # only honoured by create_all(), not by Alembic batch migrations.
            # The index is created explicitly below.
            comment="Canonical creator/owner of this thread. Used for row-level access control.",
        ),
    )

    # Create the index explicitly so it is reliably present on all environments,
    # including fresh containers where the table did not previously exist.
    create_index_if_missing(
        "ix_threads_owner_id",
        "threads",
        ["owner_id"],
    )


def downgrade() -> None:
    # ── threads.owner_id ────────────────────────────────────────────────────
    drop_index_if_exists("ix_threads_owner_id", "threads")
    drop_column_if_exists("threads", "owner_id")

    # ── messages.reasoning ──────────────────────────────────────────────────
    safe_alter_column(
        "messages",
        "reasoning",
        existing_type=sa.Text(length=4294967295),
        type_=mysql.LONGTEXT(),
        existing_comment="Stores the internal 'thinking' or reasoning tokens from the model.",
        existing_nullable=True,
    )

    # ── messages.content ────────────────────────────────────────────────────
    safe_alter_column(
        "messages",
        "content",
        existing_type=mysql.TEXT(),
        nullable=True,
    )
