"""add_decision_telemetry_to_actions

Revision ID: 741d86dd5ac8
Revises: 4c257388838e
Create Date: 2026-01-31 15:04:25.699059

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

from migrations.utils.safe_ddl import (add_column_if_missing,
                                       create_index_if_missing,
                                       drop_column_if_exists,
                                       drop_index_if_exists, safe_alter_column)

# revision identifiers, used by Alembic.
revision: str = "741d86dd5ac8"
down_revision: Union[str, None] = "4c257388838e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema safely."""

    # --- Table: actions ---
    # All three operations below use safe_ddl helpers that check has_table
    # internally — no crash if the actions table doesn't exist yet on a
    # fresh container running migrations in order for the first time.

    # 1. Add the Decision Payload (JSON)
    add_column_if_missing(
        "actions",
        sa.Column(
            "decision_payload",
            sa.JSON(),
            nullable=True,
            comment="The full structured reasoning object (reason, policy, etc) preceding the tool call.",
        ),
    )

    # 2. Add the Confidence Score (Float)
    add_column_if_missing(
        "actions",
        sa.Column(
            "confidence_score",
            sa.Float(),
            nullable=True,
            comment="Extracted confidence score (0.0-1.0) to allow fast querying of 'uncertain' agent actions.",
        ),
    )

    # 3. Create index safely — guards against missing table AND duplicate index.
    # Replaces the previous raw inspector.get_indexes("actions") call which
    # crashed with NoSuchTableError on fresh databases.
    create_index_if_missing(
        "ix_actions_confidence_score",
        "actions",
        ["confidence_score"],
    )

    # --- Table: messages ---
    # 4. Enforce non-nullable content
    safe_alter_column("messages", "content", existing_type=mysql.TEXT(), nullable=False)

    # 5. Standardize reasoning type
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

    safe_alter_column("messages", "content", existing_type=mysql.TEXT(), nullable=True)

    # --- Table: actions ---
    drop_index_if_exists("ix_actions_confidence_score", "actions")
    drop_column_if_exists("actions", "confidence_score")
    drop_column_if_exists("actions", "decision_payload")
