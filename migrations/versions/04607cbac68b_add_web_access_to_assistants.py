"""add web_access to assistants

Revision ID: 04607cbac68b
Revises: 6413efed1b18
Create Date: 2026-02-10 02:26:44.817982
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

# SafeDDL helpers
from migrations.utils.safe_ddl import (add_column_if_missing,
                                       drop_column_if_exists,
                                       safe_alter_column)

# revision identifiers, used by Alembic.
revision: str = "04607cbac68b"
down_revision: Union[str, None] = "6413efed1b18"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema safely."""

    # --- Table: assistants ---
    add_column_if_missing(
        "assistants",
        sa.Column(
            "web_access",
            sa.Boolean(),
            nullable=False,
            server_default="0",
            comment="Enable live web search and browsing capabilities.",
        ),
    )

    safe_alter_column(
        "assistants",
        "agent_mode",
        existing_type=mysql.TINYINT(display_width=1),
        nullable=False,
        comment="False = Standard (Level 2), True = Autonomous (Level 3).",
    )

    safe_alter_column(
        "assistants",
        "decision_telemetry",
        existing_type=mysql.TINYINT(display_width=1),
        nullable=False,
        server_default=sa.text("'0'"),
        comment="If True, captures detailed reasoning payloads and confidence scores.",
    )

    # --- Table: messages ---
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

    # --- Table: assistants ---
    safe_alter_column(
        "assistants",
        "decision_telemetry",
        existing_type=mysql.TINYINT(display_width=1),
        nullable=False,
        server_default=sa.text("'0'"),
        comment="Flag to enable/disable detailed decision telemetry logging.",
    )

    safe_alter_column(
        "assistants",
        "agent_mode",
        existing_type=mysql.TINYINT(display_width=1),
        nullable=False,
        comment="Boolean flag indicating the mode of the agent.",
    )

    drop_column_if_exists("assistants", "web_access")
