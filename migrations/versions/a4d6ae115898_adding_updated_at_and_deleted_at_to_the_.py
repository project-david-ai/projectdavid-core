"""adding updated_at and deleted_at to the TrainingJob

Revision ID: a4d6ae115898
Revises: 05cf57b50101
Create Date: 2026-03-20 05:40:54.930269

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

# Import your SafeDDL utilities
from migrations.utils.safe_ddl import (add_column_if_missing,
                                       create_index_if_missing,
                                       drop_column_if_exists,
                                       drop_index_if_exists)

# revision identifiers, used by Alembic.
revision: str = 'a4d6ae115898'
down_revision: Union[str, None] = '05cf57b50101'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema using SafeDDL helpers."""

    # 1. Add updated_at column
    # Using a server_default ensures existing rows don't violate NOT NULL
    add_column_if_missing(
        "training_jobs",
        sa.Column("updated_at", sa.BigInteger(), nullable=False, server_default="0"),
    )

    # 2. Add deleted_at column
    add_column_if_missing(
        "training_jobs",
        sa.Column(
            "deleted_at", sa.Integer(), nullable=True, comment="Unix timestamp of soft-deletion."
        ),
    )

    # 3. Add index for deleted_at
    create_index_if_missing("ix_training_jobs_deleted_at", "training_jobs", ["deleted_at"])


def downgrade() -> None:
    """Downgrade schema using SafeDDL helpers."""

    # 1. Drop index first
    drop_index_if_exists("ix_training_jobs_deleted_at", "training_jobs")

    # 2. Drop columns
    drop_column_if_exists("training_jobs", "deleted_at")
    drop_column_if_exists("training_jobs", "updated_at")
