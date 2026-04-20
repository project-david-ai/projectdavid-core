"""add cancelled_at to training_jobs

Revision ID: f893ede7b752
Revises: 75a8f1e1f907
Create Date: 2026-04-19 20:11:24.379268

Adds cancelled_at terminal-state timestamp to training_jobs, matching the
existing pattern of started_at / completed_at / failed_at. Used by the
training worker cancellation flow to record when a job transitioned out
of in_progress due to a user-initiated cancel.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

from migrations.utils.safe_ddl import add_column_if_missing, drop_column_if_exists

# revision identifiers, used by Alembic.
revision: str = "f893ede7b752"
down_revision: Union[str, None] = "75a8f1e1f907"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "training_jobs"


def upgrade() -> None:
    add_column_if_missing(
        TABLE,
        sa.Column(
            "cancelled_at",
            sa.BigInteger(),
            nullable=True,
            comment="Unix timestamp when job cancellation was initiated. "
            "NULL for jobs that ended via completion or failure.",
        ),
    )


def downgrade() -> None:
    drop_column_if_exists(TABLE, "cancelled_at")
