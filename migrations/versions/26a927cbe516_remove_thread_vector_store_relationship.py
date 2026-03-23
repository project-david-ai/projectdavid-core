"""Remove thread ---> vector_store relationship

Revision ID: 26a927cbe516
Revises: 3a42e4f129e4
Create Date: 2026-03-08 05:34:08.601782

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

from migrations.utils.safe_ddl import create_fk_if_not_exists, has_table, safe_alter_column

# revision identifiers, used by Alembic.
revision: str = "26a927cbe516"
down_revision: Union[str, None] = "3a42e4f129e4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    if has_table("thread_vector_stores"):
        op.drop_table("thread_vector_stores")

    if has_table("vector_store_assistants"):
        op.drop_table("vector_store_assistants")

    safe_alter_column(
        table_name="messages", column_name="content", existing_type=mysql.TEXT(), nullable=False
    )
    safe_alter_column(
        table_name="messages",
        column_name="reasoning",
        existing_type=mysql.LONGTEXT(),
        type_=sa.Text(length=4294967295),
        existing_comment="Stores the internal 'thinking' or reasoning tokens from the model.",
        existing_nullable=True,
    )


def downgrade() -> None:
    """Downgrade schema."""
    safe_alter_column(
        table_name="messages",
        column_name="reasoning",
        existing_type=sa.Text(length=4294967295),
        type_=mysql.LONGTEXT(),
        existing_comment="Stores the internal 'thinking' or reasoning tokens from the model.",
        existing_nullable=True,
    )
    safe_alter_column(
        table_name="messages", column_name="content", existing_type=mysql.TEXT(), nullable=True
    )

    # Recreate join tables without inline FKs.
    # 'assistants', 'vector_stores', and 'threads' are base tables that may not
    # exist yet on a fresh container at this point in the downgrade chain.
    # FKs are added as separate deferred steps via create_fk_if_not_exists below,
    # which checks has_table on both source and referent before acting.
    if not has_table("vector_store_assistants"):
        op.create_table(
            "vector_store_assistants",
            sa.Column("vector_store_id", mysql.VARCHAR(length=64), nullable=False),
            sa.Column("assistant_id", mysql.VARCHAR(length=64), nullable=False),
            sa.PrimaryKeyConstraint("vector_store_id", "assistant_id"),
            mysql_collate="utf8mb4_0900_ai_ci",
            mysql_default_charset="utf8mb4",
            mysql_engine="InnoDB",
        )

    create_fk_if_not_exists(
        "vector_store_assistants_ibfk_1",
        "vector_store_assistants",
        "vector_stores",
        ["vector_store_id"],
        ["id"],
    )
    create_fk_if_not_exists(
        "vector_store_assistants_ibfk_2",
        "vector_store_assistants",
        "assistants",
        ["assistant_id"],
        ["id"],
    )

    if not has_table("thread_vector_stores"):
        op.create_table(
            "thread_vector_stores",
            sa.Column("thread_id", mysql.VARCHAR(length=64), nullable=False),
            sa.Column("vector_store_id", mysql.VARCHAR(length=64), nullable=False),
            sa.PrimaryKeyConstraint("thread_id", "vector_store_id"),
            mysql_collate="utf8mb4_0900_ai_ci",
            mysql_default_charset="utf8mb4",
            mysql_engine="InnoDB",
        )

    create_fk_if_not_exists(
        "thread_vector_stores_ibfk_1",
        "thread_vector_stores",
        "threads",
        ["thread_id"],
        ["id"],
    )
    create_fk_if_not_exists(
        "thread_vector_stores_ibfk_2",
        "thread_vector_stores",
        "vector_stores",
        ["vector_store_id"],
        ["id"],
    )
