import sqlalchemy as sa
from alembic import op

from migrations.utils.safe_ddl import has_table

revision = 'e844e0ceaba2'
down_revision = 'eed80604f05c'


def upgrade() -> None:
    # 1. USERS: The most important table
    if not has_table("users"):
        op.create_table(
            "users",
            sa.Column("id", sa.String(64), primary_key=True, index=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.Column("is_admin", sa.Boolean(), server_default="0", nullable=False),
            sa.Column("email", sa.String(length=255), unique=True, index=True, nullable=True),
            sa.Column("full_name", sa.String(length=255), nullable=True),
            sa.Column("oauth_provider", sa.String(length=50), index=True, nullable=True),
            sa.Column("provider_user_id", sa.String(length=255), index=True, nullable=True),
        )
        print("[alembic.safe_ddl] ✅ Created core table: users")

    # 2. ASSISTANTS
    if not has_table("assistants"):
        op.create_table(
            "assistants",
            sa.Column("id", sa.String(64), primary_key=True, index=True),
            sa.Column("name", sa.String(length=128), nullable=False),
            sa.Column("model", sa.String(length=64), nullable=True),
            sa.Column("created_at", sa.Integer(), nullable=False),
        )
        print("[alembic.safe_ddl] ✅ Created core table: assistants")

    # 3. THREADS
    if not has_table("threads"):
        op.create_table(
            "threads",
            sa.Column("id", sa.String(length=64), primary_key=True, index=True),
            sa.Column("created_at", sa.Integer(), nullable=False),
        )
        print("[alembic.safe_ddl] ✅ Created core table: threads")
