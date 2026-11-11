"""Index active assistant ownership, keyset ordering and selective filters."""

from alembic import op

revision = "eaf2f16ac073"
down_revision = "869efe5119ee"
branch_labels = None
depends_on = None

INDEXES = {
    "ix_asst_owner_active_created": ["owner_id", "deleted_at", "created_at", "id"],
    "ix_asst_owner_active_model": [
        "owner_id",
        "deleted_at",
        "model",
        "created_at",
        "id",
    ],
    "ix_asst_owner_active_mode": [
        "owner_id",
        "deleted_at",
        "agent_mode",
        "created_at",
        "id",
    ],
    "ix_asst_owner_active_name": ["owner_id", "deleted_at", "name"],
}


def upgrade():
    for name, columns in INDEXES.items():
        op.create_index(name, "assistants", columns)


def downgrade():
    for name in reversed(INDEXES):
        op.drop_index(name, table_name="assistants")
