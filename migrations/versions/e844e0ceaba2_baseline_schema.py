"""Omega Baseline - Master Foundation Rescue
Revision ID: e844e0ceaba2
Revises: eed80604f05c
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

from migrations.utils.safe_ddl import create_fk_if_not_exists, has_table

revision = 'e844e0ceaba2'
down_revision = 'eed80604f05c'


def upgrade() -> None:
    # 1. CORE SYSTEM
    if not has_table("users"):
        op.create_table(
            "users",
            sa.Column("id", sa.String(64), primary_key=True, index=True),
            sa.Column("email", sa.String(255), unique=True, index=True),
            sa.Column("is_admin", sa.Boolean(), server_default="0"),
            sa.Column("created_at", sa.DateTime()),
            sa.Column("updated_at", sa.DateTime()),
        )

    if not has_table("api_keys"):
        op.create_table(
            "api_keys",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("user_id", sa.String(64)),
            sa.Column("prefix", sa.String(8), unique=True, index=True),
            sa.Column("hashed_key", sa.String(255), unique=True),
            sa.Column("is_active", sa.Boolean(), default=True),
            sa.Column("created_at", sa.DateTime()),
        )

    # 2. AI & CORE OBJECTS
    if not has_table("assistants"):
        op.create_table(
            "assistants",
            sa.Column("id", sa.String(64), primary_key=True),
            sa.Column("name", sa.String(128), nullable=False),
            sa.Column("model", sa.String(64)),
            sa.Column("owner_id", sa.String(64)),
            sa.Column("created_at", sa.Integer()),
        )

    if not has_table("threads"):
        op.create_table(
            "threads",
            sa.Column("id", sa.String(64), primary_key=True),
            sa.Column("owner_id", sa.String(64)),
            sa.Column("created_at", sa.Integer()),
        )

    if not has_table("messages"):
        op.create_table(
            "messages",
            sa.Column("id", sa.String(64), primary_key=True),
            sa.Column("thread_id", sa.String(64), index=True),
            sa.Column("role", sa.String(32)),
            sa.Column("content", sa.Text(length=4294967295)),
        )

    if not has_table("runs"):
        op.create_table(
            "runs",
            sa.Column("id", sa.String(64), primary_key=True),
            sa.Column("user_id", sa.String(64)),
            sa.Column("thread_id", sa.String(64)),
            sa.Column("assistant_id", sa.String(64)),
            sa.Column("status", sa.String(32)),
            sa.Column("created_at", sa.Integer()),
        )

    # 3. STORAGE & VECTOR
    if not has_table("files"):
        op.create_table(
            "files",
            sa.Column("id", sa.String(64), primary_key=True),
            sa.Column("user_id", sa.String(64)),
            sa.Column("filename", sa.String(256)),
            sa.Column("purpose", sa.String(64)),
            sa.Column("created_at", sa.DateTime()),
        )

    if not has_table("file_storage"):
        op.create_table(
            "file_storage",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("file_id", sa.String(64)),
        )

    if not has_table("vector_stores"):
        op.create_table(
            "vector_stores",
            sa.Column("id", sa.String(64), primary_key=True),
            sa.Column("user_id", sa.String(64)),
            sa.Column("name", sa.String(128)),
            sa.Column("collection_name", sa.String(128), unique=True),
            sa.Column("status", sa.String(32)),
            sa.Column("created_at", sa.BigInteger()),
        )

    if not has_table("vector_store_files"):
        op.create_table(
            "vector_store_files",
            sa.Column("id", sa.String(64), primary_key=True),
            sa.Column("vector_store_id", sa.String(64)),
        )

    # 4. TRAINING ENGINE
    if not has_table("datasets"):
        op.create_table(
            "datasets",
            sa.Column("id", sa.String(64), primary_key=True),
            sa.Column("user_id", sa.String(64)),
            sa.Column("name", sa.String(128)),
            sa.Column("status", sa.String(32)),
        )

    if not has_table("training_jobs"):
        op.create_table(
            "training_jobs",
            sa.Column("id", sa.String(64), primary_key=True),
            sa.Column("user_id", sa.String(64)),
            sa.Column("status", sa.String(32)),
        )

    if not has_table("fine_tuned_models"):
        op.create_table(
            "fine_tuned_models",
            sa.Column("id", sa.String(64), primary_key=True),
            sa.Column("user_id", sa.String(64)),
            sa.Column("status", sa.String(32)),
        )

    # 5. INFRASTRUCTURE & LOGGING
    if not has_table("audit_logs"):
        op.create_table(
            "audit_logs",
            sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
            sa.Column("user_id", sa.String(64)),
            sa.Column("action", sa.String(32)),
            sa.Column("timestamp", sa.DateTime()),
        )

    if not has_table("actions"):
        op.create_table(
            "actions",
            sa.Column("id", sa.String(64), primary_key=True),
            sa.Column("run_id", sa.String(64)),
        )

    if not has_table("sandboxes"):
        op.create_table(
            "sandboxes",
            sa.Column("id", sa.String(64), primary_key=True),
            sa.Column("user_id", sa.String(64)),
        )

    if not has_table("batfish_snapshots"):
        op.create_table(
            "batfish_snapshots",
            sa.Column("id", sa.String(64), primary_key=True),
            sa.Column("user_id", sa.String(64)),
        )

    if not has_table("compute_nodes"):
        op.create_table(
            "compute_nodes",
            sa.Column("id", sa.String(64), primary_key=True),
            sa.Column("hostname", sa.String(128)),
            sa.Column("status", sa.String(32)),
        )

    if not has_table("gpu_allocations"):
        op.create_table(
            "gpu_allocations",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("node_id", sa.String(64)),
        )

    if not has_table("base_models"):
        op.create_table(
            "base_models",
            sa.Column("id", sa.String(128), primary_key=True),
            sa.Column("name", sa.String(128)),
        )

    if not has_table("inference_deployments"):
        op.create_table(
            "inference_deployments",
            sa.Column("id", sa.String(64), primary_key=True),
            sa.Column("base_model_id", sa.String(128)),
        )

    # 6. ASSOCIATIONS
    if not has_table("thread_participants"):
        op.create_table(
            "thread_participants",
            sa.Column("thread_id", sa.String(64), primary_key=True),
            sa.Column("user_id", sa.String(64), primary_key=True),
        )

    if not has_table("user_assistants"):
        op.create_table(
            "user_assistants",
            sa.Column("user_id", sa.String(64), primary_key=True),
            sa.Column("assistant_id", sa.String(64), primary_key=True),
        )

    # 7. CONSTRAINTS (Prevent 1824)
    create_fk_if_not_exists(
        "fk_apikey_u", "api_keys", "users", ["user_id"], ["id"], ondelete="CASCADE"
    )
    create_fk_if_not_exists("fk_run_u", "runs", "users", ["user_id"], ["id"], ondelete="CASCADE")

    print("[alembic.safe_ddl] ✅ OMEGA FOUNDATION COMPLETE. All 23 tables built.")


def downgrade() -> None:
    pass
