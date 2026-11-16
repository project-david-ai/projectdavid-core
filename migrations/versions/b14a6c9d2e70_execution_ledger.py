"""Install durable execution-lineage and evidence tables.

Revision ID: b14a6c9d2e70
Revises: eaf2f16ac073

Reconstructed from the deployed MySQL DDL and execution-ledger ORM.
"""

import sqlalchemy as sa
from alembic import op

revision = "b14a6c9d2e70"
down_revision = "eaf2f16ac073"
branch_labels = None
depends_on = None


MYSQL = {
    "mysql_engine": "InnoDB",
    "mysql_charset": "utf8mb4",
    "mysql_collate": "utf8mb4_0900_ai_ci",
}

TABLES = (
    "run_execution_epochs",
    "execution_actors",
    "run_delegations",
    "execution_events",
    "evidence_retention_holds",
)


def _fk(columns, references, name):
    return sa.ForeignKeyConstraint(
        columns,
        references,
        name=name,
        ondelete="RESTRICT",
    )


def _indexes(table, definitions):
    for name, columns in definitions:
        op.create_index(name, table, list(columns))


def upgrade() -> None:
    # Existing DBs recorded at b14a6c9d2e70 skip this upgrade.
    # Refuse to recreate any existing ledger tables when unstamped.
    existing = set(sa.inspect(op.get_bind()).get_table_names())
    conflicts = sorted(existing.intersection(TABLES))

    if conflicts:
        raise RuntimeError(
            "Ledger tables exist without revision b14a6c9d2e70: "
            + ", ".join(conflicts)
            + ". Reconcile the database revision manually."
        )

    # 1. Run execution epochs
    op.create_table(
        "run_execution_epochs",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("run_id", sa.String(64), nullable=False),
        sa.Column("root_run_id", sa.String(64), nullable=False),
        sa.Column("owner_user_id", sa.String(64), nullable=False),
        sa.Column("epoch_number", sa.Integer(), nullable=False),
        sa.Column("state", sa.String(32), nullable=False),
        sa.Column("created_at", sa.BigInteger(), nullable=False),
        sa.Column("started_at", sa.BigInteger(), nullable=True),
        sa.Column("last_seen_at", sa.BigInteger(), nullable=True),
        sa.Column("ended_at", sa.BigInteger(), nullable=True),
        sa.Column("terminal_status", sa.String(32), nullable=True),
        sa.UniqueConstraint(
            "run_id",
            "epoch_number",
            name="uq_exec_epoch_run_number",
        ),
        _fk(["run_id"], ["runs.id"], "fk_exec_epoch_run"),
        _fk(
            ["root_run_id"],
            ["runs.id"],
            "fk_exec_epoch_root",
        ),
        **MYSQL,
    )

    _indexes(
        "run_execution_epochs",
        (
            (
                "ix_exec_epoch_root_created",
                ("root_run_id", "created_at"),
            ),
            (
                "ix_exec_epoch_owner_root",
                ("owner_user_id", "root_run_id"),
            ),
            (
                "ix_exec_epoch_state_seen",
                ("state", "last_seen_at"),
            ),
        ),
    )

    # 2. Execution actors
    op.create_table(
        "execution_actors",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("epoch_id", sa.String(64), nullable=False),
        sa.Column("slot", sa.String(48), nullable=False),
        sa.Column("execution_role", sa.String(48), nullable=False),
        sa.Column(
            "assistant_id_snapshot",
            sa.String(64),
            nullable=True,
        ),
        sa.Column(
            "model_snapshot",
            sa.String(128),
            nullable=True,
        ),
        sa.Column("ephemeral", sa.Boolean(), nullable=False),
        sa.Column("state", sa.String(32), nullable=False),
        sa.Column("reserved_at", sa.BigInteger(), nullable=False),
        sa.Column("activated_at", sa.BigInteger(), nullable=True),
        sa.Column("retired_at", sa.BigInteger(), nullable=True),
        sa.Column("last_seen_at", sa.BigInteger(), nullable=True),
        sa.UniqueConstraint(
            "epoch_id",
            "slot",
            name="uq_exec_actor_epoch_slot",
        ),
        _fk(
            ["epoch_id"],
            ["run_execution_epochs.id"],
            "fk_exec_actor_epoch",
        ),
        **MYSQL,
    )

    _indexes(
        "execution_actors",
        (
            (
                "ix_exec_actor_assistant",
                ("assistant_id_snapshot",),
            ),
            (
                "ix_exec_actor_state_seen",
                ("state", "last_seen_at"),
            ),
        ),
    )

    # 3. Delegation lineage
    op.create_table(
        "run_delegations",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "parent_epoch_id",
            sa.String(64),
            nullable=False,
        ),
        sa.Column(
            "child_epoch_id",
            sa.String(64),
            nullable=False,
        ),
        sa.Column(
            "parent_action_id",
            sa.String(64),
            nullable=False,
        ),
        sa.Column(
            "initiating_actor_id",
            sa.String(64),
            nullable=True,
        ),
        sa.Column("created_at", sa.BigInteger(), nullable=False),
        sa.UniqueConstraint(
            "child_epoch_id",
            name="uq_run_delegation_child_epoch",
        ),
        _fk(
            ["parent_epoch_id"],
            ["run_execution_epochs.id"],
            "fk_delegation_parent",
        ),
        _fk(
            ["child_epoch_id"],
            ["run_execution_epochs.id"],
            "fk_delegation_child",
        ),
        _fk(
            ["parent_action_id"],
            ["actions.id"],
            "fk_delegation_action",
        ),
        _fk(
            ["initiating_actor_id"],
            ["execution_actors.id"],
            "fk_delegation_actor",
        ),
        **MYSQL,
    )

    _indexes(
        "run_delegations",
        (
            (
                "ix_run_delegation_parent",
                ("parent_epoch_id",),
            ),
            (
                "ix_run_delegation_parent_action",
                ("parent_action_id",),
            ),
        ),
    )

    # InnoDB supplies fk_delegation_actor's supporting index.

    # 4. Immutable execution evidence metadata
    op.create_table(
        "execution_events",
        sa.Column(
            "id",
            sa.BigInteger(),
            primary_key=True,
            autoincrement=True,
        ),
        sa.Column("epoch_id", sa.String(64), nullable=False),
        sa.Column("actor_id", sa.String(64), nullable=True),
        sa.Column(
            "origin_run_id",
            sa.String(64),
            nullable=False,
        ),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column(
            "evidence_class",
            sa.String(24),
            nullable=False,
        ),
        sa.Column(
            "occurred_at_us",
            sa.BigInteger(),
            nullable=False,
        ),
        sa.Column(
            "recorded_at_us",
            sa.BigInteger(),
            nullable=False,
        ),
        sa.Column(
            "evidence_expires_at",
            sa.BigInteger(),
            nullable=True,
        ),
        sa.Column(
            "action_id_snapshot",
            sa.String(64),
            nullable=True,
        ),
        sa.Column(
            "tool_call_id_snapshot",
            sa.String(64),
            nullable=True,
        ),
        sa.Column("details", sa.JSON(), nullable=False),
        sa.Column(
            "evidence_ref",
            sa.String(255),
            nullable=True,
        ),
        sa.Column(
            "evidence_sha256",
            sa.String(64),
            nullable=True,
        ),
        sa.Column(
            "previous_digest",
            sa.String(64),
            nullable=True,
        ),
        sa.Column(
            "record_digest",
            sa.String(64),
            nullable=True,
        ),
        _fk(
            ["epoch_id"],
            ["run_execution_epochs.id"],
            "fk_exec_event_epoch",
        ),
        _fk(
            ["actor_id"],
            ["execution_actors.id"],
            "fk_exec_event_actor",
        ),
        _fk(
            ["origin_run_id"],
            ["runs.id"],
            "fk_exec_event_run",
        ),
        **MYSQL,
    )

    _indexes(
        "execution_events",
        (
            (
                "ix_exec_event_epoch_id",
                ("epoch_id", "id"),
            ),
            (
                "ix_exec_event_origin_time",
                ("origin_run_id", "occurred_at_us"),
            ),
            (
                "ix_exec_event_type_time",
                ("event_type", "occurred_at_us"),
            ),
        ),
    )

    # InnoDB supplies fk_exec_event_actor's supporting index.

    # 5. Evidence retention holds
    op.create_table(
        "evidence_retention_holds",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "root_run_id",
            sa.String(64),
            nullable=False,
        ),
        sa.Column(
            "case_reference",
            sa.String(128),
            nullable=False,
        ),
        sa.Column(
            "reason_code",
            sa.String(48),
            nullable=False,
        ),
        sa.Column(
            "authorised_by",
            sa.String(64),
            nullable=False,
        ),
        sa.Column("created_at", sa.BigInteger(), nullable=False),
        sa.Column("reviewed_at", sa.BigInteger(), nullable=True),
        sa.Column("expires_at", sa.BigInteger(), nullable=True),
        sa.Column("released_at", sa.BigInteger(), nullable=True),
        _fk(
            ["root_run_id"],
            ["runs.id"],
            "fk_evidence_hold_root",
        ),
        **MYSQL,
    )

    _indexes(
        "evidence_retention_holds",
        (
            (
                "ix_evidence_hold_root_released",
                ("root_run_id", "released_at"),
            ),
            (
                "ix_evidence_hold_case",
                ("case_reference",),
            ),
        ),
    )


def downgrade() -> None:
    # Destructive: never execute against the populated ledger.
    for table in reversed(TABLES):
        op.drop_table(table)
