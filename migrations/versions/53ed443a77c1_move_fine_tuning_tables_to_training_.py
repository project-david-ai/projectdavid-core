"""Move fine tuning tables to training root instance of models.py

Revision ID: 53ed443a77c1
Revises: 005820173bc4
Create Date: 2026-03-17 06:30:20.358068
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

# SafeDDL helpers
from migrations.utils.safe_ddl import has_table, safe_alter_column

# revision identifiers, used by Alembic.
revision: str = '53ed443a77c1'
down_revision: Union[str, None] = '005820173bc4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema safely."""

    # --- Table: fine_tuned_models ---
    # Dropping the table automatically clears out all attached indexes in MySQL.
    if has_table('fine_tuned_models'):
        op.drop_table('fine_tuned_models')

    # --- Table: messages ---
    safe_alter_column(
        'messages',
        'content',
        existing_type=mysql.TEXT(),
        nullable=False,
    )

    safe_alter_column(
        'messages',
        'reasoning',
        existing_type=mysql.LONGTEXT(),
        type_=sa.Text(length=4294967295),
        existing_comment="Stores the internal 'thinking' or reasoning tokens from the model.",
        existing_nullable=True,
    )


def downgrade() -> None:
    """Downgrade schema safely."""

    # --- Table: messages ---
    safe_alter_column(
        'messages',
        'reasoning',
        existing_type=sa.Text(length=4294967295),
        type_=mysql.LONGTEXT(),
        existing_comment="Stores the internal 'thinking' or reasoning tokens from the model.",
        existing_nullable=True,
    )

    safe_alter_column(
        'messages',
        'content',
        existing_type=mysql.TEXT(),
        nullable=True,
    )

    # --- Table: fine_tuned_models ---
    if not has_table('fine_tuned_models'):
        op.create_table(
            'fine_tuned_models',
            sa.Column(
                'id',
                mysql.VARCHAR(length=64),
                nullable=False,
                comment='Opaque model ID e.g. ftm_abc123',
            ),
            sa.Column('user_id', mysql.VARCHAR(length=64), nullable=False),
            sa.Column(
                'training_job_id',
                mysql.VARCHAR(length=64),
                nullable=True,
                comment='Source training job. SET NULL if job is deleted — model record is preserved.',
            ),
            sa.Column('name', mysql.VARCHAR(length=128), nullable=False),
            sa.Column('description', mysql.TEXT(), nullable=True),
            sa.Column(
                'base_model',
                mysql.VARCHAR(length=256),
                nullable=False,
                comment='Base model this was fine-tuned from.',
            ),
            sa.Column(
                'hf_repo',
                mysql.VARCHAR(length=256),
                nullable=True,
                comment='HuggingFace repository ID e.g. your-org/your-model',
            ),
            sa.Column(
                'storage_path',
                mysql.VARCHAR(length=512),
                nullable=True,
                comment='Local Samba path to model weights (used when not pushed to HF).',
            ),
            sa.Column(
                'is_active',
                mysql.TINYINT(display_width=1),
                autoincrement=False,
                nullable=False,
                comment='True when this model is currently loaded in vLLM.',
            ),
            sa.Column(
                'vllm_model_id',
                mysql.VARCHAR(length=256),
                nullable=True,
                comment='The VLLM_MODEL value used to serve this model.',
            ),
            sa.Column(
                'status',
                mysql.ENUM(
                    'deleted',
                    'active',
                    'queued',
                    'in_progress',
                    'pending_action',
                    'completed',
                    'failed',
                    'cancelling',
                    'cancelled',
                    'pending',
                    'processing',
                    'expired',
                    'retrying',
                ),
                nullable=False,
                comment='processing → active → failed',
            ),
            sa.Column('created_at', mysql.BIGINT(), autoincrement=False, nullable=False),
            sa.Column('updated_at', mysql.BIGINT(), autoincrement=False, nullable=False),
            sa.Column(
                'deleted_at',
                mysql.INTEGER(),
                autoincrement=False,
                nullable=True,
                comment='Unix timestamp of soft-deletion.',
            ),
            sa.ForeignKeyConstraint(
                ['training_job_id'],
                ['training_jobs.id'],
                name=op.f('fine_tuned_models_ibfk_1'),
                ondelete='SET NULL',
            ),
            sa.ForeignKeyConstraint(
                ['user_id'], ['users.id'], name=op.f('fine_tuned_models_ibfk_2'), ondelete='CASCADE'
            ),
            sa.PrimaryKeyConstraint('id'),
            mysql_collate='utf8mb4_0900_ai_ci',
            mysql_default_charset='utf8mb4',
            mysql_engine='InnoDB',
        )

        op.create_index(
            op.f('ix_fine_tuned_models_user_id'), 'fine_tuned_models', ['user_id'], unique=False
        )
        op.create_index(
            op.f('ix_fine_tuned_models_training_job_id'),
            'fine_tuned_models',
            ['training_job_id'],
            unique=False,
        )
        op.create_index(op.f('ix_fine_tuned_models_id'), 'fine_tuned_models', ['id'], unique=False)
        op.create_index(
            op.f('ix_fine_tuned_models_deleted_at'),
            'fine_tuned_models',
            ['deleted_at'],
            unique=False,
        )
        op.create_index(
            op.f('idx_finetunedmodel_user_id'), 'fine_tuned_models', ['user_id'], unique=False
        )
        op.create_index(
            op.f('idx_finetunedmodel_status'), 'fine_tuned_models', ['status'], unique=False
        )
        op.create_index(
            op.f('idx_finetunedmodel_is_active'), 'fine_tuned_models', ['is_active'], unique=False
        )
