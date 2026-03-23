"""Remove training data from models.py

Revision ID: 66b1d150d350
Revises: 33111f6ac0b8
Create Date: 2026-03-17 18:29:59.630222

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

# SafeDDL helpers
from migrations.utils.safe_ddl import (
    create_index_if_missing,
    drop_fk_if_exists,
    drop_index_if_exists,
    has_table,
    safe_alter_column,
)

# revision identifiers, used by Alembic.
revision: str = '66b1d150d350'
down_revision: Union[str, None] = '33111f6ac0b8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema safely."""

    # --- Table: training_jobs ---
    # 1. Drop FKs first so MySQL doesn't block dropping the indexes
    drop_fk_if_exists('training_jobs', 'training_jobs_ibfk_1')
    drop_fk_if_exists('training_jobs', 'training_jobs_ibfk_2')

    # 2. Drop indexes
    drop_index_if_exists('idx_trainingjob_dataset_id', 'training_jobs')
    drop_index_if_exists('idx_trainingjob_status', 'training_jobs')
    drop_index_if_exists('idx_trainingjob_user_id', 'training_jobs')
    drop_index_if_exists('ix_training_jobs_dataset_id', 'training_jobs')
    drop_index_if_exists('ix_training_jobs_id', 'training_jobs')
    drop_index_if_exists('ix_training_jobs_user_id', 'training_jobs')

    # 3. Drop table
    if has_table('training_jobs'):
        op.drop_table('training_jobs')

    # --- Table: datasets ---
    # 1. Drop FKs first
    drop_fk_if_exists('datasets', 'datasets_ibfk_1')

    # 2. Drop indexes
    drop_index_if_exists('idx_dataset_status', 'datasets')
    drop_index_if_exists('idx_dataset_user_id', 'datasets')
    drop_index_if_exists('ix_datasets_deleted_at', 'datasets')
    drop_index_if_exists('ix_datasets_id', 'datasets')
    drop_index_if_exists('ix_datasets_user_id', 'datasets')

    # 3. Drop table
    if has_table('datasets'):
        op.drop_table('datasets')

    # --- Table: file_storage ---
    safe_alter_column(
        'file_storage',
        'storage_path',
        existing_type=mysql.VARCHAR(length=512),
        nullable=True,
        existing_comment='Path to file in storage system (relative to share root)',
    )

    # --- Table: messages ---
    safe_alter_column('messages', 'content', existing_type=mysql.TEXT(), nullable=False)

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

    safe_alter_column('messages', 'content', existing_type=mysql.TEXT(), nullable=True)

    # --- Table: file_storage ---
    safe_alter_column(
        'file_storage',
        'storage_path',
        existing_type=mysql.VARCHAR(length=512),
        nullable=False,
        existing_comment='Path to file in storage system (relative to share root)',
    )

    # --- Table: datasets ---
    # Created FIRST because `training_jobs` holds a foreign key dependency to it.
    if not has_table('datasets'):
        op.create_table(
            'datasets',
            sa.Column(
                'id',
                mysql.VARCHAR(length=64),
                nullable=False,
                comment='Opaque dataset ID e.g. ds_abc123',
            ),
            sa.Column('user_id', mysql.VARCHAR(length=64), nullable=False),
            sa.Column('name', mysql.VARCHAR(length=128), nullable=False),
            sa.Column('description', mysql.TEXT(), nullable=True),
            sa.Column(
                'format',
                mysql.VARCHAR(length=32),
                nullable=False,
                comment='Training format: chatml | alpaca | sharegpt | jsonl',
            ),
            sa.Column(
                'storage_path',
                mysql.VARCHAR(length=512),
                nullable=True,
                comment='Path to the dataset file(s) on Samba, relative to the share root.',
            ),
            sa.Column('train_samples', mysql.INTEGER(), autoincrement=False, nullable=True),
            sa.Column('eval_samples', mysql.INTEGER(), autoincrement=False, nullable=True),
            sa.Column('config', mysql.JSON(), nullable=True),
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
                comment='pending → processing → active → failed',
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
            sa.Column(
                'file_id',
                mysql.VARCHAR(length=64),
                nullable=True,
                comment='Reference to the file_id in the core API files table.',
            ),
            sa.ForeignKeyConstraint(
                ['user_id'], ['users.id'], name='datasets_ibfk_1', ondelete='CASCADE'
            ),
            sa.PrimaryKeyConstraint('id'),
            mysql_collate='utf8mb4_0900_ai_ci',
            mysql_default_charset='utf8mb4',
            mysql_engine='InnoDB',
        )

    create_index_if_missing('ix_datasets_user_id', 'datasets', ['user_id'])
    create_index_if_missing('ix_datasets_id', 'datasets', ['id'])
    create_index_if_missing('ix_datasets_deleted_at', 'datasets', ['deleted_at'])
    create_index_if_missing('idx_dataset_user_id', 'datasets', ['user_id'])
    create_index_if_missing('idx_dataset_status', 'datasets', ['status'])

    # --- Table: training_jobs ---
    if not has_table('training_jobs'):
        op.create_table(
            'training_jobs',
            sa.Column(
                'id',
                mysql.VARCHAR(length=64),
                nullable=False,
                comment='Opaque job ID e.g. tj_abc123',
            ),
            sa.Column('user_id', mysql.VARCHAR(length=64), nullable=False),
            sa.Column(
                'dataset_id',
                mysql.VARCHAR(length=64),
                nullable=True,
                comment='Source dataset. SET NULL if dataset is deleted — job record is preserved.',
            ),
            sa.Column(
                'base_model',
                mysql.VARCHAR(length=256),
                nullable=False,
                comment='Base model identifier e.g. Qwen/Qwen2.5-7B-Instruct',
            ),
            sa.Column(
                'framework',
                mysql.VARCHAR(length=32),
                nullable=False,
                comment='Training framework: axolotl | unsloth',
            ),
            sa.Column(
                'config',
                mysql.JSON(),
                nullable=True,
                comment='Complete training configuration passed to the training container.',
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
                comment='queued → in_progress → completed | failed | cancelled',
            ),
            sa.Column('created_at', mysql.BIGINT(), autoincrement=False, nullable=False),
            sa.Column('started_at', mysql.BIGINT(), autoincrement=False, nullable=True),
            sa.Column('completed_at', mysql.BIGINT(), autoincrement=False, nullable=True),
            sa.Column('failed_at', mysql.BIGINT(), autoincrement=False, nullable=True),
            sa.Column('last_error', mysql.TEXT(), nullable=True),
            sa.Column(
                'metrics',
                mysql.JSON(),
                nullable=True,
                comment='Final training metrics: loss, eval_loss, perplexity etc.',
            ),
            sa.Column(
                'output_path',
                mysql.VARCHAR(length=512),
                nullable=True,
                comment='Samba path to the training output checkpoint.',
            ),
            sa.ForeignKeyConstraint(
                ['dataset_id'], ['datasets.id'], name='training_jobs_ibfk_1', ondelete='SET NULL'
            ),
            sa.ForeignKeyConstraint(
                ['user_id'], ['users.id'], name='training_jobs_ibfk_2', ondelete='CASCADE'
            ),
            sa.PrimaryKeyConstraint('id'),
            mysql_collate='utf8mb4_0900_ai_ci',
            mysql_default_charset='utf8mb4',
            mysql_engine='InnoDB',
        )

    create_index_if_missing('ix_training_jobs_user_id', 'training_jobs', ['user_id'])
    create_index_if_missing('ix_training_jobs_id', 'training_jobs', ['id'])
    create_index_if_missing('ix_training_jobs_dataset_id', 'training_jobs', ['dataset_id'])
    create_index_if_missing('idx_trainingjob_user_id', 'training_jobs', ['user_id'])
    create_index_if_missing('idx_trainingjob_status', 'training_jobs', ['status'])
    create_index_if_missing('idx_trainingjob_dataset_id', 'training_jobs', ['dataset_id'])
