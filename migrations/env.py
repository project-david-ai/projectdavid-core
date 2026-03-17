import os
import sys
from logging.config import fileConfig

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import (Column, MetaData, String, Table, engine_from_config,
                        pool)

# --- PATH FIX ---
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

# --- MODELS IMPORT ---
from src.api.entities_api.models.models import Base
from src.api.training.models.models import Base as TrainingBase

# Stub the users table into the training metadata so FK resolution works.
# The real users table is owned by entities_api Base — this is just a
# reference anchor for Alembic's FK graph traversal.
if "users" not in TrainingBase.metadata.tables:
    Table(
        "users",
        TrainingBase.metadata,
        Column("id", String(64), primary_key=True),
        keep_existing=True,
    )

load_dotenv()

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = [Base.metadata, TrainingBase.metadata]

DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise ValueError("FATAL: DATABASE_URL environment variable is not set or empty.")

config.set_main_option("sqlalchemy.url", DB_URL)


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
