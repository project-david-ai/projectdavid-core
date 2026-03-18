import os
import sys
from logging.config import fileConfig

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from projectdavid_orm.projectdavid_orm.base import Base

load_dotenv()

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Combine both metadata objects so autogenerate sees all tables
from sqlalchemy import MetaData

combined_metadata = MetaData()

for table in Base.metadata.tables.values():
    table.tometadata(combined_metadata)


target_metadata = combined_metadata

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
