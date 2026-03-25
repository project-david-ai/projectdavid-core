# C:\Users\franc\PycharmProjects\projectdavid-core\migrations\env.py
import os
import sys
from logging.config import fileConfig

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool

# Ensure project root is in path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

load_dotenv()

from projectdavid_common.projectdavid_orm.base import Base
from projectdavid_orm.projectdavid_orm import models

# 2. EXTRACT METADATA DIRECTLY FROM A MODEL
# Since 'User' inherits from Base, User.metadata IS the registry we want.
# This bypasses any "duplicate Base instance" issues.
target_metadata = models.User.metadata

# --- DEBUG BLOCK ---
print("-" * 50)
print(f"DEBUG: Found {len(target_metadata.tables)} tables in extracted MetaData")
for t in target_metadata.tables.keys():
    print(f"  - {t}")
if len(target_metadata.tables) == 0:
    print("CRITICAL: STILL NO TABLES FOUND. Check ormInterface imports.")
print("-" * 50)
# -------------------

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise ValueError("FATAL: DATABASE_URL not set.")

config.set_main_option("sqlalchemy.url", DB_URL)

# ... (rest of the run_migrations functions stay exactly the same) ...


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
