# src/api/entities_api/db/database.py
import os
import time
from pathlib import Path

from projectdavid_common import UtilsInterface
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

logging_utility = UtilsInterface.LoggingUtility()

# --- ALL ENGINE AND SESSION LOGIC IS NOW CENTRALIZED HERE ---

DATABASE_URL = os.getenv("DATABASE_URL")
SPECIAL_DB_URL = os.getenv("SPECIAL_DB_URL")


# Container-aware resolver logic is kept with the engine definitions
def running_in_docker() -> bool:
    return os.getenv("RUNNING_IN_DOCKER") == "1" or Path("/.dockerenv").exists()


def resolve_special_db_runtime_url(special_raw: str | None) -> str | None:
    if running_in_docker():
        return DATABASE_URL
    return special_raw or None


SPECIAL_DB_RUNTIME_URL = resolve_special_db_runtime_url(SPECIAL_DB_URL)

# 1. The ONE configured main engine for the entire application
engine = create_engine(
    DATABASE_URL,
    echo=True,
    pool_size=20,
    max_overflow=40,
    pool_timeout=30,
    pool_recycle=280,
)

# The separate, special-purpose engine
special_engine = (
    create_engine(
        SPECIAL_DB_RUNTIME_URL,
        echo=True,
        pool_size=10,
        max_overflow=20,
        pool_timeout=30,
        pool_recycle=280,
    )
    if SPECIAL_DB_RUNTIME_URL
    else None
)

# 2. The ONE session factory, bound to the correct engine
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# 3. The ONE authoritative dependency for getting a DB session
def get_db():
    """
    FastAPI dependency that yields a transactional DB session from our
    correctly configured engine. The session is guaranteed to be closed.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Optional: You can also move the wait logic here to keep all DB startup
# code together, which makes app.py even cleaner.


def _wait_for_engine(engine_to_check, db_name, logger, retries=30, delay=3):
    if not engine_to_check:
        return
    host_hint = str(engine_to_check.url).split("@")[-1]
    logger.info(f"Waiting for database '{db_name}' to be available... [{host_hint}]")
    for i in range(retries):
        try:
            with engine_to_check.connect() as connection:
                connection.execute(text("SELECT 1"))
            logger.info(f"✅ Database '{db_name}' is connected!")
            return
        except Exception as e:
            logger.warning(
                f"Attempt {i+1}/{retries}: DB '{db_name}' not ready. Error: {e}"
            )
            if i < retries - 1:
                time.sleep(delay)
            else:
                logger.error(
                    f"Could not connect to '{db_name}' after {retries} attempts."
                )
                raise


def wait_for_databases():
    if not DATABASE_URL:
        raise ValueError("FATAL: DATABASE_URL environment variable is not set.")
    _wait_for_engine(engine, "Main DB", logging_utility)
    if os.getenv("WAIT_FOR_SPECIAL_DB", "0") in ("1", "true", "True"):
        _wait_for_engine(special_engine, "Special DB", logging_utility)
