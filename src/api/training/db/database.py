# src/api/training/db/database.py
import os
import time
from pathlib import Path

from projectdavid_common import UtilsInterface
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

logging_utility = UtilsInterface.LoggingUtility()

DATABASE_URL = os.getenv("DATABASE_URL")


def running_in_docker() -> bool:
    return os.getenv("RUNNING_IN_DOCKER") == "1" or Path("/.dockerenv").exists()


if not DATABASE_URL:
    raise ValueError("FATAL: DATABASE_URL environment variable is not set.")

engine = create_engine(
    DATABASE_URL,
    echo=False,  # set True locally if you want query logging
    pool_size=10,  # lighter than entities_api — training service has lower concurrency
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=280,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """
    FastAPI dependency — yields a transactional DB session.
    Session is guaranteed to be closed on exit.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _wait_for_engine(engine_to_check, db_name: str, retries: int = 30, delay: int = 3):
    host_hint = str(engine_to_check.url).split("@")[-1]
    logging_utility.info("Training service — waiting for database '%s'... [%s]", db_name, host_hint)
    for i in range(retries):
        try:
            with engine_to_check.connect() as conn:
                conn.execute(text("SELECT 1"))
            logging_utility.info("Training service — database '%s' is connected.", db_name)
            return
        except Exception as e:
            logging_utility.warning(
                "Attempt %d/%d: DB '%s' not ready. Error: %s", i + 1, retries, db_name, e
            )
            if i < retries - 1:
                time.sleep(delay)
            else:
                logging_utility.error(
                    "Could not connect to '%s' after %d attempts.", db_name, retries
                )
                raise


def wait_for_db():
    """Block until the training service DB connection is healthy."""
    _wait_for_engine(engine, "Training DB")
