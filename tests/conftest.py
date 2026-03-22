"""
tests/conftest.py
─────────────────
Session-scoped stubs that prevent unit tests from needing a live
DATABASE_URL, Redis, or any other infrastructure at import time.

Problem:
  src/api/entities_api/db/database.py calls create_engine() at module
  level.  The moment any test imports a module that transitively touches
  that file, SQLAlchemy raises:

      ArgumentError: Expected string or URL object, got None

  because DATABASE_URL is not set in the CI environment.

Solution:
  Use pytest's monkeypatch (session-scoped via autouse fixture) to:
    1. Inject a dummy DATABASE_URL into os.environ BEFORE any import
       can reach database.py.
    2. Replace create_engine / SessionLocal / get_db with no-op stubs
       so modules that cache those objects at import time also get safe
       values.
    3. Stub out Redis so workers and dependency modules don't need a
       live broker either.

All unit tests remain fully isolated — no DB, no Redis, no network.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# 1. Inject env vars before any src module is imported
#    (this module is loaded by pytest before test collection begins)
# ---------------------------------------------------------------------------

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("SECRET_KEY", "test-secret-key")
os.environ.setdefault("DEFAULT_SECRET_KEY", "test-default-secret-key")

# ---------------------------------------------------------------------------
# 2. Patch create_engine at the SQLAlchemy level so even cached imports
#    that already ran get a safe mock engine.
# ---------------------------------------------------------------------------

_mock_engine = MagicMock(name="MockEngine")
_mock_session_local = MagicMock(name="MockSessionLocal")
_mock_session = MagicMock(name="MockSession")
_mock_session_local.return_value = _mock_session


@pytest.fixture(autouse=True, scope="session")
def _stub_database():
    """
    Replaces the SQLAlchemy engine and session factory for the entire
    test session.  autouse=True means every test gets this for free.
    """
    with (
        patch("sqlalchemy.create_engine", return_value=_mock_engine),
        patch(
            "src.api.entities_api.db.database.engine",
            _mock_engine,
            create=True,
        ),
        patch(
            "src.api.entities_api.db.database.SessionLocal",
            _mock_session_local,
            create=True,
        ),
        patch(
            "src.api.entities_api.db.database.get_db",
            return_value=iter([_mock_session]),
            create=True,
        ),
    ):
        yield


# ---------------------------------------------------------------------------
# 3. Stub Redis so dependency modules that call get_redis_sync() at import
#    time don't fail either.
# ---------------------------------------------------------------------------

_mock_redis = MagicMock(name="MockRedis")


@pytest.fixture(autouse=True, scope="session")
def _stub_redis():
    with (
        patch("redis.from_url", return_value=_mock_redis),
        patch("redis.Redis", return_value=_mock_redis),
    ):
        yield
