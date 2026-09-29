import os
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# tests/conftest.py
#
# Stubs that prevent unit tests from needing a live DATABASE_URL, Redis,
# or any other infrastructure at import time.
#
# WHY MODULE-LEVEL PATCHING:
#   database.py calls create_engine() at module level with MySQL-specific
#   pool kwargs (max_overflow, pool_timeout).  pytest collects test files
#   — triggering those imports — BEFORE any fixture runs.  Fixtures are
#   therefore too late.  We call patch.start() here at module level so
#   the mock is active from the moment conftest.py is loaded, which is
#   always before collection begins.
# ---------------------------------------------------------------------------

# ── Step 1: env vars ────────────────────────────────────────────────────────
os.environ.setdefault("DATABASE_URL", "mysql+pymysql://test:test@localhost/test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("SECRET_KEY", "test-secret-key")
os.environ.setdefault("DEFAULT_SECRET_KEY", "test-default-secret-key")

# ── Step 2: patch create_engine before any src module is imported ───────────
_mock_engine = MagicMock(name="MockEngine")
_mock_engine.connect.return_value.__enter__ = MagicMock(return_value=_mock_engine)
_mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

_create_engine_patcher = patch("sqlalchemy.create_engine", return_value=_mock_engine)
_create_engine_patcher.start()

_mock_session = MagicMock(name="MockSession")
_mock_session_local = MagicMock(name="MockSessionLocal", return_value=_mock_session)

_session_local_patcher = patch(
    "src.api.entities_api.db.database.SessionLocal",
    _mock_session_local,
    create=True,
)
try:
    _session_local_patcher.start()
except Exception:  # noqa: BLE001
    pass

# ── Step 3: patch Redis ──────────────────────────────────────────────────────
_mock_redis = MagicMock(name="MockRedis")

_redis_patcher = patch("redis.from_url", return_value=_mock_redis)
_redis_patcher.start()

_redis_class_patcher = patch("redis.Redis", return_value=_mock_redis)
_redis_class_patcher.start()


# ── Step 4: clean up at session end ─────────────────────────────────────────
def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    _create_engine_patcher.stop()
    _redis_patcher.stop()
    _redis_class_patcher.stop()
    try:
        _session_local_patcher.stop()
    except Exception:  # noqa: BLE001
        pass
