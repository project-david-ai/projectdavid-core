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

"""
tests/conftest.py
─────────────────
Session-scoped stubs that prevent unit tests from needing a live
DATABASE_URL, Redis, or any other infrastructure at import time.

Problem:
  src/api/entities_api/db/database.py calls create_engine() at module
  level with MySQL-specific pool kwargs (max_overflow, pool_timeout).
  pytest loads test modules during collection — before any fixture runs —
  so by the time autouse fixtures are active, the bad create_engine()
  call has already executed and raised.

Solution:
  Patch sqlalchemy.create_engine at *module level* in conftest.py,
  immediately when conftest is loaded (which is before collection).
  This means the patch is active before any src import can reach
  database.py, regardless of pool kwargs or missing DATABASE_URL.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Step 1 — env vars first, in case anything reads them at import time
# ---------------------------------------------------------------------------

os.environ.setdefault("DATABASE_URL", "mysql+pymysql://test:test@localhost/test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("SECRET_KEY", "test-secret-key")
os.environ.setdefault("DEFAULT_SECRET_KEY", "test-default-secret-key")

# ---------------------------------------------------------------------------
# Step 2 — patch create_engine NOW, at module level, before pytest imports
#           any src package.  Fixtures run too late for collection-time
#           module-level calls.
# ---------------------------------------------------------------------------

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
# This patcher may fail if the module hasn't been imported yet — that's fine,
# the create_engine patch above will have already prevented the crash.
try:
    _session_local_patcher.start()
except Exception:
    pass

# ---------------------------------------------------------------------------
# Step 3 — patch Redis at module level for the same reason
# ---------------------------------------------------------------------------

_mock_redis = MagicMock(name="MockRedis")

_redis_patcher = patch("redis.from_url", return_value=_mock_redis)
_redis_patcher.start()

_redis_class_patcher = patch("redis.Redis", return_value=_mock_redis)
_redis_class_patcher.start()


# ---------------------------------------------------------------------------
# Step 4 — stop all patchers cleanly at the end of the session
# ---------------------------------------------------------------------------


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    _create_engine_patcher.stop()
    _redis_patcher.stop()
    _redis_class_patcher.stop()
    try:
        _session_local_patcher.stop()
    except Exception:
        pass
