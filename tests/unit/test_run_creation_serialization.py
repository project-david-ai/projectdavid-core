import sys
import types
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from projectdavid_common.validation import StatusEnum

from src.api.entities_api.models.models import Run, Thread

# runs_service imports SessionLocal at module import time.
# This unit suite exercises only the pure locking helper, so keep
# production database bootstrap completely outside the test.
_database_stub = types.ModuleType("src.api.entities_api.db.database")
_database_stub.SessionLocal = None
sys.modules["src.api.entities_api.db.database"] = _database_stub

from src.api.entities_api.services.runs_service import (  # noqa: E402
    TERMINAL_RUN_STATUSES,
    RunService,
)


class _FakeQuery:
    def __init__(self, result):
        self.result = result
        self.locked = False

    def filter(self, *args, **kwargs):
        return self

    def order_by(self, *args, **kwargs):
        return self

    def with_for_update(self):
        self.locked = True
        return self

    def first(self):
        return self.result


class _FakeDb:
    def __init__(
        self,
        *,
        thread_result,
        run_result,
    ):
        self.thread_query = _FakeQuery(thread_result)
        self.run_query = _FakeQuery(run_result)
        self.query_calls = []

    def query(self, model):
        self.query_calls.append(model)

        if model is Thread:
            return self.thread_query

        if model is Run:
            return self.run_query

        raise AssertionError(f"Unexpected model: {model}")


def test_terminal_run_statuses_are_exact():
    assert TERMINAL_RUN_STATUSES == (
        StatusEnum.completed,
        StatusEnum.failed,
        StatusEnum.cancelled,
        StatusEnum.expired,
    )


def test_free_thread_is_locked_and_allowed():
    db = _FakeDb(
        thread_result=SimpleNamespace(id="thread_a"),
        run_result=None,
    )

    RunService._lock_thread_for_run_creation(
        db,
        "thread_a",
    )

    assert db.query_calls == [
        Thread,
        Run,
    ]

    assert db.thread_query.locked is True
    assert db.run_query.locked is False


def test_non_terminal_run_rejects_creation():
    db = _FakeDb(
        thread_result=SimpleNamespace(id="thread_a"),
        run_result=SimpleNamespace(
            id="run_active",
            status=StatusEnum.queued,
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        RunService._lock_thread_for_run_creation(
            db,
            "thread_a",
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Thread already has a non-terminal run."

    assert db.thread_query.locked is True
    assert db.run_query.locked is False


def test_missing_thread_is_rejected_before_run_query():
    db = _FakeDb(
        thread_result=None,
        run_result=None,
    )

    with pytest.raises(HTTPException) as exc_info:
        RunService._lock_thread_for_run_creation(
            db,
            "missing_thread",
        )

    assert exc_info.value.status_code == 404

    assert db.query_calls == [
        Thread,
    ]
