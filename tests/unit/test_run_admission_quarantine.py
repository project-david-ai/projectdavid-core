"""Run admission must serialize against durable account quarantine."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from sqlalchemy.engine import create_engine
from sqlalchemy.orm import sessionmaker

from src.api.entities_api.models.models import ApiKey, User
from src.api.entities_api.services import account_erasure, runs_service
from src.api.entities_api.services.erasure_state import (
    COMPLETE,
    PENDING,
    erasures,
    subject_hash,
)


@pytest.fixture
def admission_db(monkeypatch):
    engine = create_engine("sqlite:///:memory:")

    try:
        User.__table__.create(engine)
        ApiKey.__table__.create(engine)
        erasures.create(engine)
        factory = sessionmaker(engine)
        monkeypatch.setattr(runs_service, "SessionLocal", factory)
        monkeypatch.setattr(account_erasure, "SessionLocal", factory)

        with factory() as db:
            db.add_all([User(id="owner"), User(id="unrelated")])
            db.commit()

        yield factory
    finally:
        engine.dispose()


@pytest.mark.parametrize("is_protected_key", [False, True])
def test_pending_account_cannot_create_run(admission_db, monkeypatch, is_protected_key):
    assert account_erasure.begin_erasure("owner") == PENDING
    service = runs_service.RunService()
    thread_guard = Mock()
    monkeypatch.setattr(service, "_lock_thread_for_run_creation", thread_guard)

    with pytest.raises(HTTPException) as error:
        service.create_run(
            SimpleNamespace(thread_id="thread_test"),
            user_id="owner",
            is_protected_key=is_protected_key,
        )

    assert error.value.status_code == 409
    thread_guard.assert_not_called()


def test_completed_receipt_prevents_run_creation(admission_db, monkeypatch):
    # Defensive case: a receipt and a user row coexist unexpectedly.
    with admission_db() as db:
        db.execute(
            erasures.insert().values(
                subject_hash=subject_hash("owner"),
                state=COMPLETE,
                created_at=100,
                completed_at=200,
            )
        )
        db.commit()

    service = runs_service.RunService()
    thread_guard = Mock()
    monkeypatch.setattr(service, "_lock_thread_for_run_creation", thread_guard)

    with pytest.raises(HTTPException) as error:
        service.create_run(
            SimpleNamespace(thread_id="thread_test"),
            user_id="owner",
        )

    assert error.value.status_code == 409
    thread_guard.assert_not_called()


def test_missing_account_cannot_create_run(admission_db, monkeypatch):
    service = runs_service.RunService()
    thread_guard = Mock()
    monkeypatch.setattr(service, "_lock_thread_for_run_creation", thread_guard)

    with pytest.raises(HTTPException) as error:
        service.create_run(
            SimpleNamespace(thread_id="thread_test"),
            user_id="unknown",
        )

    assert error.value.status_code == 404
    thread_guard.assert_not_called()


def test_unrelated_account_reaches_thread_admission(admission_db, monkeypatch):
    assert account_erasure.begin_erasure("owner") == PENDING
    service = runs_service.RunService()

    # Stop at the existing thread boundary. The real database-backed
    # account guard must allow this unrelated user to reach that boundary.
    thread_guard = Mock(
        side_effect=HTTPException(status_code=404, detail="Thread not found")
    )
    monkeypatch.setattr(service, "_lock_thread_for_run_creation", thread_guard)

    with pytest.raises(HTTPException) as error:
        service.create_run(
            SimpleNamespace(thread_id="thread_test"),
            user_id="unrelated",
        )

    assert error.value.status_code == 404
    assert error.value.detail == "Thread not found"
    thread_guard.assert_called_once()
    args = thread_guard.call_args.args
    assert args[1:] == ("thread_test", "unrelated")
