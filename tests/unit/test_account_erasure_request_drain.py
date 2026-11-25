"""An admitted HTTP request must prevent destructive account erasure."""

import asyncio
from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from projectdavid_common.validation import StatusEnum
from sqlalchemy.engine import create_engine
from sqlalchemy.orm import sessionmaker

from src.api.entities_api.models.models import ApiKey, Run, User
from src.api.entities_api.services import account_erasure
from src.api.entities_api.services.erasure_state import (
    COMPLETE,
    PENDING,
    erasure_state,
    erasures,
    subject_hash,
)
from src.api.entities_api.services.request_activity import (
    RequestActivityMiddleware,
    RequestActivityRegistry,
    track_authenticated_request,
)


@pytest.fixture
def drain_db(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    try:
        for model in (User, ApiKey, Run):
            model.__table__.create(engine)
        erasures.create(engine)
        factory = sessionmaker(engine)
        monkeypatch.setattr(account_erasure, "SessionLocal", factory)
        with factory() as db:
            db.add(User(id="owner"))
            db.commit()
        yield factory
    finally:
        engine.dispose()


async def _receive():
    return {"type": "http.disconnect"}


async def _send(message):
    pass


def _assert_pending(factory):
    with factory() as db:
        assert erasure_state(db, "owner") == PENDING
        assert db.get(User, "owner") is not None


def _add_nonterminal_run(factory):
    with factory() as db:
        db.add(
            Run(
                id="run_owner",
                user_id="owner",
                assistant_id="assistant_test",
                thread_id="thread_test",
                object="thread.run",
                status=StatusEnum.in_progress,
            )
        )
        db.commit()


def test_active_request_blocks_erasure_and_preserves_receipt(drain_db, monkeypatch):
    registry = RequestActivityRegistry()
    monkeypatch.setattr(account_erasure, "request_activity", registry)
    storage = Mock()

    # No Run is required to block erasure: stateless requests count too.
    async def app(scope, receive, send):
        assert track_authenticated_request("owner")
        with pytest.raises(HTTPException) as error:
            account_erasure.erase_user(storage, "owner")
        assert error.value.status_code == 409
        assert "active request" in error.value.detail
        _assert_pending(drain_db)

    asyncio.run(
        RequestActivityMiddleware(app, registry)({"type": "http"}, _receive, _send)
    )
    assert registry.active_count("owner") == 0
    storage._erase_physical_files.assert_not_called()
    storage._erase_vector_store_collections.assert_not_called()
    _assert_pending(drain_db)

    # After request exit, retry must advance past HTTP admission but still
    # respect the independent nonterminal execution blocker.
    _add_nonterminal_run(drain_db)
    with pytest.raises(HTTPException) as error:
        account_erasure.erase_user(storage, "owner")
    assert error.value.status_code == 409
    assert "resource reconciliation" in error.value.detail
    storage._erase_physical_files.assert_not_called()
    _assert_pending(drain_db)


def test_unrelated_request_does_not_block_target_http_drain(drain_db, monkeypatch):
    registry = RequestActivityRegistry()
    monkeypatch.setattr(account_erasure, "request_activity", registry)
    storage = Mock()
    _add_nonterminal_run(drain_db)

    async def app(scope, receive, send):
        assert track_authenticated_request("unrelated")
        with pytest.raises(HTTPException) as error:
            account_erasure.erase_user(storage, "owner")
        assert error.value.status_code == 409
        assert "resource reconciliation" in error.value.detail
        assert registry.active_count("unrelated") == 1

    asyncio.run(
        RequestActivityMiddleware(app, registry)({"type": "http"}, _receive, _send)
    )
    storage._erase_physical_files.assert_not_called()
    _assert_pending(drain_db)


def test_completed_receipt_remains_idempotent(drain_db, monkeypatch):
    registry = RequestActivityRegistry()
    monkeypatch.setattr(account_erasure, "request_activity", registry)
    storage = Mock()
    with drain_db() as db:
        db.execute(
            erasures.insert().values(
                subject_hash=subject_hash("previously_erased"),
                state=COMPLETE,
                created_at=100,
                completed_at=200,
            )
        )
        db.commit()

    account_erasure.erase_user(storage, "previously_erased")
    storage._erase_physical_files.assert_not_called()
    storage._erase_vector_store_collections.assert_not_called()
    with drain_db() as db:
        assert erasure_state(db, "previously_erased") == COMPLETE
