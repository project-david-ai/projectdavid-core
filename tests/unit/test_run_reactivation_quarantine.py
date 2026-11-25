"""Quarantine blocks new Run transitions, not terminal reconciliation."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException
from projectdavid_common.validation import StatusEnum
from sqlalchemy.engine import create_engine
from sqlalchemy.orm import sessionmaker

from src.api.entities_api.models.models import ApiKey, Run, User
from src.api.entities_api.orchestration.engine.orchestrator_core import OrchestratorCore
from src.api.entities_api.services import account_erasure, runs_service
from src.api.entities_api.services.erasure_state import PENDING, erasures


@pytest.fixture
def reactivation_db(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    try:
        for model in (User, ApiKey, Run):
            model.__table__.create(engine)
        erasures.create(engine)
        factory = sessionmaker(engine)
        monkeypatch.setattr(runs_service, "SessionLocal", factory)
        monkeypatch.setattr(account_erasure, "SessionLocal", factory)
        # These tests exercise persisted transitions, not SDK serialization.
        monkeypatch.setattr(
            runs_service.RunService, "_to_read_model", lambda self, run: run.status
        )
        with factory() as db:
            db.add_all([User(id="owner"), User(id="unrelated")])
            db.flush()
            for run_id, owner_id in (
                ("run_owner", "owner"),
                ("run_unrelated", "unrelated"),
                ("run_ownerless", None),
            ):
                db.add(
                    Run(
                        id=run_id,
                        user_id=owner_id,
                        assistant_id="assistant_test",
                        thread_id="thread_test",
                        object="thread.run",
                        status=StatusEnum.in_progress,
                    )
                )
            db.commit()
        yield factory
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    "status",
    [
        StatusEnum.queued.value,
        StatusEnum.in_progress.value,
        StatusEnum.pending_action.value,
    ],
)
def test_pending_account_cannot_enter_nonterminal_status(reactivation_db, status):
    assert account_erasure.begin_erasure("owner") == PENDING
    # Reopening a terminal Run must also fail, including SDK continuations.
    with reactivation_db() as db:
        run = db.get(Run, "run_owner")
        run.status = StatusEnum.completed
        run.completed_at = 100
        db.commit()

    with pytest.raises(HTTPException) as error:
        runs_service.RunService().update_run_status("run_owner", status)

    assert error.value.status_code == 409
    with reactivation_db() as db:
        run = db.get(Run, "run_owner")
        assert run.status == StatusEnum.completed
        assert run.completed_at == 100


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled", "expired"])
def test_pending_account_can_reconcile_terminal_status(reactivation_db, status):
    assert account_erasure.begin_erasure("owner") == PENDING
    runs_service.RunService().update_run_status("run_owner", status)

    with reactivation_db() as db:
        assert db.get(Run, "run_owner").status == StatusEnum(status)


def test_unrelated_account_can_enter_nonterminal_status(reactivation_db):
    assert account_erasure.begin_erasure("owner") == PENDING
    runs_service.RunService().update_run_status(
        "run_unrelated", StatusEnum.pending_action.value
    )

    with reactivation_db() as db:
        assert db.get(Run, "run_unrelated").status == StatusEnum.pending_action
        assert db.get(Run, "run_owner").status == StatusEnum.in_progress


def test_ownerless_run_cannot_enter_nonterminal_status(reactivation_db):
    with pytest.raises(HTTPException) as error:
        runs_service.RunService().update_run_status("run_ownerless", "queued")

    assert error.value.status_code == 409
    with reactivation_db() as db:
        assert db.get(Run, "run_ownerless").status == StatusEnum.in_progress


@pytest.mark.parametrize(
    "admission_error",
    [
        HTTPException(409, "Account deletion is pending or complete."),
        HTTPException(404, "Run not found"),
        RuntimeError("Database unavailable"),
    ],
    ids=["quarantine", "missing-run", "database-unavailable"],
)
def test_orchestrator_does_not_dispatch_when_admission_fails(admission_error):
    async def update_status(run_id, status):
        if status == StatusEnum.in_progress.value:
            raise admission_error

    native = SimpleNamespace(
        retrieve_run=AsyncMock(return_value=SimpleNamespace(current_turn=0)),
        update_run_status=AsyncMock(side_effect=update_status),
        update_run_fields=AsyncMock(),
    )
    orchestrator = SimpleNamespace(
        assistant_id="assistant_test",
        _ensure_config_loaded=AsyncMock(),
        _native_exec=native,
        stream=Mock(),
    )

    async def invoke():
        return [
            json.loads(event)
            async for event in OrchestratorCore.process_conversation(
                orchestrator,
                thread_id="thread_test",
                message_id="message_test",
                run_id="run_owner",
                assistant_id="assistant_test",
                model="test-model",
            )
        ]

    events = asyncio.run(invoke())
    orchestrator.stream.assert_not_called()
    assert events and events[-1]["type"] == "error"
    assert native.update_run_status.await_args_list[0].args == (
        "run_owner",
        StatusEnum.in_progress.value,
    )
    # Only the existing error-reconciliation path may write fields.
    for call in native.update_run_fields.await_args_list:
        assert "current_turn" not in call.kwargs
        assert "started_at" not in call.kwargs
