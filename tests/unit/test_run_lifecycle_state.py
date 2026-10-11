from __future__ import annotations

import asyncio
from types import SimpleNamespace

from projectdavid_common.validation import StatusEnum

import src.api.entities_api.services.runs_service as runs_module
from src.api.entities_api.orchestration.mixins.consumer_tool_handlers_mixin import (
    ConsumerToolHandlersMixin,
)
from src.api.entities_api.services.runs_service import RunService


def _run(status: StatusEnum):
    return SimpleNamespace(
        status=status,
        completed_at=None,
        failed_at=None,
        cancelled_at=None,
        last_error=None,
        incomplete_details=None,
    )


def test_completed_transition_stamps_terminal_epoch_once(monkeypatch):
    monkeypatch.setattr(
        runs_module.time,
        "time",
        lambda: 1234,
    )

    run = _run(StatusEnum.in_progress)

    RunService._apply_status_transition(
        run,
        StatusEnum.completed,
    )

    assert run.status == StatusEnum.completed
    assert run.completed_at == 1234
    assert run.failed_at is None
    assert run.cancelled_at is None

    monkeypatch.setattr(
        runs_module.time,
        "time",
        lambda: 9999,
    )

    RunService._apply_status_transition(
        run,
        StatusEnum.completed,
    )

    assert run.completed_at == 1234


def test_reentering_active_epoch_clears_old_terminal_metadata():
    run = _run(StatusEnum.completed)

    run.completed_at = 100
    run.failed_at = 200
    run.cancelled_at = 300
    run.last_error = "old error"
    run.incomplete_details = "old incomplete state"

    RunService._apply_status_transition(
        run,
        StatusEnum.in_progress,
    )

    assert run.status == StatusEnum.in_progress
    assert run.completed_at is None
    assert run.failed_at is None
    assert run.cancelled_at is None
    assert run.last_error is None
    assert run.incomplete_details is None


def test_failed_transition_keeps_failure_diagnostics(monkeypatch):
    monkeypatch.setattr(
        runs_module.time,
        "time",
        lambda: 4321,
    )

    run = _run(StatusEnum.in_progress)

    run.last_error = "provider exploded"
    run.incomplete_details = "turn budget exhausted"

    RunService._apply_status_transition(
        run,
        StatusEnum.failed,
    )

    assert run.status == StatusEnum.failed
    assert run.failed_at == 4321
    assert run.completed_at is None
    assert run.cancelled_at is None
    assert run.last_error == "provider exploded"
    assert run.incomplete_details == "turn budget exhausted"


def test_pending_action_is_not_terminal():
    run = _run(StatusEnum.in_progress)

    RunService._apply_status_transition(
        run,
        StatusEnum.pending_action,
    )

    assert run.status == StatusEnum.pending_action
    assert run.completed_at is None
    assert run.failed_at is None
    assert run.cancelled_at is None


class _NativeExec:
    def __init__(self):
        self.statuses = []

    async def update_run_status(
        self,
        run_id: str,
        status: str,
    ):
        self.statuses.append((run_id, status))


class _PersistenceHarness(ConsumerToolHandlersMixin):
    def __init__(self):
        self._native_exec = _NativeExec()
        self.saved = []

    def _save_assistant_message(
        self,
        thread_id,
        assistant_id,
        run_id,
        content="",
        *,
        is_error,
        forced_status=None,
    ):
        self.saved.append(
            (
                thread_id,
                assistant_id,
                run_id,
                content,
                is_error,
            )
        )


def test_assistant_persistence_does_not_implicitly_complete_run():
    harness = _PersistenceHarness()

    asyncio.run(
        harness.finalize_conversation(
            "tool-call envelope",
            "thread_1",
            "assistant_1",
            "run_1",
        )
    )

    assert harness.saved == [
        (
            "thread_1",
            "assistant_1",
            "run_1",
            "tool-call envelope",
            False,
        )
    ]

    assert harness._native_exec.statuses == []


def test_explicit_finalize_status_still_transitions():
    harness = _PersistenceHarness()

    asyncio.run(
        harness.finalize_conversation(
            "terminal output",
            "thread_1",
            "assistant_1",
            "run_1",
            final_status=StatusEnum.completed.value,
        )
    )

    assert harness._native_exec.statuses == [
        (
            "run_1",
            StatusEnum.completed.value,
        )
    ]
