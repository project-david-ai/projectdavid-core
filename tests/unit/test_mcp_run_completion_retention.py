import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from projectdavid_common.validation import StatusEnum

import src.api.entities_api.orchestration.mixins.consumer_tool_handlers_mixin as consumer_module
from src.api.entities_api.orchestration.mixins.consumer_tool_handlers_mixin import (
    ConsumerToolHandlersMixin,
)


class _DummyConsumer(ConsumerToolHandlersMixin):
    pass


def _consumer():
    obj = _DummyConsumer()

    # finalize_conversation/handle_error call this through asyncio.to_thread.
    obj._save_assistant_message = lambda *args, **kwargs: None

    return obj


def test_normal_completion_retains_run_affine_mcp_session(monkeypatch) -> None:
    manager = SimpleNamespace(
        close_run=AsyncMock(),
    )

    monkeypatch.setattr(
        consumer_module,
        "get_mcp_run_session_manager",
        lambda: manager,
    )

    obj = _consumer()

    asyncio.run(
        obj.finalize_conversation(
            assistant_reply="done",
            thread_id="thread_1",
            assistant_id="assistant_1",
            run_id="run_resumable",
        )
    )

    manager.close_run.assert_not_awaited()


def test_empty_normal_completion_retains_run_affine_mcp_session(
    monkeypatch,
) -> None:
    manager = SimpleNamespace(
        close_run=AsyncMock(),
    )

    monkeypatch.setattr(
        consumer_module,
        "get_mcp_run_session_manager",
        lambda: manager,
    )

    obj = _consumer()

    asyncio.run(
        obj.finalize_conversation(
            assistant_reply="",
            thread_id="thread_1",
            assistant_id="assistant_1",
            run_id="run_resumable",
        )
    )

    manager.close_run.assert_not_awaited()


def test_terminal_error_still_disposes_run_affine_mcp_session(
    monkeypatch,
) -> None:
    manager = SimpleNamespace(
        close_run=AsyncMock(),
    )

    monkeypatch.setattr(
        consumer_module,
        "get_mcp_run_session_manager",
        lambda: manager,
    )

    obj = _consumer()

    lifecycle = SimpleNamespace(
        update_run_status=AsyncMock(),
    )
    obj._native_exec = lifecycle

    asyncio.run(
        obj.handle_error(
            assistant_reply="boom",
            thread_id="thread_1",
            assistant_id="assistant_1",
            run_id="run_failed",
        )
    )

    lifecycle.update_run_status.assert_awaited_once_with(
        "run_failed",
        StatusEnum.failed.value,
    )

    manager.close_run.assert_awaited_once_with(
        "run_failed",
        cancel=True,
    )
