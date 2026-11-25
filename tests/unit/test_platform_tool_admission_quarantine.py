"""Platform tools cannot dispatch after failed Run admission."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException
from projectdavid_common import ValidationInterface

from src.api.entities_api.orchestration.mixins.platform_tool_handlers_mixin import (
    PlatformToolHandlersMixin,
)


@pytest.fixture
def handler():
    action = SimpleNamespace(id="action_test")
    return SimpleNamespace(
        set_assistant_id=Mock(),
        set_thread_id=Mock(),
        _native_exec=SimpleNamespace(
            update_run_status=AsyncMock(),
            create_action=AsyncMock(return_value=action),
        ),
        platform_tool_service=SimpleNamespace(
            call_function=Mock(return_value="tool result"),
        ),
        _handle_web_search=AsyncMock(),
        _handle_code_interpreter=AsyncMock(),
        _handle_computer=AsyncMock(),
        _submit_platform_tool_output=AsyncMock(),
    )


def invoke(handler):
    return asyncio.run(
        PlatformToolHandlersMixin._process_platform_tool_calls(
            handler,
            thread_id="thread_test",
            assistant_id="assistant_test",
            run_id="run_test",
            tool_call_id="call_test",
            content={"name": "web_search", "arguments": {"query": "test"}},
        )
    )


@pytest.mark.parametrize(
    "error",
    [
        HTTPException(409, "Account deletion is pending or complete."),
        HTTPException(404, "Run not found"),
        RuntimeError("Database unavailable"),
    ],
    ids=["quarantine", "missing-run", "database-unavailable"],
)
def test_failed_admission_prevents_action_and_dispatch(handler, error):
    handler._native_exec.update_run_status.side_effect = error

    with pytest.raises(type(error)) as caught:
        invoke(handler)

    assert caught.value is error
    handler._native_exec.update_run_status.assert_awaited_once_with(
        "run_test", ValidationInterface.StatusEnum.pending_action.value
    )
    handler._native_exec.create_action.assert_not_called()
    handler.platform_tool_service.call_function.assert_not_called()
    handler._handle_web_search.assert_not_called()
    handler._submit_platform_tool_output.assert_not_called()


def test_action_creation_failure_prevents_dispatch(handler):
    handler._native_exec.create_action.side_effect = RuntimeError("Action unavailable")

    with pytest.raises(RuntimeError, match="Action unavailable"):
        invoke(handler)

    handler._native_exec.update_run_status.assert_awaited_once()
    handler.platform_tool_service.call_function.assert_not_called()
    handler._handle_web_search.assert_not_called()


def test_successful_admission_preserves_tool_dispatch(handler):
    order = []
    action = SimpleNamespace(id="action_test")

    async def admit(*args, **kwargs):
        order.append("admit")

    async def create(*args, **kwargs):
        order.append("action")
        return action

    def dispatch(*args, **kwargs):
        order.append("dispatch")
        return "tool result"

    handler._native_exec.update_run_status.side_effect = admit
    handler._native_exec.create_action.side_effect = create
    handler.platform_tool_service.call_function.side_effect = dispatch
    invoke(handler)

    assert order == ["admit", "action", "dispatch"]
    handler.platform_tool_service.call_function.assert_called_once_with(
        "web_search", {"query": "test"}
    )
    handler._handle_web_search.assert_awaited_once_with(
        thread_id="thread_test",
        assistant_id="assistant_test",
        tool_call_id="call_test",
        output="tool result",
        action=action,
    )
