"""MCP-6A long-running execution tests."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from mcp.types import CallToolResult, TextContent, Tool
from projectdavid_common.validation import StatusEnum

from src.api.entities_api.orchestration.mcp_remote_client import RemoteMcpClient
from src.api.entities_api.orchestration.mcp_tool_discovery import adapt_mcp_tools
from src.api.entities_api.orchestration.mcp_tool_executor import McpToolExecutor
from src.api.entities_api.orchestration.mixins.consumer_tool_handlers_mixin import (
    ConsumerToolHandlersMixin,
)
from src.api.entities_api.orchestration.tool_abi import ToolCallEnvelope


def discovered_tool():
    remote = Tool.model_validate(
        {
            "name": "slow.operation",
            "description": "Slow operation",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        }
    )

    (tool,) = adapt_mcp_tools("remote", [remote])
    return tool


def call_for(tool):
    return ToolCallEnvelope(
        name=tool.provider_name,
        arguments={},
        run_id="run_cancel",
        thread_id="thread_1",
        assistant_id="assistant_1",
        tool_call_id="call_1",
    )


def success_result():
    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text="done",
            )
        ],
        isError=False,
    )


def test_remote_client_forwards_progress_callback():
    observed = {}

    class ActiveClient:
        async def call_tool(
            self,
            name,
            arguments,
            *,
            progress_callback=None,
        ):
            observed["name"] = name
            observed["arguments"] = arguments
            observed["callback"] = progress_callback
            return success_result()

    async def callback(progress, total, message):
        return None

    async def exercise():
        client = object.__new__(RemoteMcpClient)
        client._connected = True
        client._client = ActiveClient()

        return await client.call_tool(
            "slow.operation",
            {"x": 1},
            progress_callback=callback,
        )

    result = asyncio.run(exercise())

    assert result.is_error is False
    assert observed == {
        "name": "slow.operation",
        "arguments": {"x": 1},
        "callback": callback,
    }


def test_executor_forwards_progress_and_preserves_result():
    tool = discovered_tool()
    progress_events = []

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def call_tool(
            self,
            name,
            arguments=None,
            *,
            progress_callback=None,
        ):
            assert progress_callback is not None

            await progress_callback(
                1.0,
                2.0,
                "half way",
            )

            return success_result()

    async def progress(progress, total, message):
        progress_events.append((progress, total, message))

    result = asyncio.run(
        McpToolExecutor(
            tool,
            Client,
        ).execute(
            call_for(tool),
            progress_callback=progress,
        )
    )

    assert result.content == "done"
    assert result.is_error is False
    assert progress_events == [(1.0, 2.0, "half way")]


def test_executor_classifies_timeout_without_losing_tool_failure_contract():
    tool = discovered_tool()

    class WrappedTimeout(RuntimeError):
        pass

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def call_tool(
            self,
            name,
            arguments=None,
            *,
            progress_callback=None,
        ):
            try:
                raise TimeoutError("remote deadline")
            except TimeoutError as exc:
                raise WrappedTimeout("wrapped request failure") from exc

    result = asyncio.run(
        McpToolExecutor(
            tool,
            Client,
        ).execute(call_for(tool))
    )

    assert result.is_error is True
    assert result.metadata["mcp_timeout"] is True
    assert result.metadata["exception_type"] == "WrappedTimeout"
    assert "timed out" in result.content


def test_executor_does_not_swallow_asyncio_cancellation():
    tool = discovered_tool()

    started = asyncio.Event()

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def call_tool(
            self,
            name,
            arguments=None,
            *,
            progress_callback=None,
        ):
            started.set()
            await asyncio.Event().wait()

    async def exercise():
        task = asyncio.create_task(
            McpToolExecutor(
                tool,
                Client,
            ).execute(call_for(tool))
        )

        await started.wait()

        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())


def test_run_cancellation_cancels_inflight_mcp_and_marks_action_cancelled():
    tool = discovered_tool()
    remote_cancelled = asyncio.Event()

    class SlowClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def call_tool(
            self,
            name,
            arguments=None,
            *,
            progress_callback=None,
        ):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                remote_cancelled.set()
                raise

    class NativeExec:
        def __init__(self):
            self.polls = 0
            self.status_updates = []

        async def retrieve_run(self, run_id):
            self.polls += 1

            status = (
                StatusEnum.pending_action.value
                if self.polls == 1
                else StatusEnum.cancelled.value
            )

            return SimpleNamespace(
                id=run_id,
                status=status,
            )

        async def update_action_status(
            self,
            action_id,
            status,
        ):
            self.status_updates.append((action_id, status))

    class Harness(ConsumerToolHandlersMixin):
        _MCP_CANCEL_POLL_SECONDS = 0.01

        def __init__(self):
            self._native_exec = NativeExec()

    async def exercise():
        harness = Harness()

        result = await harness._execute_mcp_tool_with_run_cancellation(
            executor=McpToolExecutor(
                tool,
                SlowClient,
            ),
            call=call_for(tool),
            action=SimpleNamespace(id="action_1"),
        )

        assert result is None
        assert remote_cancelled.is_set()
        assert harness._native_exec.status_updates == [
            (
                "action_1",
                StatusEnum.cancelled.value,
            )
        ]

    asyncio.run(exercise())


def test_completed_result_is_suppressed_when_run_is_already_cancelled():
    tool = discovered_tool()

    class FastClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def call_tool(
            self,
            name,
            arguments=None,
            *,
            progress_callback=None,
        ):
            return success_result()

    class NativeExec:
        def __init__(self):
            self.status_updates = []

        async def retrieve_run(self, run_id):
            return SimpleNamespace(
                id=run_id,
                status=StatusEnum.cancelled.value,
            )

        async def update_action_status(
            self,
            action_id,
            status,
        ):
            self.status_updates.append((action_id, status))

    class Harness(ConsumerToolHandlersMixin):
        _MCP_CANCEL_POLL_SECONDS = 0.01

        def __init__(self):
            self._native_exec = NativeExec()

    async def exercise():
        harness = Harness()

        result = await harness._execute_mcp_tool_with_run_cancellation(
            executor=McpToolExecutor(
                tool,
                FastClient,
            ),
            call=call_for(tool),
            action=SimpleNamespace(id="action_race"),
        )

        assert result is None
        assert harness._native_exec.status_updates == [
            (
                "action_race",
                StatusEnum.cancelled.value,
            )
        ]

    asyncio.run(exercise())
