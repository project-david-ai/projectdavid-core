"""MCP-3 execution integration tests."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from mcp.types import CallToolResult, Tool
from projectdavid_common.validation import StatusEnum

from src.api.entities_api.orchestration.mcp_tool_discovery import adapt_mcp_tools
from src.api.entities_api.orchestration.mcp_tool_executor import McpToolExecutor
from src.api.entities_api.orchestration.mixins.consumer_tool_handlers_mixin import (
    ConsumerToolHandlersMixin,
)
from src.api.entities_api.orchestration.mixins.tool_routing_mixin import (
    ToolRoutingMixin,
)
from src.api.entities_api.orchestration.tool_abi import ToolCallEnvelope


class FakeMcpClient:
    def __init__(self, result: CallToolResult) -> None:
        self.result = result
        self.calls: list[tuple[str, dict | None]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def call_tool(
        self,
        name: str,
        arguments: dict | None = None,
    ) -> CallToolResult:
        self.calls.append((name, arguments))
        return self.result


class RecordingNativeExec:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    async def create_action(self, **kwargs):
        self.calls.append(("create_action", kwargs))
        return SimpleNamespace(id="action_mcp_1")

    async def update_run_status(self, *args):
        self.calls.append(("update_run_status", args))

    async def submit_tool_output(self, **kwargs):
        self.calls.append(("submit_tool_output", kwargs))

    async def update_action_status(self, *args):
        self.calls.append(("update_action_status", args))


class FakeAssistantCache:
    async def retrieve(self, assistant_id: str):
        return {"tools": []}


class RouterHarness(ToolRoutingMixin, ConsumerToolHandlersMixin):
    def __init__(self) -> None:
        self._native_exec = RecordingNativeExec()
        self._batfish_owner_user_id = "user_1"
        self._assistant_cache = FakeAssistantCache()

    def get_assistant_cache(self):
        return self._assistant_cache


def discovered_tool():
    remote = Tool.model_validate(
        {
            "name": "search.issues",
            "description": "Search issues",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
            },
        }
    )

    (tool,) = adapt_mcp_tools("github", [remote])
    return tool


def result_with_text(
    text: str,
    *,
    is_error: bool = False,
) -> CallToolResult:
    return CallToolResult.model_validate(
        {
            "content": [
                {
                    "type": "text",
                    "text": text,
                }
            ],
            "isError": is_error,
        }
    )


def test_executor_calls_remote_name_not_provider_alias():
    tool = discovered_tool()
    client = FakeMcpClient(result_with_text("three issues"))

    executor = McpToolExecutor(tool, lambda: client)

    call = ToolCallEnvelope(
        name=tool.provider_name,
        arguments={"query": "bug"},
        run_id="run_1",
        thread_id="thread_1",
        assistant_id="assistant_1",
        tool_call_id="call_1",
    )

    result = asyncio.run(executor.execute(call))

    assert client.calls == [
        (
            "search.issues",
            {"query": "bug"},
        )
    ]
    assert result.content == "three issues"
    assert result.is_error is False


def test_executor_preserves_mcp_error_flag():
    tool = discovered_tool()
    client = FakeMcpClient(
        result_with_text(
            "remote failure",
            is_error=True,
        )
    )

    result = asyncio.run(
        McpToolExecutor(tool, lambda: client).execute(
            ToolCallEnvelope(
                name=tool.provider_name,
                arguments={},
                run_id="run_1",
                thread_id="thread_1",
                assistant_id="assistant_1",
            )
        )
    )

    assert result.content == "remote failure"
    assert result.is_error is True


def test_executor_flattens_structured_result_when_no_text_is_available():
    tool = discovered_tool()

    client = FakeMcpClient(
        CallToolResult.model_validate(
            {
                "content": [],
                "structuredContent": {
                    "count": 3,
                    "items": ["a", "b", "c"],
                },
            }
        )
    )

    result = asyncio.run(
        McpToolExecutor(tool, lambda: client).execute(
            ToolCallEnvelope(
                name=tool.provider_name,
                arguments={},
                run_id="run_1",
                thread_id="thread_1",
                assistant_id="assistant_1",
            )
        )
    )

    assert json.loads(result.content) == {
        "count": 3,
        "items": ["a", "b", "c"],
    }


def test_executor_rejects_call_for_different_alias():
    tool = discovered_tool()
    client = FakeMcpClient(result_with_text("unused"))

    executor = McpToolExecutor(tool, lambda: client)

    with pytest.raises(ValueError, match="different provider alias"):
        asyncio.run(
            executor.execute(
                ToolCallEnvelope(
                    name="different__tool",
                    arguments={},
                    run_id="run_1",
                    thread_id="thread_1",
                    assistant_id="assistant_1",
                )
            )
        )

    assert client.calls == []


def test_router_executes_bound_mcp_tool_inside_existing_action_lifecycle():
    harness = RouterHarness()

    tool = discovered_tool()
    client = FakeMcpClient(result_with_text("three issues"))
    executor = McpToolExecutor(tool, lambda: client)

    harness.bind_mcp_tool_executor(executor)

    harness.set_function_call_state(
        {
            "id": "call_mcp_1",
            "name": tool.provider_name,
            "arguments": {"query": "bug"},
        }
    )

    async def exercise():
        return [
            chunk
            async for chunk in harness.process_tool_calls(
                thread_id="thread_1",
                run_id="run_1",
                assistant_id="assistant_1",
            )
        ]

    output = asyncio.run(exercise())

    assert output == []

    assert client.calls == [
        (
            "search.issues",
            {"query": "bug"},
        )
    ]

    assert harness._native_exec.calls == [
        (
            "create_action",
            {
                "tool_name": tool.provider_name,
                "run_id": "run_1",
                "tool_call_id": "call_mcp_1",
                "function_args": {"query": "bug"},
                "decision": None,
            },
        ),
        (
            "update_run_status",
            ("run_1", StatusEnum.pending_action.value),
        ),
        (
            "submit_tool_output",
            {
                "thread_id": "thread_1",
                "assistant_id": "assistant_1",
                "tool_call_id": "call_mcp_1",
                "content": "three issues",
                "action_id": "action_mcp_1",
                "is_error": False,
            },
        ),
        (
            "update_action_status",
            ("action_mcp_1", "completed"),
        ),
    ]


def test_router_marks_action_failed_when_mcp_returns_error():
    harness = RouterHarness()

    tool = discovered_tool()
    client = FakeMcpClient(
        result_with_text(
            "permission denied",
            is_error=True,
        )
    )

    harness.bind_mcp_tool_executor(McpToolExecutor(tool, lambda: client))

    harness.set_function_call_state(
        {
            "id": "call_mcp_error",
            "name": tool.provider_name,
            "arguments": {},
        }
    )

    async def exercise():
        return [
            chunk
            async for chunk in harness.process_tool_calls(
                thread_id="thread_1",
                run_id="run_1",
                assistant_id="assistant_1",
            )
        ]

    assert asyncio.run(exercise()) == []

    assert harness._native_exec.calls[-2:] == [
        (
            "submit_tool_output",
            {
                "thread_id": "thread_1",
                "assistant_id": "assistant_1",
                "tool_call_id": "call_mcp_error",
                "content": "permission denied",
                "action_id": "action_mcp_1",
                "is_error": True,
            },
        ),
        (
            "update_action_status",
            ("action_mcp_1", "failed"),
        ),
    ]


def test_unbound_tool_still_uses_existing_consumer_handoff():
    harness = RouterHarness()

    harness.set_function_call_state(
        {
            "id": "call_consumer_1",
            "name": "get_weather",
            "arguments": {"city": "Accra"},
        }
    )

    async def exercise():
        return [
            json.loads(chunk)
            async for chunk in harness.process_tool_calls(
                thread_id="thread_1",
                run_id="run_1",
                assistant_id="assistant_1",
            )
        ]

    output = asyncio.run(exercise())

    assert output == [
        {
            "type": "tool_call_manifest",
            "run_id": "run_1",
            "action_id": "action_mcp_1",
            "tool_call_id": "call_consumer_1",
            "tool": "get_weather",
            "args": {"city": "Accra"},
        }
    ]


def test_router_rejects_binding_over_internal_tool_name():
    harness = RouterHarness()
    tool = discovered_tool()

    conflicting = type(tool)(
        server_id=tool.server_id,
        remote_name=tool.remote_name,
        canonical_id=tool.canonical_id,
        provider_name="delegate_research_task",
        definition=tool.definition,
    )

    with pytest.raises(ValueError, match="collides with an internal tool"):
        harness.bind_mcp_tool_executor(
            McpToolExecutor(
                conflicting,
                lambda: FakeMcpClient(result_with_text("unused")),
            )
        )


def test_duplicate_mcp_alias_binding_is_rejected():
    harness = RouterHarness()
    tool = discovered_tool()

    first = McpToolExecutor(
        tool,
        lambda: FakeMcpClient(result_with_text("first")),
    )
    second = McpToolExecutor(
        tool,
        lambda: FakeMcpClient(result_with_text("second")),
    )

    harness.bind_mcp_tool_executor(first)

    with pytest.raises(ValueError, match="already bound"):
        harness.bind_mcp_tool_executor(second)


def test_mcp_binding_can_be_removed():
    harness = RouterHarness()
    tool = discovered_tool()

    executor = McpToolExecutor(
        tool,
        lambda: FakeMcpClient(result_with_text("unused")),
    )

    harness.bind_mcp_tool_executor(executor)

    assert harness.unbind_mcp_tool_executor(tool.provider_name) is True
    assert harness.unbind_mcp_tool_executor(tool.provider_name) is False
