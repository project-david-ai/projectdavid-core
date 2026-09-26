"""MCP-0 tests for the minimal internal Tool ABI."""

from __future__ import annotations

import asyncio

import pytest

from src.api.entities_api.orchestration.tool_abi import (
    ToolCallEnvelope,
    ToolDefinition,
    ToolExecutor,
    ToolResultEnvelope,
)
from src.api.entities_api.platform_tools.definitions.code_interpreter import (
    code_interpreter,
)


def test_tool_definition_round_trips_current_provider_schema():
    definition = ToolDefinition.from_function_tool(code_interpreter)

    assert definition.name == "code_interpreter"
    assert definition.input_schema["required"] == ["code"]
    assert definition.to_function_tool() == code_interpreter


def test_tool_definition_rejects_non_function_provider_shape():
    with pytest.raises(ValueError, match="function tool definition"):
        ToolDefinition.from_function_tool({"type": "web_search"})


def test_tool_call_round_trips_current_flat_call_shape_with_context():
    legacy = {
        "id": "call_1",
        "name": "get_weather",
        "arguments": {"city": "Accra"},
    }

    call = ToolCallEnvelope.from_legacy_call(
        legacy,
        run_id="run_1",
        thread_id="thread_1",
        assistant_id="assistant_1",
    )

    assert call.run_id == "run_1"
    assert call.thread_id == "thread_1"
    assert call.assistant_id == "assistant_1"
    assert call.to_legacy_call() == legacy


def test_explicit_tool_call_id_keeps_current_dispatch_precedence():
    call = ToolCallEnvelope.from_legacy_call(
        {"id": "call_from_batch", "name": "get_weather", "arguments": {}},
        run_id="run_1",
        thread_id="thread_1",
        assistant_id="assistant_1",
        tool_call_id="call_from_orchestrator",
    )

    assert call.tool_call_id == "call_from_orchestrator"


def test_tool_result_round_trips_current_string_contract():
    result = ToolResultEnvelope.from_legacy_result('{"temperature": 24}', is_error=True)

    assert result.to_legacy_result() == {
        "content": '{"temperature": 24}',
        "is_error": True,
    }


def test_tool_executor_protocol_matches_async_adapter_shape():
    class EchoExecutor:
        async def execute(self, call: ToolCallEnvelope) -> ToolResultEnvelope:
            return ToolResultEnvelope(content=call.name)

    executor = EchoExecutor()
    call = ToolCallEnvelope(
        name="echo",
        arguments={},
        run_id="run_1",
        thread_id="thread_1",
        assistant_id="assistant_1",
    )

    assert isinstance(executor, ToolExecutor)
    assert asyncio.run(executor.execute(call)) == ToolResultEnvelope(content="echo")


def test_tool_result_preserves_rich_fields_without_changing_legacy_projection():
    structured = {"answer": 42, "nested": {"ok": True}}
    blocks = [
        {
            "type": "image",
            "data": "YWJjZA==",
            "mimeType": "image/png",
        }
    ]
    metadata = {
        "mcp_result_type": "complete",
        "mcp_meta": {"trace": "abc"},
    }

    result = ToolResultEnvelope(
        content='{"answer": 42}',
        structured_content=structured,
        content_blocks=tuple(blocks),
        metadata=metadata,
        is_error=False,
    )

    structured["answer"] = 99
    blocks[0]["mimeType"] = "image/jpeg"
    metadata["mcp_result_type"] = "input_required"

    assert result.structured_content == {
        "answer": 42,
        "nested": {"ok": True},
    }
    assert result.content_blocks == (
        {
            "type": "image",
            "data": "YWJjZA==",
            "mimeType": "image/png",
        },
    )
    assert result.metadata == {
        "mcp_result_type": "complete",
        "mcp_meta": {"trace": "abc"},
    }

    assert result.to_legacy_result() == {
        "content": '{"answer": 42}',
        "is_error": False,
    }
