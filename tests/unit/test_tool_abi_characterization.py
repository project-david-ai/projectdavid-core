"""Characterize the legacy tool boundaries that MCP-0 must preserve."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from projectdavid_common.validation import StatusEnum

from src.api.entities_api.orchestration.mixins.consumer_tool_handlers_mixin import (
    ConsumerToolHandlersMixin,
)
from src.api.entities_api.platform_tools.definitions.code_interpreter import (
    code_interpreter,
)


class RecordingNativeExec:
    def __init__(self) -> None:
        self.calls = []

    async def create_action(self, **kwargs):
        self.calls.append(("create_action", kwargs))
        return SimpleNamespace(id="action_123")

    async def update_run_status(self, *args):
        self.calls.append(("update_run_status", args))

    async def submit_tool_output(self, **kwargs):
        self.calls.append(("submit_tool_output", kwargs))

    async def update_action_status(self, *args):
        self.calls.append(("update_action_status", args))


class ConsumerHarness(ConsumerToolHandlersMixin):
    def __init__(self) -> None:
        self._native_exec = RecordingNativeExec()


def test_provider_tool_definition_uses_openai_function_shape():
    assert code_interpreter["type"] == "function"
    assert code_interpreter["function"]["name"] == "code_interpreter"
    assert isinstance(code_interpreter["function"]["description"], str)
    assert code_interpreter["function"]["parameters"] == {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": code_interpreter["function"]["parameters"]["properties"][
                    "code"
                ]["description"],
            }
        },
        "required": ["code"],
    }


def test_consumer_handoff_preserves_legacy_manifest_and_action_mapping():
    harness = ConsumerHarness()

    async def exercise():
        return [
            item
            async for item in harness._handover_to_consumer(
                thread_id="thread_1",
                assistant_id="assistant_1",
                content={"name": "get_weather", "arguments": {"city": "Accra"}},
                run_id="run_1",
                tool_call_id="call_1",
                decision={"confidence": 0.9},
            )
        ]

    output = asyncio.run(exercise())

    assert [json.loads(item) for item in output] == [
        {
            "type": "tool_call_manifest",
            "run_id": "run_1",
            "action_id": "action_123",
            "tool_call_id": "call_1",
            "tool": "get_weather",
            "args": {"city": "Accra"},
        }
    ]
    assert harness._native_exec.calls == [
        (
            "create_action",
            {
                "tool_name": "get_weather",
                "run_id": "run_1",
                "tool_call_id": "call_1",
                "function_args": {"city": "Accra"},
                "decision": {"confidence": 0.9},
            },
        ),
        ("update_run_status", ("run_1", StatusEnum.pending_action.value)),
    ]


def test_string_tool_result_and_error_flag_are_persisted_unchanged():
    harness = ConsumerHarness()

    asyncio.run(
        harness.submit_tool_output(
            thread_id="thread_1",
            assistant_id="assistant_1",
            tool_call_id="call_1",
            content='{"temperature": 24}',
            action=SimpleNamespace(id="action_123"),
            is_error=True,
        )
    )

    assert harness._native_exec.calls == [
        (
            "submit_tool_output",
            {
                "thread_id": "thread_1",
                "assistant_id": "assistant_1",
                "tool_call_id": "call_1",
                "content": '{"temperature": 24}',
                "action_id": "action_123",
                "is_error": True,
            },
        ),
        ("update_action_status", ("action_123", StatusEnum.failed.value)),
    ]
