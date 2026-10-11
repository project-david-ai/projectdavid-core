from __future__ import annotations

import asyncio
import json

from src.api.entities_api.orchestration.mixins.consumer_tool_handlers_mixin import (
    ConsumerToolHandlersMixin,
)


def function_tool(
    name: str,
) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    }


class FakeAssistantCache:
    def __init__(
        self,
        tools,
        *,
        fail: bool = False,
    ) -> None:
        self.tools = list(tools)

        self.fail = fail

    async def retrieve(
        self,
        assistant_id: str,
    ):
        if self.fail:
            raise RuntimeError("assistant cache unavailable")

        return {
            "id": assistant_id,
            "tools": self.tools,
        }


class RecordingNativeExec:
    def __init__(
        self,
    ) -> None:
        self.created_actions = []
        self.tool_outputs = []
        self.run_statuses = []

    async def create_action(
        self,
        **kwargs,
    ):
        self.created_actions.append(dict(kwargs))

        return type(
            "Action",
            (),
            {
                "id": "act_authority_test",
            },
        )()

    async def submit_tool_output(
        self,
        **kwargs,
    ):
        self.tool_outputs.append(dict(kwargs))

    async def update_run_status(
        self,
        run_id,
        status,
    ):
        self.run_statuses.append(
            (
                run_id,
                status,
            )
        )


class Harness(ConsumerToolHandlersMixin):
    def __init__(
        self,
        tools,
        *,
        fail_cache: bool = False,
    ) -> None:
        self._assistant_cache = FakeAssistantCache(
            tools,
            fail=fail_cache,
        )

        self._native_exec = RecordingNativeExec()

    def get_assistant_cache(
        self,
    ):
        return self._assistant_cache

    def _get_mcp_tool_executor(
        self,
        provider_name: str,
    ):
        # Existing MCP integration tests cover native execution.
        # These tests isolate the authority boundary itself.
        return None


async def handover(
    harness,
    name: str,
):
    return [
        item
        async for item in harness._handover_to_consumer(
            thread_id="thread_auth",
            assistant_id="asst_auth",
            content={
                "name": name,
                "arguments": {},
            },
            run_id="run_auth",
            tool_call_id="call_auth",
        )
    ]


def test_forbidden_unattached_mcp_alias_creates_no_action():
    harness = Harness([function_tool("playwright-primary__browser_snapshot")])

    emitted = asyncio.run(
        handover(
            harness,
            "playwright-primary__browser_click",
        )
    )

    assert emitted == []

    assert harness._native_exec.created_actions == []

    assert len(harness._native_exec.tool_outputs) == 1

    result = harness._native_exec.tool_outputs[0]

    assert result["is_error"] is True

    assert "browser_click" in result["content"]


def test_hallucinated_sdk_function_creates_no_action():
    harness = Harness([function_tool("jobs_delegate")])

    emitted = asyncio.run(
        handover(
            harness,
            "totally_invented_function",
        )
    )

    assert emitted == []

    assert harness._native_exec.created_actions == []


def test_authorized_sdk_function_still_creates_manifest():
    harness = Harness([function_tool("jobs_delegate")])

    emitted = asyncio.run(
        handover(
            harness,
            "jobs_delegate",
        )
    )

    assert len(harness._native_exec.created_actions) == 1

    assert len(emitted) == 1

    manifest = json.loads(emitted[0])

    assert manifest["type"] == "tool_call_manifest"

    assert manifest["tool"] == "jobs_delegate"


def test_authorized_mcp_alias_passes_authority_gate():
    alias = "playwright-primary__browser_snapshot"

    harness = Harness([function_tool(alias)])

    emitted = asyncio.run(
        handover(
            harness,
            alias,
        )
    )

    assert len(harness._native_exec.created_actions) == 1

    assert len(emitted) == 1

    assert json.loads(emitted[0])["tool"] == alias


def test_assistant_config_failure_fails_closed():
    harness = Harness(
        [],
        fail_cache=True,
    )

    emitted = asyncio.run(
        handover(
            harness,
            "jobs_delegate",
        )
    )

    assert emitted == []

    assert harness._native_exec.created_actions == []

    assert len(harness._native_exec.tool_outputs) == 1
