"""Budget propagation through the router, provider handler and durable loop."""

import asyncio
from types import MethodType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.api.entities_api.orchestration.engine.orchestrator_core import OrchestratorCore
from src.api.entities_api.orchestration.handlers.together_handler import (
    TogetherAIHandler,
)
from tests.unit.test_run_model_authority import _run_stateful_with_mocks


@pytest.mark.parametrize("current_turn", [7, 8, 9])
def test_worker_budget_reaches_durable_orchestrator(current_turn):
    model = "together-ai/Qwen/Qwen3.8-Flash"
    invoke, selector, _, _, _ = _run_stateful_with_mocks(
        model,
        model,
        stream=True,
        assistant_lookup=AsyncMock(
            return_value=SimpleNamespace(
                meta_data={"is_research_worker": "true"},
                max_turns=8,
            )
        ),
    )
    native = SimpleNamespace(
        retrieve_run=AsyncMock(
            return_value=SimpleNamespace(
                current_turn=current_turn,
                started_at=123,
            )
        ),
        update_run_fields=AsyncMock(),
        update_run_status=AsyncMock(),
    )
    turns = []

    async def stream(**kwargs):
        turns.append(kwargs)
        if False:
            yield

    async def tools(**kwargs):
        if False:
            yield

    worker = SimpleNamespace(
        _native_exec=native,
        _ensure_config_loaded=AsyncMock(),
        get_function_call_state=lambda: [{"name": "read_scratchpad", "arguments": {}}],
        set_function_call_state=MagicMock(),
        set_tool_response_state=MagicMock(),
        _resolve_authorized_consumer_tool_names=AsyncMock(return_value=frozenset()),
        _get_mcp_tool_executor=lambda _: None,
        assistant_id="assistant_1",
        _decision_payload=None,
        stream=stream,
        process_tool_calls=tools,
    )
    worker.process_conversation = MethodType(
        OrchestratorCore.process_conversation, worker
    )
    handler = TogetherAIHandler.__new__(TogetherAIHandler)
    handler._get_specific_handler_instance = MagicMock(return_value=worker)
    selector.select_provider.return_value = handler, model

    result = asyncio.run(invoke())

    assert "Maximum conversation turns exceeded" in result
    assert len(turns) == (1 if current_turn == 7 else 0)
    if turns:
        assert turns[0]["model"] == model
        assert turns[0]["run_id"] == "run_1"
        native.update_run_fields.assert_any_await("run_1", current_turn=8)
    native.update_run_status.assert_awaited_with("run_1", "failed")
    assert any(
        "Max turns (8)" in call.kwargs.get("incomplete_details", "")
        for call in native.update_run_fields.await_args_list
    )
