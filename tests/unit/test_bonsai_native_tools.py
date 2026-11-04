"""Contracts for Bonsai native tool streaming and replay."""

import ast
import asyncio
import json
import logging
import os
import runpy
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from src.api.entities_api.clients.delta_normalizer import DeltaNormalizer

ROOT = Path(__file__).resolve().parents[2] / "src/api/entities_api/orchestration"
HISTORY = runpy.run_path(str(ROOT / "mixins/g97_native_history.py"))
MODEL = "Prism-ML/Ternary-Bonsai-27B"


def call(call_id, arguments, index=0):
    return {
        "index": index,
        "id": call_id,
        "function": {"name": "file_search", "arguments": arguments},
    }


def provider_stream(calls, finish):
    async def raw():
        yield {"choices": [{"delta": {"tool_calls": calls}}]}
        if finish:
            yield {"choices": [{"delta": {}, "finish_reason": finish}]}

    return raw()


def test_legacy_provider_keeps_original_event_and_eof_behavior():
    async def collect():
        return [
            event["content"]
            async for event in DeltaNormalizer.async_iter_deltas(
                provider_stream([call("provider-1", "{}")], None), "run"
            )
            if event["type"] == "tool_call"
        ]

    assert asyncio.run(collect()) == [{"name": "file_search", "arguments": "{}"}]


def test_native_id_survives_argument_fragments():
    async def raw():
        yield {"choices": [{"delta": {"tool_calls": [call("one", '{"query":')]}}]}
        continuation = {"index": 0, "function": {"arguments": '"x"}'}}
        yield {"choices": [{"delta": {"tool_calls": [continuation]}}]}
        yield {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}

    async def collect():
        return [
            event["content"]
            async for event in DeltaNormalizer.async_iter_deltas(
                raw(), "run", strict_native_tool_calls=True
            )
            if event["type"] == "tool_call"
        ]

    assert asyncio.run(collect()) == [
        {"id": "one", "name": "file_search", "arguments": '{"query":"x"}'}
    ]


def test_orphan_results_are_only_demoted_outside_a_pending_batch():
    prepare = HISTORY["prepare_bonsai_native_history"]
    orphan = {"role": "tool", "tool_call_id": "old", "content": "legacy"}
    native = {"id": "new", "name": "file_search", "arguments": {"query": "x"}}
    assistant = {
        "role": "assistant",
        "tool_calls": json.loads(HISTORY["to_native_assistant_envelope"]([native])),
    }
    result = {"role": "tool", "tool_call_id": "new", "content": "matched"}
    assert prepare([orphan, assistant, result]) == [
        {"role": "user", "content": "Legacy tool result (unlinked):\nlegacy"},
        assistant,
        result,
    ]
    with pytest.raises(ValueError, match="MISSING_TOOL_OUTPUT"):
        prepare([assistant, orphan, result])


def load_method(file, class_name, name, namespace):
    path = ROOT / file
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    method = next(n for n in cls.body if getattr(n, "name", None) == name)
    method.decorator_list = []
    future = ast.parse("from __future__ import annotations").body
    exec(
        compile(ast.Module(body=[*future, method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace[name]


@pytest.fixture
def worker_factory(monkeypatch):
    monkeypatch.delenv("G97_DISABLE_NATIVE_BONSAI", raising=False)
    log = logging.getLogger(__name__)
    normalise = load_method(
        "mixins/tool_routing_mixin.py",
        "ToolRoutingMixin",
        "normalize_structured_function_calls",
        {"json": json, "uuid": uuid, "LOG": log},
    )
    stream = load_method(
        "workers/base_workers/default_base.py",
        "DefaultBaseWorker",
        "stream",
        {
            **HISTORY,
            "os": os,
            "LOG": log,
            "DeltaNormalizer": DeltaNormalizer,
            "StreamState": lambda: SimpleNamespace(
                current_block=None,
                accumulated="",
                assistant_reply="",
                decision_buffer="",
            ),
            "StatusEnum": SimpleNamespace(
                completed=SimpleNamespace(value="completed"),
                pending_action=SimpleNamespace(value="pending_action"),
            ),
            "is_multimodal": lambda ctx: False,
        },
    )
    worker_type = type("TogetherDefaultWorker", (), {"stream": stream})

    def make(calls, finish):
        worker = worker_type()
        worker.assistant_config = {}
        worker._batfish_owner_user_id = None
        worker._native_exec = SimpleNamespace(
            retrieve_run=AsyncMock(
                return_value=SimpleNamespace(user_id="user", meta_data={})
            ),
            update_run_status=AsyncMock(),
        )
        worker.start_cancellation_monitor = Mock(
            side_effect=lambda run_id: asyncio.Event()
        )
        worker._ensure_config_loaded = AsyncMock()
        worker._handle_role_based_identity_swap = AsyncMock()
        worker._set_up_context_window = AsyncMock(return_value=[])
        worker.prepare_native_tool_context = Mock(
            return_value=([], [{"type": "function"}])
        )
        worker._get_client_instance = Mock(
            return_value=SimpleNamespace(
                stream_chat_completion=Mock(
                    side_effect=lambda **kwargs: provider_stream(calls, finish),
                )
            )
        )
        worker.normalize_structured_function_calls = normalise
        worker.finalize_conversation = AsyncMock()
        worker.set_tool_response_state = Mock()
        worker.set_function_call_state = Mock()
        return worker

    return make


async def collect(worker, events):
    async for event in worker.stream(
        "thread", None, "run", "assistant", MODEL, api_key="test"
    ):
        events.append(json.loads(event))


@pytest.mark.parametrize(
    ("calls", "finish", "error"),
    [
        ([call("one", "{}"), call("two", "{", 1)], "tool_calls", "VALIDATION_FAILED"),
        ([call("one", "{}")], None, "without tool_calls finish"),
        ([call("one", "{}")], "length", "without tool_calls finish"),
    ],
)
def test_failed_batch_never_becomes_routable(worker_factory, calls, finish, error):
    worker = worker_factory(calls, finish)
    events = []
    with pytest.raises((RuntimeError, ValueError), match=error):
        asyncio.run(collect(worker, events))
    worker.set_function_call_state.assert_not_called()
    worker.finalize_conversation.assert_not_awaited()
    assert not any(event["type"] == "tool_call" for event in events)


def test_persistence_failure_never_becomes_routable(worker_factory):
    worker = worker_factory([call("provider-1", "{}")], "tool_calls")
    worker.finalize_conversation.side_effect = RuntimeError("write failed")
    events = []
    with pytest.raises(RuntimeError, match="write failed"):
        asyncio.run(collect(worker, events))
    worker.set_function_call_state.assert_not_called()
    assert not any(event["type"] == "tool_call" for event in events)


def test_completed_batch_persists_provider_id_before_dispatch(worker_factory):
    worker = worker_factory([call("provider-1", '{"query":"x"}')], "tool_calls")

    def assert_persisted(batch):
        worker.finalize_conversation.assert_awaited_once()
        saved = json.loads(worker.finalize_conversation.await_args.args[0])
        assert saved[0]["id"] == batch[0]["id"] == "provider-1"

    worker.set_function_call_state.side_effect = assert_persisted
    events = []
    asyncio.run(collect(worker, events))
    worker.set_function_call_state.assert_called_once()
    assert [
        event["content"]["id"] for event in events if event["type"] == "tool_call"
    ] == ["provider-1"]
