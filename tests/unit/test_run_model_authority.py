"""Focused model-authority regressions for ProjectDavid Core.

Place under tests/unit/ and run with:
    python -m pytest tests/unit/test_run_model_authority.py -q

Tests are isolated from a real database, Redis or inference provider.
"""

import asyncio
import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from src.api.entities_api.orchestration.mixins.delegation_mixin import DelegationMixin
from src.api.entities_api.services import runs_service
from src.api.entities_api.services.native_execution_service import (
    NativeExecutionService,
)

inference_router = importlib.import_module(
    "src.api.entities_api.routers.inference_router"
)


def _run_request(model=None, resources=None):
    return SimpleNamespace(
        assistant_id="assistant_1",
        thread_id="thread_1",
        model=model,
        instructions="",
        meta_data={},
        tool_resources=resources,
        truncation_strategy=None,
    )


def _create_run_with_mock_db(monkeypatch, *, override, default, resources=None):
    db = MagicMock()
    db_ctx = MagicMock()
    db_ctx.__enter__.return_value = db
    assistant = SimpleNamespace(
        model=default,
        tool_configs=[{"type": "function", "function": {"name": "foo"}}],
        tool_resources={"code_interpreter": {"file_ids": ["file_1"]}},
        owner_id="user_1",
        users=[],
        temperature=0.7,
        top_p=0.9,
        max_turns=1,
        agent_mode=False,
        instructions="system",
    )
    db.query.return_value.filter.return_value.first.return_value = assistant

    monkeypatch.setattr(runs_service, "SessionLocal", lambda: db_ctx)
    monkeypatch.setattr(
        runs_service.RunService,
        "_lock_thread_for_run_creation",
        lambda *_: None,
    )
    monkeypatch.setattr(runs_service, "Run", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(
        runs_service.RunService, "_to_read_model", lambda _, result: result
    )
    svc = runs_service.RunService()
    return (
        svc.create_run(
            _run_request(model=override, resources=resources), user_id="user_1"
        ),
        db,
    )


def test_run_inherits_assistant_model_and_resources(monkeypatch):
    created, db = _create_run_with_mock_db(
        monkeypatch, override=None, default="together-ai/base-model"
    )
    assert created.model == "together-ai/base-model"
    assert created.tools[0]["function"]["name"] == "foo"
    assert created.tool_resources == {"code_interpreter": {"file_ids": ["file_1"]}}
    db.commit.assert_called_once()


def test_explicitly_empty_run_resources_do_not_inherit(monkeypatch):
    created, _ = _create_run_with_mock_db(
        monkeypatch,
        override=None,
        default="together-ai/base-model",
        resources={},
    )
    assert created.tool_resources == {}


def test_run_override_wins_and_is_persisted(monkeypatch):
    created, _ = _create_run_with_mock_db(
        monkeypatch,
        override="hyperbolic/different-model",
        default="together-ai/base-model",
    )
    assert created.model == "hyperbolic/different-model"


@pytest.mark.parametrize("override,default", [(None, None), (None, ""), ("   ", "ok")])
def test_missing_or_blank_effective_model_is_422(monkeypatch, override, default):
    with pytest.raises(HTTPException) as err:
        _create_run_with_mock_db(monkeypatch, override=override, default=default)
    assert err.value.status_code == 422


def _stateful(model="together-ai/base-model", *, stream=False):
    return inference_router.StatefulCompletionRequest(
        model=model,
        thread_id="thread_1",
        assistant_id="assistant_1",
        run_id="run_1",
        message_id="message_1",
        api_key="provider-key",
        stream=stream,
    )


def _run_stateful_with_mocks(request_model, persisted_model, *, stream=False):
    run = SimpleNamespace(
        id="run_1",
        user_id="user_1",
        assistant_id="assistant_1",
        thread_id="thread_1",
        model=persisted_model,
    )
    native = MagicMock()
    native.retrieve_run = AsyncMock(return_value=run)
    native.assert_assistant_access = AsyncMock()
    selector = MagicMock()
    worker_calls = []

    async def fake_process_conversation(**kwargs):
        worker_calls.append(kwargs)
        yield {"type": "content", "content": "model-authority-ok"}

    worker = SimpleNamespace(process_conversation=fake_process_conversation)
    selector.select_provider.return_value = (worker, "provider-model")
    db_ctx = MagicMock()
    db_ctx.__enter__.return_value = MagicMock()

    async def invoke():
        with (
            patch.object(
                inference_router, "NativeExecutionService", return_value=native
            ),
            patch.object(inference_router, "SessionLocal", return_value=db_ctx),
            patch.object(
                inference_router,
                "get_api_key",
                new=AsyncMock(return_value=SimpleNamespace(user_id="user_1")),
            ),
            patch.object(inference_router, "InferenceArbiter"),
            patch.object(
                inference_router, "InferenceProviderSelector", return_value=selector
            ),
        ):
            response = await inference_router.completions(
                _stateful(request_model, stream=stream),
                redis=MagicMock(),
                project_api_key="project-key",
            )
            if stream:
                response_chunks = [chunk async for chunk in response.body_iterator]
                return "".join(
                    chunk.decode() if isinstance(chunk, bytes) else chunk
                    for chunk in response_chunks
                )
            return json.loads(response.body)

    return invoke, selector, worker_calls


def test_stateful_buffered_inference_dispatches_persisted_model():
    invoke, selector, calls = _run_stateful_with_mocks(
        "hyperbolic/overridden", "hyperbolic/overridden"
    )
    body = asyncio.run(invoke())
    selector.select_provider.assert_called_once_with(model_id="hyperbolic/overridden")
    assert calls[0]["model"] == "hyperbolic/overridden"
    assert body["model"] == "hyperbolic/overridden"
    assert body["content"] == "model-authority-ok"


def test_stateful_streaming_inference_dispatches_persisted_model():
    invoke, selector, calls = _run_stateful_with_mocks(
        "together-ai/base-model", "together-ai/base-model", stream=True
    )
    data = asyncio.run(invoke())
    selector.select_provider.assert_called_once_with(model_id="together-ai/base-model")
    assert calls[0]["model"] == "together-ai/base-model"
    assert "model-authority-ok" in data
    assert "[DONE]" in data


def test_stateful_mismatch_returns_409_before_provider_dispatch():
    invoke, selector, calls = _run_stateful_with_mocks(
        "together-ai/base-model", "hyperbolic/overridden"
    )
    with pytest.raises(HTTPException) as err:
        asyncio.run(invoke())
    assert err.value.status_code == 409
    selector.select_provider.assert_not_called()
    assert calls == []


def test_stateful_persisted_missing_model_returns_409():
    invoke, selector, _ = _run_stateful_with_mocks("together-ai/base-model", "")
    with pytest.raises(HTTPException) as err:
        asyncio.run(invoke())
    assert err.value.status_code == 409
    selector.select_provider.assert_not_called()


def test_native_create_run_forwards_explicit_child_model():
    native = object.__new__(NativeExecutionService)
    native.run_svc = MagicMock()
    native.run_svc.create_run.return_value = SimpleNamespace(id="run_child")
    asyncio.run(
        native.create_run(
            assistant_id="assistant_child",
            thread_id="thread_child",
            user_id="user_1",
            model="hyperbolic/delegated",
        )
    )
    args, kwargs = native.run_svc.create_run.call_args
    assert args[0].model == "hyperbolic/delegated"
    assert kwargs["user_id"] == "user_1"


def test_research_delegation_persists_model_before_child_stream():
    mixin = object.__new__(DelegationMixin)
    mixin._batfish_owner_user_id = "user_1"
    mixin._scratch_pad_thread = None
    mixin._delete_ephemeral_thread = False
    mixin._assistant_manager_svc = SimpleNamespace(
        create_ephemeral_worker_assistant=AsyncMock(
            return_value=SimpleNamespace(id="assistant_child", name="research_worker")
        )
    )
    native = SimpleNamespace(
        create_action=AsyncMock(return_value=SimpleNamespace(id="action_1")),
        retrieve_run=AsyncMock(
            return_value=SimpleNamespace(
                user_id="user_1",
                meta_data={
                    "api_key": "provider-key",
                    "project_david_api_key": "project-key",
                    "delegated_model": "hyperbolic/delegated",
                },
            )
        ),
        create_thread=AsyncMock(return_value=SimpleNamespace(id="thread_child")),
        create_message=AsyncMock(return_value=SimpleNamespace(id="message_child")),
        create_run=AsyncMock(side_effect=RuntimeError("stop before worker stream")),
        update_run_fields=AsyncMock(),
    )
    mixin._native_exec = native
    mixin._ephemeral_clean_up = AsyncMock()

    async def invoke():
        generator = mixin.handle_delegate_research_task(
            thread_id="thread_1",
            run_id="run_parent",
            assistant_id="assistant_1",
            arguments_dict={"task": "research something"},
            tool_call_id="call_1",
            decision="yes",
        )
        async for _ in generator:
            pass

    asyncio.run(invoke())
    assert native.create_run.await_args.kwargs["model"] == "hyperbolic/delegated"
    assert native.create_run.await_args.kwargs["assistant_id"] == "assistant_child"
