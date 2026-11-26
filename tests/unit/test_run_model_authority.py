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
    # The account exists and has no erasure receipt.
    db.execute.return_value.scalar_one_or_none.side_effect = [
        SimpleNamespace(id="user_1"),
        None,
    ]
    db_ctx = MagicMock()
    db_ctx.__enter__.return_value = db

    assistant = SimpleNamespace(
        model=default,
        tool_configs=[{"type": "function", "function": {"name": "foo"}}],
        tool_resources={"code_interpreter": {"file_ids": ["file_1"]}},
        owner_id="user_1",
        deleted_at=None,
        is_protected=False,
        users=[],
        temperature=0.7,
        top_p=0.9,
        max_turns=1,
        agent_mode=False,
        instructions="system",
    )

    db.query.return_value.filter.return_value.first.return_value = assistant

    monkeypatch.setattr(
        runs_service,
        "SessionLocal",
        lambda: db_ctx,
    )

    monkeypatch.setattr(
        runs_service.RunService,
        "_lock_thread_for_run_creation",
        lambda *_: None,
    )

    monkeypatch.setattr(
        runs_service,
        "Run",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    monkeypatch.setattr(
        runs_service.RunService,
        "_to_read_model",
        lambda _, result: result,
    )

    svc = runs_service.RunService()

    return (
        svc.create_run(
            _run_request(model=override, resources=resources),
            user_id="user_1",
        ),
        db,
    )


def test_run_inherits_assistant_model_and_resources(monkeypatch):
    created, db = _create_run_with_mock_db(
        monkeypatch,
        override=None,
        default="together-ai/base-model",
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


@pytest.mark.parametrize(
    "override,default",
    [
        (None, None),
        (None, ""),
        ("   ", "ok"),
    ],
)
def test_missing_or_blank_effective_model_is_422(
    monkeypatch,
    override,
    default,
):
    with pytest.raises(HTTPException) as err:
        _create_run_with_mock_db(
            monkeypatch,
            override=override,
            default=default,
        )

    assert err.value.status_code == 422


def _stateful(model="together-ai/base-model", *, stream=False):
    return inference_router.StatefulCompletionRequest(
        model=model,
        thread_id="thread_1",
        assistant_id="assistant_1",
        run_id="run_1",
        message_id="message_1",
        api_key=None if model.startswith("hyperbolic/") else "provider-key",
        provider_key_id=(
            "pkey_model_authority" if model.startswith("hyperbolic/") else None
        ),
        stream=stream,
    )


def _run_stateful_with_mocks(
    request_model,
    persisted_model,
    *,
    stream=False,
    status="queued",
    response_consumer=None,
    assistant_lookup=None,
    authenticated_user_id="user_1",
    assistant_access_error=None,
):
    run = SimpleNamespace(
        id="run_1",
        user_id="user_1",
        assistant_id="assistant_1",
        thread_id="thread_1",
        model=persisted_model,
        status=status,
    )

    native = MagicMock()
    native.retrieve_run = AsyncMock(return_value=run)
    native.assert_assistant_access = AsyncMock(side_effect=assistant_access_error)
    native.retrieve_assistant = (
        assistant_lookup
        if assistant_lookup is not None
        else AsyncMock(return_value=SimpleNamespace(meta_data={}, max_turns=200))
    )

    native_service_class = MagicMock(return_value=native)
    native_service_class.resolve_provider_key_for_inference = AsyncMock(
        return_value="provider-key"
    )

    selector = MagicMock()
    worker_calls = []

    async def fake_process_conversation(**kwargs):
        worker_calls.append(kwargs)
        yield {
            "type": "content",
            "content": "model-authority-ok",
        }

    worker = SimpleNamespace(process_conversation=fake_process_conversation)

    selector.select_provider.return_value = (
        worker,
        "provider-model",
    )

    # The router re-reads and locks the persisted Run before model binding.
    db = MagicMock()
    locked_query = (
        db.query.return_value.filter.return_value.with_for_update.return_value
    )
    locked_query.first.return_value = run
    db_ctx = MagicMock()
    db_ctx.__enter__.return_value = db

    async def invoke():
        with (
            patch.object(
                inference_router,
                "NativeExecutionService",
                new=native_service_class,
            ),
            patch.object(
                inference_router,
                "SessionLocal",
                return_value=db_ctx,
            ),
            patch.object(
                inference_router,
                "get_api_key",
                new=AsyncMock(
                    return_value=SimpleNamespace(
                        user_id=authenticated_user_id, is_protected=False
                    )
                ),
            ),
            patch.object(
                inference_router,
                "InferenceArbiter",
            ),
            patch.object(
                inference_router,
                "InferenceProviderSelector",
                return_value=selector,
            ),
        ):
            response = await inference_router.completions(
                _stateful(request_model, stream=stream),
                redis=MagicMock(),
                project_api_key="project-key",
            )

            if response_consumer is not None:
                return await response_consumer(response)

            if stream:
                response_chunks = [chunk async for chunk in response.body_iterator]

                return "".join(
                    (chunk.decode() if isinstance(chunk, bytes) else chunk)
                    for chunk in response_chunks
                )

            return json.loads(response.body)

    return invoke, selector, worker_calls, run, db


def test_stateful_buffered_inference_dispatches_persisted_model():
    invoke, selector, calls, run, db = _run_stateful_with_mocks(
        "hyperbolic/overridden",
        "hyperbolic/overridden",
    )

    body = asyncio.run(invoke())

    selector.select_provider.assert_called_once_with(model_id="hyperbolic/overridden")

    assert calls[0]["model"] == "hyperbolic/overridden"
    assert calls[0]["api_key"] == "provider-key"
    assert "max_turns" not in calls[0]
    assert body["model"] == "hyperbolic/overridden"
    assert body["content"] == "model-authority-ok"
    assert run.model == "hyperbolic/overridden"
    db.commit.assert_not_called()


def test_stateful_streaming_inference_dispatches_persisted_model():
    invoke, selector, calls, run, db = _run_stateful_with_mocks(
        "together-ai/base-model",
        "together-ai/base-model",
        stream=True,
    )

    data = asyncio.run(invoke())

    selector.select_provider.assert_called_once_with(model_id="together-ai/base-model")

    assert calls[0]["model"] == "together-ai/base-model"
    assert "max_turns" not in calls[0]
    assert "model-authority-ok" in data
    assert "[DONE]" in data
    assert run.model == "together-ai/base-model"
    db.commit.assert_not_called()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "metadata",
    [
        {"is_research_worker": "true"},
        {"is_research_worker": True},
        {"research_worker_calling": "true"},
    ],
)
@pytest.mark.parametrize("model", ["hyperbolic/overridden", "together-ai/base-model"])
def test_research_worker_budget_is_forwarded_from_assistant(stream, metadata, model):
    lookup = AsyncMock(return_value=SimpleNamespace(meta_data=metadata, max_turns=8))
    invoke, selector, calls, run, db = _run_stateful_with_mocks(
        model, model, stream=stream, assistant_lookup=lookup
    )

    result = asyncio.run(invoke())

    lookup.assert_awaited_once_with("assistant_1")
    selector.select_provider.assert_called_once_with(model_id=model)
    assert len(calls) == 1
    assert calls[0]["max_turns"] == 8
    assert calls[0]["model"] == model
    assert run.model == model
    db.commit.assert_not_called()
    if stream:
        assert "model-authority-ok" in result
        assert "[DONE]" in result
    else:
        assert result["content"] == "model-authority-ok"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("budget", [0, -1, None, True, "8", 8.0])
def test_invalid_research_worker_budget_rejected_before_dispatch(stream, budget):
    lookup = AsyncMock(
        return_value=SimpleNamespace(
            meta_data={"is_research_worker": "true"}, max_turns=budget
        )
    )
    invoke, selector, calls, _, db = _run_stateful_with_mocks(
        "together-ai/base-model",
        "together-ai/base-model",
        stream=stream,
        assistant_lookup=lookup,
    )

    with pytest.raises(HTTPException) as error:
        asyncio.run(invoke())

    assert error.value.status_code == 500
    assert error.value.detail == "Invalid research worker turn budget."
    lookup.assert_awaited_once_with("assistant_1")
    selector.select_provider.assert_not_called()
    assert calls == []
    db.commit.assert_not_called()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {"is_research_worker": False},
        {"deep_research": True},
        {"junior_engineer": "true"},
        {"is_research_worker": "false", "research_worker_calling": "true"},
    ],
)
def test_non_worker_execution_budget_is_unchanged(stream, metadata):
    lookup = AsyncMock(return_value=SimpleNamespace(meta_data=metadata, max_turns=8))
    invoke, _, calls, _, _ = _run_stateful_with_mocks(
        "together-ai/base-model",
        "together-ai/base-model",
        stream=stream,
        assistant_lookup=lookup,
    )

    asyncio.run(invoke())

    assert len(calls) == 1
    assert "max_turns" not in calls[0]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("denial", ["run-owner", "assistant-access"])
def test_research_budget_lookup_requires_authorization(stream, denial):
    lookup = AsyncMock(
        return_value=SimpleNamespace(
            meta_data={"is_research_worker": "true"}, max_turns=8
        )
    )
    access_error = (
        HTTPException(status_code=403, detail="Assistant access denied.")
        if denial == "assistant-access"
        else None
    )
    invoke, selector, calls, _, db = _run_stateful_with_mocks(
        "together-ai/base-model",
        "together-ai/base-model",
        stream=stream,
        assistant_lookup=lookup,
        authenticated_user_id="other_user" if denial == "run-owner" else "user_1",
        assistant_access_error=access_error,
    )

    with pytest.raises(HTTPException) as error:
        asyncio.run(invoke())

    assert error.value.status_code == 403
    lookup.assert_not_awaited()
    selector.select_provider.assert_not_called()
    assert calls == []
    db.commit.assert_not_called()


def test_missing_assistant_rejected_before_provider_dispatch():
    lookup = AsyncMock(return_value=None)
    invoke, selector, calls, _, db = _run_stateful_with_mocks(
        "together-ai/base-model",
        "together-ai/base-model",
        assistant_lookup=lookup,
    )

    with pytest.raises(HTTPException) as error:
        asyncio.run(invoke())

    assert error.value.status_code == 404
    assert error.value.detail == "Assistant not found."
    selector.select_provider.assert_not_called()
    assert calls == []
    db.commit.assert_not_called()


def test_inference_model_overrides_queued_run_model_and_is_persisted():
    invoke, selector, calls, run, db = _run_stateful_with_mocks(
        "together-ai/base-model",
        "hyperbolic/overridden",
    )

    body = asyncio.run(invoke())

    selector.select_provider.assert_called_once_with(model_id="together-ai/base-model")
    assert calls[0]["model"] == "together-ai/base-model"
    assert body["model"] == "together-ai/base-model"
    assert run.model == "together-ai/base-model"
    db.commit.assert_called_once()


def test_missing_persisted_model_is_bound_by_first_inference():
    invoke, selector, calls, run, db = _run_stateful_with_mocks(
        "together-ai/base-model",
        "",
    )

    body = asyncio.run(invoke())

    assert body["model"] == "together-ai/base-model"
    assert run.model == "together-ai/base-model"
    db.commit.assert_called_once()
    selector.select_provider.assert_called_once_with(model_id="together-ai/base-model")
    assert calls[0]["model"] == "together-ai/base-model"


def test_inference_model_cannot_change_after_run_started():
    invoke, selector, calls, run, db = _run_stateful_with_mocks(
        "together-ai/base-model",
        "hyperbolic/overridden",
        status="in_progress",
    )

    with pytest.raises(HTTPException) as err:
        asyncio.run(invoke())

    assert err.value.status_code == 409
    assert err.value.detail == "Cannot change the Run model after execution begins."
    assert run.model == "hyperbolic/overridden"
    db.commit.assert_not_called()
    assert calls == []


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
            return_value=SimpleNamespace(
                id="assistant_child",
                name="research_worker",
            )
        )
    )

    parent_metadata = {
        "delegated_model": "hyperbolic/delegated",
    }

    native = SimpleNamespace(
        create_action=AsyncMock(return_value=SimpleNamespace(id="action_1")),
        retrieve_run=AsyncMock(
            return_value=SimpleNamespace(
                id="run_parent",
                user_id="user_1",
                status="in_progress",
                meta_data=parent_metadata,
            )
        ),
        create_thread=AsyncMock(return_value=SimpleNamespace(id="thread_child")),
        create_message=AsyncMock(return_value=SimpleNamespace(id="message_child")),
        create_run=AsyncMock(side_effect=RuntimeError("stop before worker stream")),
        update_run_fields=AsyncMock(),
        update_action_status=AsyncMock(),
    )

    mixin._native_exec = native
    mixin._ephemeral_clean_up = AsyncMock()
    mixin.submit_tool_output = AsyncMock()

    async def invoke():
        generator = mixin.handle_delegate_research_task(
            thread_id="thread_1",
            run_id="run_parent",
            assistant_id="assistant_1",
            arguments_dict={"task": "research something"},
            tool_call_id="call_1",
            decision="yes",
            inference_api_key="provider-key",
            project_david_api_key="project-key",
        )

        return [json.loads(event) async for event in generator]

    events = asyncio.run(invoke())

    native.create_run.assert_awaited_once()

    kwargs = native.create_run.await_args.kwargs

    assert kwargs["assistant_id"] == "assistant_child"
    assert kwargs["thread_id"] == "thread_child"
    assert kwargs["user_id"] == "user_1"
    assert kwargs["model"] == "hyperbolic/delegated"
    assert kwargs["trusted_metadata"] is True

    assert kwargs["meta_data"] == {
        "batfish_owner_user_id": "user_1",
        "scratch_pad_thread": "thread_1",
        "root_run_id": "run_parent",
        "parent_run_id": "run_parent",
        "parent_action_id": "action_1",
    }

    assert "api_key" not in kwargs["meta_data"]
    assert "project_david_api_key" not in kwargs["meta_data"]

    assert parent_metadata == {"delegated_model": "hyperbolic/delegated"}

    assert events[-1]["state"] == "error"
    assert not any(
        event["state"] == "completed"
        for event in events
        if event.get("type") == "research_status"
    )


def test_buffered_error_closes_execution_before_returning():
    invoke, selector, _, _, _ = _run_stateful_with_mocks(
        "together-ai/base-model", "together-ai/base-model"
    )
    finalized = []

    async def worker(**kwargs):
        try:
            yield {"type": "error", "message": "inference failed"}
            pytest.fail("Buffered execution must stop consuming after an error")
        finally:
            finalized.append(True)

    selector.select_provider.return_value = (
        SimpleNamespace(process_conversation=worker),
        "provider-model",
    )

    async def check():
        with pytest.raises(HTTPException) as error:
            await invoke()
        assert error.value.status_code == 500
        assert error.value.detail == "inference failed"
        # Assert before asyncio.run() can finalize orphaned generators.
        assert finalized == [True]

    asyncio.run(check())


@pytest.mark.parametrize("exit_path", ["explicit-close", "send-failure", "cancel"])
def test_stream_shutdown_closes_suspended_execution(exit_path):
    finalized = []

    async def worker(**kwargs):
        try:
            yield {"type": "content", "content": "first chunk"}
            pytest.fail("Execution must not resume after the consumer stops")
        finally:
            await asyncio.sleep(0)
            finalized.append(True)

    async def consume(response):
        if exit_path == "explicit-close":
            first = await anext(response.body_iterator)
            assert "first chunk" in first
            await response.body_iterator.aclose()
        elif exit_path == "send-failure":

            async def send(message):
                if message["type"] == "http.response.body":
                    raise OSError("client disconnected")

            with pytest.raises(OSError, match="client disconnected"):
                await response.stream_response(send)
        else:
            from anyio import CancelScope

            with CancelScope() as scope:

                async def send(message):
                    if message["type"] == "http.response.body":
                        scope.cancel()
                        await asyncio.sleep(0)

                await response.stream_response(send)

        # Assert before event-loop shutdown can run delayed finalizers.
        assert finalized == [True]

    invoke, selector, _, _, _ = _run_stateful_with_mocks(
        "together-ai/base-model",
        "together-ai/base-model",
        stream=True,
        response_consumer=consume,
    )
    selector.select_provider.return_value = (
        SimpleNamespace(process_conversation=worker),
        "provider-model",
    )
    asyncio.run(invoke())
