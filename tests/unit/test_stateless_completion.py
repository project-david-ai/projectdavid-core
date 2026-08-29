import asyncio
import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import TypeAdapter, ValidationError

from src.api.entities_api.services.stateless_inference_service import (
    StatelessInferenceService,
)

inference_router = importlib.import_module(
    "src.api.entities_api.routers.inference_router"
)


def _stateful_payload(**overrides):
    payload = {
        "model": "together-ai/example/model",
        "api_key": "provider-key",
        "thread_id": "thread_1",
        "message_id": "message_1",
        "run_id": "run_1",
        "assistant_id": "assistant_1",
        "stream": False,
    }
    payload.update(overrides)
    return payload


def _decode_response(response):
    return json.loads(response.body.decode("utf-8"))


def test_stateless_completion_succeeds_without_conversation_identifiers():
    request = inference_router.StatelessCompletionRequest(
        model="together-ai/example/model",
        prompt="Name this conversation",
        stateless=True,
        stream=False,
        max_tokens=24,
    )
    service = MagicMock()
    service.create_completion = AsyncMock(return_value="Project David Lifecycle")

    with (
        patch.object(
            inference_router,
            "StatelessInferenceService",
            return_value=service,
        ),
        patch.object(
            inference_router,
            "_authenticate_stateless_request",
            new=AsyncMock(),
        ),
    ):
        response = asyncio.run(
            inference_router.completions(
                request,
                redis=MagicMock(),
                project_api_key="pd-key",
            )
        )

    body = _decode_response(response)
    assert body["choices"][0]["text"] == "Project David Lifecycle"
    assert body["object"] == "text_completion"
    service.create_completion.assert_awaited_once()
    call = service.create_completion.await_args.kwargs
    assert call["messages"] == [{"role": "user", "content": "Name this conversation"}]
    assert not {
        "thread_id",
        "message_id",
        "run_id",
        "assistant_id",
    }.intersection(call)


def test_stateless_route_skips_stateful_conversation_machinery():
    request = inference_router.StatelessCompletionRequest(
        model="together-ai/example/model",
        prompt="Name this conversation",
        stateless=True,
    )
    service = MagicMock()
    service.create_completion = AsyncMock(return_value="Lifecycle Hardening")

    with (
        patch.object(
            inference_router,
            "StatelessInferenceService",
            return_value=service,
        ),
        patch.object(inference_router, "NativeExecutionService") as native,
        patch.object(inference_router, "InferenceProviderSelector") as selector,
        patch.object(
            inference_router,
            "_authenticate_stateless_request",
            new=AsyncMock(),
        ),
    ):
        asyncio.run(
            inference_router.completions(
                request,
                redis=MagicMock(),
                project_api_key="pd-key",
            )
        )

    native.assert_not_called()
    selector.assert_not_called()


def test_stateless_route_authenticates_with_project_david_api_key():
    request = inference_router.StatelessCompletionRequest(
        model="vllm/bm_example",
        prompt="Name this conversation",
        stateless=True,
    )
    service = MagicMock()
    service.create_completion = AsyncMock(return_value="Authenticated title")
    authenticate = AsyncMock()

    with (
        patch.object(
            inference_router,
            "StatelessInferenceService",
            return_value=service,
        ),
        patch.object(
            inference_router,
            "_authenticate_stateless_request",
            new=authenticate,
        ),
    ):
        asyncio.run(
            inference_router.completions(
                request,
                redis=MagicMock(),
                project_api_key="pd-key",
            )
        )

    authenticate.assert_awaited_once_with("pd-key")


def test_stateless_service_calls_low_level_provider_once_without_tools():
    chunks = [
        {"choices": [{"delta": {"content": "Project David"}}]},
        {"choices": [{"delta": {"content": " Lifecycle"}}]},
    ]

    async def stream():
        for chunk in chunks:
            yield chunk

    client = MagicMock()
    client.stream_chat_completion.return_value = stream()
    worker = MagicMock(spec=[])
    worker._get_client_instance = MagicMock(return_value=client)
    worker.process_conversation = AsyncMock()
    worker.process_tool_calls = AsyncMock()
    selector = MagicMock()
    selector.select_provider_worker.return_value = (worker, "example/model")

    service = StatelessInferenceService(selector=selector)
    result = asyncio.run(
        service.create_completion(
            model="together-ai/example/model",
            messages=[{"role": "user", "content": "Name this"}],
            provider_api_key="provider-key",
            max_tokens=24,
            temperature=0.2,
            top_p=1.0,
        )
    )

    assert result == "Project David Lifecycle"
    client.stream_chat_completion.assert_called_once()
    request = client.stream_chat_completion.call_args.kwargs
    assert "tools" not in request
    assert request["max_tokens"] == 24
    worker.process_conversation.assert_not_awaited()
    worker.process_tool_calls.assert_not_awaited()


def test_stateless_service_uses_canonical_provider_and_model_routing():
    async def stream():
        yield {"choices": [{"text": "Routed title"}]}

    client = MagicMock()
    client.stream_chat_completion.return_value = stream()
    worker = MagicMock(spec=[])
    worker._get_client_instance = MagicMock(return_value=client)
    selector = MagicMock()
    selector.select_provider_worker.return_value = (worker, "provider/model")

    service = StatelessInferenceService(selector=selector)
    asyncio.run(
        service.create_completion(
            model="together-ai/provider/model",
            messages=[{"role": "user", "content": "Name this"}],
            provider_api_key="provider-key",
            max_tokens=24,
            temperature=0.2,
            top_p=1.0,
        )
    )

    selector.select_provider_worker.assert_called_once_with(
        "together-ai/provider/model"
    )
    assert client.stream_chat_completion.call_args.kwargs["model"] == ("provider/model")


def test_stateless_local_resolution_performs_no_conversation_database_writes():
    async def stream(**kwargs):
        assert kwargs["tools"] is None
        yield {"choices": [{"delta": {"content": "Local title"}}]}

    worker = MagicMock(spec=[])
    worker.base_url = "http://vllm:8000"
    worker._stream_vllm_raw = MagicMock(side_effect=stream)
    worker.process_conversation = AsyncMock()
    selector = MagicMock()
    selector.select_provider_worker.return_value = (worker, "bm_example")
    db = MagicMock()

    service = StatelessInferenceService(
        selector=selector,
        session_factory=MagicMock(return_value=db),
    )
    with patch(
        "src.api.entities_api.services.stateless_inference_service."
        "InferenceResolver.resolve_vllm_url",
        return_value="http://ray/vllm_dep_example",
    ):
        result = asyncio.run(
            service.create_completion(
                model="vllm/bm_example",
                messages=[{"role": "user", "content": "Name this"}],
                provider_api_key=None,
                max_tokens=24,
                temperature=0.2,
                top_p=1.0,
            )
        )

    assert result == "Local title"
    db.add.assert_not_called()
    db.commit.assert_not_called()
    db.delete.assert_not_called()
    db.close.assert_called_once_with()
    worker.process_conversation.assert_not_awaited()


def test_stateless_provider_error_is_not_returned_as_completion_text():
    async def stream():
        yield {
            "choices": [
                {
                    "delta": {"content": "[vLLM error 503]"},
                    "finish_reason": "error",
                }
            ]
        }

    worker = MagicMock(spec=[])
    worker._get_client_instance = MagicMock(
        return_value=SimpleNamespace(
            stream_chat_completion=MagicMock(return_value=stream())
        )
    )
    selector = MagicMock()
    selector.select_provider_worker.return_value = (worker, "provider/model")
    service = StatelessInferenceService(selector=selector)

    with pytest.raises(RuntimeError, match="vLLM error 503"):
        asyncio.run(
            service.create_completion(
                model="together-ai/provider/model",
                messages=[{"role": "user", "content": "Name this"}],
                provider_api_key="provider-key",
                max_tokens=24,
                temperature=0.2,
                top_p=1.0,
            )
        )


def test_stateless_messages_are_supported_without_memory_hydration():
    request = inference_router.StatelessCompletionRequest(
        model="vllm/bm_example",
        messages=[
            {"role": "system", "content": "Return a short title."},
            {"role": "user", "content": "Explain BGP route reflectors."},
        ],
        stateless=True,
    )

    assert request.inference_messages() == [
        {"role": "system", "content": "Return a short title."},
        {"role": "user", "content": "Explain BGP route reflectors."},
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {"model": "vllm/bm_example", "stateless": True},
        {
            "model": "vllm/bm_example",
            "prompt": "   ",
            "stateless": True,
        },
        {
            "model": "vllm/bm_example",
            "prompt": "Name this",
            "stream": True,
            "stateless": True,
        },
        {
            "model": "vllm/bm_example",
            "prompt": "Name this",
            "tools": [{"type": "function"}],
            "stateless": True,
        },
    ],
)
def test_invalid_stateless_payloads_fail_validation(payload):
    with pytest.raises(ValidationError):
        TypeAdapter(inference_router.CompletionRequest).validate_python(payload)


@pytest.mark.parametrize("stateless_value", [None, False])
def test_stateful_contract_is_selected_when_stateless_is_absent_or_false(
    stateless_value,
):
    payload = _stateful_payload()
    if stateless_value is not None:
        payload["stateless"] = stateless_value

    request = TypeAdapter(inference_router.CompletionRequest).validate_python(payload)

    assert isinstance(request, inference_router.StatefulCompletionRequest)
    assert request.thread_id == "thread_1"
    assert request.run_id == "run_1"


def test_stateful_identifier_validation_remains_required():
    payload = _stateful_payload()
    del payload["run_id"]

    with pytest.raises(ValidationError) as new_error:
        TypeAdapter(inference_router.CompletionRequest).validate_python(payload)
    with pytest.raises(ValidationError) as existing_error:
        inference_router.ValidationInterface.StreamRequest.model_validate(payload)

    assert any(error["loc"][-1] == "run_id" for error in new_error.value.errors())
    assert any(error["loc"][-1] == "run_id" for error in existing_error.value.errors())


def test_stateful_buffered_completion_retains_existing_processing_path():
    request = inference_router.StatefulCompletionRequest(**_stateful_payload())
    native = MagicMock()
    native.retrieve_run = AsyncMock(
        return_value=SimpleNamespace(
            thread_id="thread_1",
            assistant_id="assistant_1",
            user_id="user_1",
        )
    )
    native.assert_assistant_access = AsyncMock()

    class Handler:
        async def process_conversation(self, **kwargs):
            yield {"type": "content", "content": "Existing behavior"}

    selector = MagicMock()
    selector.select_provider.return_value = (Handler(), "provider/model")

    with (
        patch.object(
            inference_router,
            "NativeExecutionService",
            return_value=native,
        ),
        patch.object(inference_router, "InferenceArbiter", return_value=MagicMock()),
        patch.object(
            inference_router,
            "InferenceProviderSelector",
            return_value=selector,
        ),
    ):
        response = asyncio.run(inference_router.completions(request, redis=MagicMock()))

    body = _decode_response(response)
    assert body == {
        "run_id": "run_1",
        "content": "Existing behavior",
        "type": "content",
        "model": "together-ai/example/model",
        "elapsed_s": body["elapsed_s"],
    }
    native.retrieve_run.assert_awaited_once_with("run_1")
    native.assert_assistant_access.assert_awaited_once_with(
        assistant_id="assistant_1",
        user_id="user_1",
    )
    selector.select_provider.assert_called_once_with(
        model_id="together-ai/example/model"
    )
