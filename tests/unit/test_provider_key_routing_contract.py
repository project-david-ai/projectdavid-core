"""Registered provider dispatch and legacy compatibility regressions."""

import importlib

import pytest
from fastapi import HTTPException

inference_router = importlib.import_module(
    "src.api.entities_api.routers.inference_router"
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model,provider",
    [
        ("hyperbolic/kimi-k3", "hyperbolic"),
        ("together-ai/meta-llama", "together-ai"),
        ("deepseek-ai/deepseek-flash", "deepseek-ai"),
    ],
)
async def test_registered_provider_is_scoped(monkeypatch, model, provider):
    observed = {}

    async def resolve(**kwargs):
        observed.update(kwargs)
        return "resolved-in-memory-only"

    monkeypatch.setattr(
        inference_router.NativeExecutionService,
        "resolve_provider_key_for_inference",
        resolve,
    )
    result = await inference_router._resolve_provider_api_key(
        model=model,
        user_id="owner-a",
        provider_key_id="pkey-owned",
        raw_api_key=None,
    )
    assert result == "resolved-in-memory-only"
    assert observed == {
        "provider_key_id": "pkey-owned",
        "user_id": "owner-a",
        "provider": provider,
    }


@pytest.mark.asyncio
async def test_raw_hyperbolic_key_rejected():
    with pytest.raises(HTTPException) as exc:
        await inference_router._resolve_provider_api_key(
            model="hyperbolic/kimi-k3",
            user_id="owner-a",
            provider_key_id=None,
            raw_api_key="raw-secret",
        )
    assert exc.value.status_code == 422


@pytest.mark.asyncio
async def test_raw_legacy_together_key_still_supported():
    assert (
        await inference_router._resolve_provider_api_key(
            model="together-ai/some-model",
            user_id="owner-a",
            provider_key_id=None,
            raw_api_key="legacy-key",
        )
        == "legacy-key"
    )


@pytest.mark.asyncio
async def test_registered_key_rejected_for_local_provider():
    with pytest.raises(HTTPException) as exc:
        await inference_router._resolve_provider_api_key(
            model="vllm/Qwen/Qwen2.5-3B-Instruct",
            user_id="owner-a",
            provider_key_id="pkey-owned",
            raw_api_key=None,
        )
    assert exc.value.status_code == 422


@pytest.mark.asyncio
async def test_registered_child_stream_never_transmits_plaintext_key(monkeypatch):
    import json

    import httpx

    from src.api.entities_api.orchestration.mixins import delegation_mixin

    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = json.loads(request.content)
        assert body["provider_key_id"] == "pkey-child"
        assert "api_key" not in body
        assert request.headers["x-api-key"] == "core-test-auth"
        return httpx.Response(
            200,
            text='data: {"type":"content","content":"ok"}\n\ndata: [DONE]\n\n',
            headers={"content-type": "text/event-stream"},
        )

    actual_client = httpx.AsyncClient

    def mock_client(*args, **kwargs):
        return actual_client(*args, transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(delegation_mixin.httpx, "AsyncClient", mock_client)
    adapter = delegation_mixin._RegisteredProviderInferenceClient(
        base_url="http://core.internal",
        core_api_key="core-test-auth",
        provider_key_id="pkey-child",
    )
    chunks = [
        chunk
        async for chunk in adapter.stream_inference_response(
            model="hyperbolic/kimi-k3",
            thread_id="t",
            message_id="m",
            run_id="r",
            assistant_id="a",
            api_key=None,
        )
    ]
    assert chunks == [{"type": "content", "content": "ok"}]
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_registered_child_rejects_plaintext_override():
    from src.api.entities_api.orchestration.mixins.delegation_mixin import (
        _RegisteredProviderInferenceClient,
    )

    adapter = _RegisteredProviderInferenceClient(
        base_url="http://core.internal",
        core_api_key="core-test-auth",
        provider_key_id="pkey-child",
    )
    with pytest.raises(RuntimeError, match="plaintext"):
        async for _ in adapter.stream_inference_response(
            model="hyperbolic/kimi-k3",
            thread_id="t",
            message_id="m",
            run_id="r",
            assistant_id="a",
            api_key="should-not-send",
        ):
            pass
