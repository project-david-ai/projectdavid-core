"""BYOK credential resolution boundary: Hyperbolic first, legacy elsewhere."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from src.api.entities_api.routers.inference_router import (
    StatefulCompletionRequest,
    StatelessCompletionRequest,
    _authenticate_stateless_request,
    _resolve_provider_api_key,
)
from src.api.entities_api.services.credential_service import (
    CredentialConfigurationError,
)
from src.api.entities_api.services.native_execution_service import (
    NativeExecutionService,
)
from src.api.entities_api.services.provider_key_service import (
    ProviderKeyConfigurationError,
    ProviderKeyNotFoundError,
    ProviderKeyUnavailableError,
)


def resolve(
    model="hyperbolic/meta-llama/llama",
    provider_key_id="pkey_A",
    raw_api_key=None,
):
    return asyncio.run(
        _resolve_provider_api_key(
            model=model,
            user_id="tenant_A",
            provider_key_id=provider_key_id,
            raw_api_key=raw_api_key,
        )
    )


def test_hyperbolic_registered_key_resolves_for_authenticated_owner(monkeypatch):
    observed = []

    async def fake_resolve(**kwargs):
        observed.append(kwargs)
        return "sensitive-provider-secret"

    monkeypatch.setattr(
        NativeExecutionService,
        "resolve_provider_key_for_inference",
        staticmethod(fake_resolve),
    )

    assert resolve() == "sensitive-provider-secret"
    assert observed == [
        {
            "provider_key_id": "pkey_A",
            "user_id": "tenant_A",
            "provider": "hyperbolic",
        }
    ]


@pytest.mark.parametrize(
    "model", ["vllm/meta-llama", "ollama/llama3", " VLLM/deployment "]
)
def test_keyless_local_models_are_unchanged(model):
    assert resolve(model=model, provider_key_id=None) is None


def test_unmigrated_provider_retains_legacy_raw_key():
    assert (
        resolve(
            model="together-ai/meta-llama/llama",
            provider_key_id=None,
            raw_api_key="legacy-key",
        )
        == "legacy-key"
    )


@pytest.mark.parametrize("model", ["vllm/test", "ollama/test"])
def test_registered_keys_not_forwarded_to_unmigrated_providers(model):
    with pytest.raises(HTTPException) as exc:
        resolve(model=model)
    assert exc.value.status_code == 422


@pytest.mark.parametrize("provided", ["plaintext", ""])
def test_direct_hyperbolic_key_rejected_even_with_registered_key(provided, monkeypatch):
    resolver = AsyncMock(return_value="resolved")
    monkeypatch.setattr(
        NativeExecutionService,
        "resolve_provider_key_for_inference",
        staticmethod(resolver),
    )
    with pytest.raises(HTTPException) as exc:
        resolve(raw_api_key=provided)
    assert exc.value.status_code == 422
    assert "plaintext" not in exc.value.detail
    resolver.assert_not_awaited()


def test_hyperbolic_key_required_before_dispatch():
    with pytest.raises(HTTPException) as exc:
        resolve(provider_key_id=None)
    assert exc.value.status_code == 422


@pytest.mark.parametrize(
    "error",
    [
        ProviderKeyNotFoundError("another tenant"),
        ProviderKeyUnavailableError("revoked or wrong provider"),
    ],
)
def test_foreign_revoked_or_wrong_provider_hidden_from_caller(error, monkeypatch):
    async def fake_resolve(**_kwargs):
        raise error

    monkeypatch.setattr(
        NativeExecutionService,
        "resolve_provider_key_for_inference",
        staticmethod(fake_resolve),
    )
    with pytest.raises(HTTPException) as exc:
        resolve()
    assert exc.value.status_code == 404
    assert exc.value.detail == "Provider key unavailable."


@pytest.mark.parametrize(
    "error",
    [
        ProviderKeyConfigurationError("secret configuration"),
        CredentialConfigurationError("credential configuration"),
    ],
)
def test_security_config_errors_fail_closed(error, monkeypatch):
    async def fake_resolve(**_kwargs):
        raise error

    monkeypatch.setattr(
        NativeExecutionService,
        "resolve_provider_key_for_inference",
        staticmethod(fake_resolve),
    )
    with pytest.raises(HTTPException) as exc:
        resolve()
    assert exc.value.status_code == 503
    assert "secret configuration" not in exc.value.detail


def test_stateful_contract_accepts_registered_reference():
    request = StatefulCompletionRequest(
        model="hyperbolic/meta-llama/llama",
        thread_id="thread_a",
        message_id="message_a",
        assistant_id="assistant_a",
        run_id="run_a",
        provider_key_id="pkey_A",
    )
    assert request.provider_key_id == "pkey_A"
    assert request.api_key is None


def test_stateless_contract_accepts_registered_reference():
    request = StatelessCompletionRequest(
        model="hyperbolic/meta-llama/llama",
        prompt="hello",
        stateless=True,
        provider_key_id="pkey_A",
    )
    assert request.provider_key_id == "pkey_A"
    assert request.api_key is None


def test_stateless_auth_returns_trusted_owner(monkeypatch):
    import importlib

    inference_router = importlib.import_module(
        "src.api.entities_api.routers.inference_router"
    )

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    authenticate = AsyncMock(return_value=SimpleNamespace(user_id="tenant_A"))
    monkeypatch.setattr(inference_router, "SessionLocal", FakeSession)
    monkeypatch.setattr(inference_router, "get_api_key", authenticate)

    assert asyncio.run(_authenticate_stateless_request("project-key")) == "tenant_A"
    assert authenticate.await_args.kwargs["api_key_header"] == "project-key"


def test_native_execution_service_resolves_without_sdk(monkeypatch):
    from src.api.entities_api.services import provider_key_service

    observed = []

    class FakeProviderKeyService:
        def resolve_for_inference(self, **kwargs):
            observed.append(kwargs)
            return "resolved-internal-secret"

    monkeypatch.setattr(
        provider_key_service, "ProviderKeyService", FakeProviderKeyService
    )
    result = asyncio.run(
        NativeExecutionService.resolve_provider_key_for_inference(
            provider_key_id="pkey_A",
            user_id="tenant_A",
            provider="hyperbolic",
        )
    )
    assert result == "resolved-internal-secret"
    assert observed == [
        {
            "provider_key_id": "pkey_A",
            "user_id": "tenant_A",
            "provider": "hyperbolic",
        }
    ]
