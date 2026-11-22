"""BYOK HTTP management contract: ownership, metadata-only output, errors."""

from datetime import datetime
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from projectdavid_common import ValidationInterface

from src.api.entities_api.dependencies import get_api_key
from src.api.entities_api.routers.provider_keys_router import (
    get_provider_key_service,
    router,
)
from src.api.entities_api.services.credential_service import (
    CredentialConfigurationError,
)
from src.api.entities_api.services.provider_key_service import (
    ProviderKeyConfigurationError,
    ProviderKeyNotFoundError,
    ProviderKeyUnavailableError,
)

validator = ValidationInterface()


def _key(name: str = "Primary") -> validator.ProviderKeyRead:
    return validator.ProviderKeyRead(
        id="pkey_owned",
        owner_id="tenant_a",
        provider="hyperbolic",
        name=name,
        created_at=datetime(2026, 1, 1),
        revoked_at=None,
    )


class FakeProviderKeyService:
    def __init__(self):
        self.calls = []
        self.created = True
        self.registration_error = None
        self.revoke_error = None

    def register(self, registration, *, user_id):
        self.calls.append(("register", user_id, registration))
        if self.registration_error:
            raise self.registration_error
        return _key(), self.created

    def list_keys(self, *, user_id):
        self.calls.append(("list", user_id))
        return [_key()]

    def get_key(self, *, provider_key_id, user_id):
        self.calls.append(("get", provider_key_id, user_id))
        if provider_key_id != "pkey_owned" or user_id != "tenant_a":
            raise ProviderKeyNotFoundError("not found")
        return _key()

    def update_key(self, *, provider_key_id, user_id, update):
        self.calls.append(("update", provider_key_id, user_id, update))
        if provider_key_id != "pkey_owned" or user_id != "tenant_a":
            raise ProviderKeyNotFoundError("not found")
        return _key(name=update.name)

    def revoke(self, *, provider_key_id, user_id):
        self.calls.append(("revoke", provider_key_id, user_id))
        if self.revoke_error:
            raise self.revoke_error
        if provider_key_id != "pkey_owned" or user_id != "tenant_a":
            raise ProviderKeyNotFoundError("not found")


@pytest.fixture()
def client_and_service():
    service = FakeProviderKeyService()
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_api_key] = lambda: SimpleNamespace(user_id="tenant_a")
    app.dependency_overrides[get_provider_key_service] = lambda: service
    with TestClient(app) as client:
        yield client, service


def test_register_returns_safe_metadata_and_uses_authenticated_owner(
    client_and_service,
):
    client, service = client_and_service
    response = client.post(
        "/provider-keys",
        json={
            "provider": "Hyperbolic",
            "api_key": "never-reflect-this-secret",
            "name": "Primary",
        },
    )
    assert response.status_code == 201
    assert response.json()["owner_id"] == "tenant_a"
    assert response.json()["provider"] == "hyperbolic"
    assert "api_key" not in response.json()
    assert "credential_id" not in response.json()
    assert "fingerprint" not in response.json()
    assert "never-reflect-this-secret" not in response.text
    operation, owner, registration = service.calls[-1]
    assert (operation, owner) == ("register", "tenant_a")
    assert registration.api_key.get_secret_value() == "never-reflect-this-secret"


def test_duplicate_registration_returns_200(client_and_service):
    client, service = client_and_service
    service.created = False
    response = client.post(
        "/provider-keys",
        json={"provider": "hyperbolic", "api_key": "secret"},
    )
    assert response.status_code == 200


def test_registration_cannot_override_tenant(client_and_service):
    client, service = client_and_service
    response = client.post(
        "/provider-keys",
        json={
            "provider": "hyperbolic",
            "api_key": "secret",
            "owner_id": "tenant_b",
        },
    )
    assert response.status_code == 422
    assert not service.calls


def test_list_and_get_are_tenant_scoped(client_and_service):
    client, service = client_and_service
    listed = client.get("/provider-keys")
    assert listed.status_code == 200
    assert listed.json()[0]["id"] == "pkey_owned"
    assert service.calls[-1] == ("list", "tenant_a")

    fetched = client.get("/provider-keys/pkey_owned")
    assert fetched.status_code == 200
    assert fetched.json()["owner_id"] == "tenant_a"
    assert service.calls[-1] == ("get", "pkey_owned", "tenant_a")

    missing = client.get("/provider-keys/pkey_other")
    assert missing.status_code == 404
    assert missing.json() == {"detail": "Provider key not found"}


def test_rename_is_metadata_only(client_and_service):
    client, service = client_and_service
    response = client.patch("/provider-keys/pkey_owned", json={"name": "Renamed"})
    assert response.status_code == 200
    assert response.json()["name"] == "Renamed"
    operation, key_id, owner, update = service.calls[-1]
    assert (operation, key_id, owner, update.name) == (
        "update",
        "pkey_owned",
        "tenant_a",
        "Renamed",
    )

    forbidden = client.patch(
        "/provider-keys/pkey_owned",
        json={"name": "Other", "api_key": "replacement-secret"},
    )
    assert forbidden.status_code == 422


def test_rename_missing_key_is_404(client_and_service):
    client, _ = client_and_service
    response = client.patch("/provider-keys/pkey_other", json={"name": "Nope"})
    assert response.status_code == 404


def test_revoke_returns_204_and_preserves_tenant_ownership(client_and_service):
    client, service = client_and_service
    response = client.delete("/provider-keys/pkey_owned")
    assert response.status_code == 204
    assert response.content == b""
    assert service.calls[-1] == ("revoke", "pkey_owned", "tenant_a")
    assert client.delete("/provider-keys/pkey_other").status_code == 404


def test_revoke_unavailable_key_is_safe_conflict(client_and_service):
    client, service = client_and_service
    service.revoke_error = ProviderKeyUnavailableError("internal storage detail")
    response = client.delete("/provider-keys/pkey_owned")
    assert response.status_code == 409
    assert "internal storage detail" not in response.text


@pytest.mark.parametrize(
    "exception",
    [
        ProviderKeyConfigurationError("internal config"),
        CredentialConfigurationError("internal config"),
    ],
)
def test_registration_fail_closed_on_missing_crypto_configuration(
    client_and_service, exception
):
    client, service = client_and_service
    service.registration_error = exception
    response = client.post(
        "/provider-keys",
        json={"provider": "hyperbolic", "api_key": "secret"},
    )
    assert response.status_code == 503
    assert "internal config" not in response.text


def test_unauthed_request_is_rejected():
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        response = client.get("/provider-keys")
    assert response.status_code == 401
