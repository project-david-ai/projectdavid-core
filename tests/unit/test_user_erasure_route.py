"""Administrative account deletion must use the durable coordinator."""

from importlib import import_module
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from src.api.entities_api.services import account_erasure
from src.api.entities_api.services.user_service import UserService

users_router = import_module("src.api.entities_api.routers.users_router")


@pytest.fixture
def harness(monkeypatch):
    db = MagicMock()
    actor = SimpleNamespace(id="admin", is_admin=True)
    db.query.return_value.filter.return_value.first.return_value = actor
    key = SimpleNamespace(user_id="admin")
    coordinator = MagicMock()
    monkeypatch.setattr(account_erasure, "erase_user", coordinator)
    app = FastAPI()
    app.include_router(users_router.router, prefix="/v1")
    app.dependency_overrides[users_router.get_db] = lambda: db
    app.dependency_overrides[users_router.get_api_key] = lambda: key
    with TestClient(app) as client:
        yield SimpleNamespace(
            client=client, db=db, actor=actor, key=key, coordinator=coordinator
        )


def test_admin_deletion_delegates_and_returns_empty_204(harness):
    h = harness
    response = h.client.delete("/v1/users/fixture")
    assert response.status_code == 204
    assert response.content == b""
    h.coordinator.assert_called_once()
    storage, uid = h.coordinator.call_args.args
    assert isinstance(storage, UserService)
    assert uid == "fixture"


@pytest.mark.parametrize("code", [404, 409, 503])
def test_coordinator_failure_status_is_preserved(harness, code):
    h = harness
    h.coordinator.side_effect = HTTPException(code, "Reconciliation status")
    response = h.client.delete("/v1/users/fixture")
    assert response.status_code == code
    assert response.json() == {"detail": "Reconciliation status"}
    h.coordinator.assert_called_once()


def test_non_admin_cannot_request_deletion(harness):
    h = harness
    h.actor.is_admin = False
    assert h.client.delete("/v1/users/fixture").status_code == 403
    h.coordinator.assert_not_called()


def test_admin_self_deletion_is_disallowed(harness):
    h = harness
    assert h.client.delete("/v1/users/admin").status_code == 403
    h.coordinator.assert_not_called()


def test_missing_requesting_user_is_denied(harness):
    h = harness
    h.db.query.return_value.filter.return_value.first.return_value = None
    assert h.client.delete("/v1/users/fixture").status_code == 403
    h.coordinator.assert_not_called()


def test_unexpected_error_does_not_report_completion(harness):
    h = harness
    h.coordinator.side_effect = RuntimeError("private failure")
    response = h.client.delete("/v1/users/fixture")
    assert response.status_code == 500
    assert "private failure" not in response.text


def test_internal_service_caller_uses_same_coordinator(harness):
    service = UserService()
    service.erase_user("fixture")
    harness.coordinator.assert_called_once_with(service, "fixture")
