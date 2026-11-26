"""Signed bearer URLs must respect account quarantine and HTTP drain."""

from importlib import import_module
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.entities_api.services.request_activity import (
    RequestActivityMiddleware,
    RequestActivityRegistry,
)

files_router = import_module("src.api.entities_api.routers.files_router")


@pytest.fixture
def download_harness(monkeypatch):
    registry = RequestActivityRegistry()
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = SimpleNamespace(
        user_id="owner"
    )
    admission_db = MagicMock()
    factory = MagicMock()
    factory.return_value.__enter__.return_value = admission_db
    monkeypatch.setattr(files_router, "SessionLocal", factory)
    receipt = MagicMock(return_value=None)
    monkeypatch.setattr(files_router, "erasure_state", receipt)
    signature = MagicMock(return_value=True)
    monkeypatch.setattr(files_router, "_verify_sig", signature)
    service = MagicMock()
    monkeypatch.setattr(files_router, "FileService", service)

    app = FastAPI()
    app.add_middleware(RequestActivityMiddleware, registry=registry)
    app.include_router(files_router.router, prefix="/v1")
    app.dependency_overrides[files_router.get_db] = lambda: db
    with TestClient(app) as client:
        yield SimpleNamespace(
            client=client,
            registry=registry,
            db=db,
            admission_db=admission_db,
            factory=factory,
            receipt=receipt,
            signature=signature,
            service=service,
        )


def download(h):
    return h.client.get(
        "/v1/files/download",
        params={"file_id": "file_test", "expires": 4102444800, "signature": "test"},
    )


@pytest.mark.parametrize("state", ["pending", "complete"])
def test_signed_download_rejects_quarantined_owner(download_harness, state):
    h = download_harness
    h.receipt.return_value = state
    response = download(h)
    assert response.status_code == 404
    assert response.json() == {"detail": "File not found"}
    h.receipt.assert_called_once_with(h.admission_db, "owner")
    h.service.assert_not_called()
    assert h.registry.active_count("owner") == 0


def test_signed_download_tracks_owner_before_receipt_and_through_stream(
    download_harness,
):
    h = download_harness
    observed = []

    def read_receipt(db, user_id):
        assert db is h.admission_db
        assert user_id == "owner"
        observed.append(h.registry.active_count("owner"))
        return None

    def stream():
        observed.append(h.registry.active_count("owner"))
        assert h.registry.active_count("other") == 0
        yield b"first"
        observed.append(h.registry.active_count("owner"))
        yield b"second"

    h.receipt.side_effect = read_receipt
    h.service.return_value.get_file_with_metadata.return_value = (
        stream(),
        "test.txt",
        "text/plain",
    )
    response = download(h)
    assert response.status_code == 200
    assert response.content == b"firstsecond"
    assert observed == [1, 1, 1]
    h.service.return_value.get_file_with_metadata.assert_called_once_with(
        "file_test", user_id="owner"
    )
    assert h.registry.active_count("owner") == 0


def test_invalid_signature_never_opens_storage(download_harness):
    h = download_harness
    h.signature.return_value = False
    assert download(h).status_code == 403
    h.db.query.assert_not_called()
    h.factory.assert_not_called()
    h.service.assert_not_called()


@pytest.mark.parametrize("record", [None, SimpleNamespace(user_id=None)])
def test_missing_file_or_owner_never_opens_storage(download_harness, record):
    h = download_harness
    h.db.query.return_value.filter.return_value.first.return_value = record
    assert download(h).status_code == 404
    h.factory.assert_not_called()
    h.service.assert_not_called()
