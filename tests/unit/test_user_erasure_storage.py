"""Erasure must retain durable storage locators when cleanup is unconfirmed."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from smb.smb_structs import OperationFailure
from sqlalchemy.engine import create_engine
from sqlalchemy.orm import sessionmaker

from src.api.entities_api.models.models import File, FileStorage, User, VectorStore
from src.api.entities_api.services import user_service as module
from src.api.entities_api.services.vector_runtime.base_vector_store import (
    VectorStoreError,
)
from src.api.entities_api.services.vector_runtime.vector_store_manager import (
    VectorStoreManager,
)
from src.api.entities_api.utilities.samba_client import SambaClient


@pytest.fixture
def storage_db(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    for model in (User, File, FileStorage, VectorStore):
        model.__table__.create(engine)
    factory = sessionmaker(engine)
    monkeypatch.setattr(module, "SessionLocal", factory)
    with factory() as db:
        db.add(User(id="owner"))
        db.add(
            File(
                id="file",
                user_id="owner",
                bytes=4,
                filename="private.txt",
                purpose="assistants",
            )
        )
        db.flush()
        db.add(
            FileStorage(
                file_id="file", storage_system="samba", storage_path="private/path"
            )
        )
        db.commit()
    yield factory
    engine.dispose()


def test_samba_failure_keeps_user_and_locator(storage_db, monkeypatch):
    samba = Mock()
    samba.delete_file.side_effect = PermissionError("private path and credentials")
    monkeypatch.setattr(module, "SambaClient", lambda *a: samba)
    service = module.UserService()
    later = Mock()
    monkeypatch.setattr(service, "_erase_vector_store_collections", later)
    with storage_db() as db:
        with pytest.raises(HTTPException) as error:
            service._erase_physical_files(db, "owner")
    assert error.value.status_code == 503
    assert "private" not in error.value.detail
    later.assert_not_called()
    with storage_db() as db:
        assert db.get(User, "owner") is not None
        assert db.query(FileStorage).one().storage_path == "private/path"
    samba.conn.close.assert_called_once()


def test_unsupported_storage_keeps_locator(storage_db, monkeypatch):
    with storage_db() as db:
        db.query(FileStorage).update({"storage_system": "s3"})
        db.commit()
    connect = Mock()
    monkeypatch.setattr(module, "SambaClient", connect)
    with pytest.raises(HTTPException) as error:
        module.UserService().erase_user("owner")
    assert error.value.status_code == 409
    connect.assert_not_called()
    with storage_db() as db:
        assert db.query(FileStorage).count() == 1


def test_retry_reuses_all_locators_after_partial_cleanup(storage_db, monkeypatch):
    with storage_db() as db:
        db.add(
            FileStorage(file_id="file", storage_system="samba", storage_path="second")
        )
        db.commit()
    samba = Mock()
    samba.delete_file.side_effect = [None, ConnectionError(), None, None]
    monkeypatch.setattr(module, "SambaClient", lambda *a: samba)
    service = module.UserService()
    with storage_db() as db:
        with pytest.raises(HTTPException):
            service._erase_physical_files(db, "owner")
        db.rollback()
    with storage_db() as db:
        service._erase_physical_files(db, "owner")
        assert db.query(FileStorage).count() == 2
    assert [call.args[0] for call in samba.delete_file.call_args_list] == [
        "private/path",
        "second",
        "private/path",
        "second",
    ]
    assert all(
        call.kwargs == {"missing_ok": True} for call in samba.delete_file.call_args_list
    )


def test_empty_account_does_not_connect_to_storage(storage_db, monkeypatch):
    samba, vector = Mock(), Mock()
    monkeypatch.setattr(module, "SambaClient", samba)
    monkeypatch.setattr(module, "VectorStoreManager", vector)
    with storage_db() as db:
        module.UserService()._erase_physical_files(db, "different-owner")
        module.UserService()._erase_vector_store_collections(db, "different-owner")
    samba.assert_not_called()
    vector.assert_not_called()


def test_qdrant_failure_keeps_storage_references(storage_db, monkeypatch):
    with storage_db() as db:
        db.add(
            VectorStore(
                id="store",
                name="test",
                user_id="owner",
                collection_name="collection",
                vector_size=3,
                distance_metric="COSINE",
                status="active",
            )
        )
        db.commit()
    manager = Mock()
    manager.delete_store.side_effect = ConnectionError("private topology")
    monkeypatch.setattr(module, "VectorStoreManager", lambda: manager)
    with storage_db() as db:
        with pytest.raises(HTTPException) as error:
            module.UserService()._erase_vector_store_collections(db, "owner")
        assert error.value.status_code == 503
        assert "private" not in error.value.detail
        db.rollback()
    with storage_db() as db:
        assert db.get(VectorStore, "store").collection_name == "collection"
    manager.client.close.assert_called_once()


@pytest.mark.parametrize("code", [0xC0000034, 0xC000003A])
def test_samba_missing_is_idempotent_only_when_requested(code):
    client = SambaClient.__new__(SambaClient)
    client.conn = Mock()
    client.share = "test"
    client.conn.deleteFiles.side_effect = OperationFailure(
        "missing", [SimpleNamespace(status=code)]
    )
    client.delete_file("path", missing_ok=True)
    with pytest.raises(OperationFailure):
        client.delete_file("path")


@pytest.mark.parametrize("code", [0xC0000022, 0xC00000B5, None])
def test_samba_permission_timeout_and_unknown_errors_propagate(code):
    client = SambaClient.__new__(SambaClient)
    client.conn = Mock()
    client.share = "test"
    client.conn.deleteFiles.side_effect = OperationFailure(
        "failed", [SimpleNamespace(status=code)]
    )
    with pytest.raises(OperationFailure):
        client.delete_file("path", missing_ok=True)


@pytest.mark.parametrize("confirmed,exists_after", [(False, False), (True, True)])
def test_qdrant_unconfirmed_deletion_fails(confirmed, exists_after):
    manager = VectorStoreManager.__new__(VectorStoreManager)
    manager.client = Mock()
    manager.client.collection_exists.side_effect = [True, exists_after]
    manager.client.delete_collection.return_value = confirmed
    with pytest.raises(VectorStoreError):
        manager.delete_store("collection")


def test_qdrant_absent_collection_is_success():
    manager = VectorStoreManager.__new__(VectorStoreManager)
    manager.client = Mock()
    manager.client.collection_exists.return_value = False
    assert manager.delete_store("collection")["status"] == "absent"
    manager.client.delete_collection.assert_not_called()
