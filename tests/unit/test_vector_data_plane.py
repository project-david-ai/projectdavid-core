"""Storage boundary contracts without external services or embedding models."""

import ast
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from projectdavid_common import ValidationInterface as V
from pydantic import ValidationError
from qdrant_client import QdrantClient

from src.api.entities_api.services.vector_runtime.schemas import (
    VectorSearch,
    VectorUpsert,
)
from src.api.entities_api.services.vector_runtime.vector_store_manager import (
    VectorStoreManager,
)


def load_router():
    name = "src.api.entities_api.routers"
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(Path("src/api/entities_api/routers").resolve())]
        sys.modules[name] = package
    spec = importlib.util.spec_from_file_location(
        name + ".vectors_router", "src/api/entities_api/routers/vectors_router.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


r = load_router()


@pytest.fixture
def boundary(monkeypatch):
    events = []
    service = Mock()
    store = SimpleNamespace(
        id="store",
        collection_name="collection",
        user_id="owner",
        vector_size=2,
        status=V.StatusEnum.active,
    )
    service.get_vector_store_by_id.return_value = store
    runtime = Mock()
    runtime.create_store.side_effect = lambda **kw: events.append("physical")
    service.create_vector_store.side_effect = (
        lambda **kw: events.append("metadata") or store
    )
    runtime.add_to_store.return_value = {"points_inserted": 1, "point_ids": ["point"]}
    monkeypatch.setattr(r, "VectorStoreDBService", lambda db: service)
    monkeypatch.setattr(r, "get_vector_runtime", lambda: runtime)
    monkeypatch.setattr(r, "_is_admin", lambda *a: False)
    return SimpleNamespace(
        service=service,
        runtime=runtime,
        store=store,
        db=Mock(),
        auth=SimpleNamespace(user_id="owner"),
        events=events,
    )


def create(b):
    data = V.VectorStoreCreateWithSharedId(
        shared_id="store", name="test", vector_size=2, distance_metric="COSINE"
    )
    return r.create_vector_store(data, owner_id=None, db=b.db, auth_key=b.auth)


def test_create_order_and_compensation(boundary):
    b = boundary
    create(b)
    assert b.events == ["physical", "metadata"]
    b.service.create_vector_store.side_effect = r.VectorStoreDBError("failure")
    with pytest.raises(HTTPException) as exc:
        create(b)
    assert exc.value.status_code == 500
    b.runtime.delete_store.assert_called_once_with("store")


def test_storage_failure_prevents_db_creation(boundary):
    b = boundary
    b.runtime.create_store.side_effect = RuntimeError("private topology")
    with pytest.raises(HTTPException) as exc:
        create(b)
    assert exc.value.status_code == 503
    assert "private" not in exc.value.detail
    b.service.create_vector_store.assert_not_called()


@pytest.mark.parametrize("operation", ["upsert", "search", "delete", "file"])
def test_other_owner_rejected_before_storage(boundary, operation):
    b = boundary
    b.auth.user_id = "intruder"
    with pytest.raises(HTTPException) as exc:
        if operation == "upsert":
            r.upsert_vectors(
                VectorUpsert(texts=["x"], vectors=[[1, 0]], metadata=[{}]),
                "store",
                b.db,
                b.auth,
            )
        elif operation == "search":
            r.search_vectors(VectorSearch(query_vector=[1, 0]), "store", b.db, b.auth)
        elif operation == "file":
            r.delete_file("store", "file", b.db, b.auth)
        else:
            r.delete_vector_store("store", True, b.db, b.auth)
    assert exc.value.status_code == 404
    assert not b.runtime.mock_calls


@pytest.mark.parametrize(
    "change",
    [
        dict(vectors=[]),
        dict(metadata=[]),
        dict(vectors=[[1, 0], [1]]),
        dict(vectors=[[float("nan"), 0]]),
        dict(texts=["x" * 65537]),
    ],
)
def test_upsert_validation(change):
    payload = dict(texts=["x"], vectors=[[1, 0]], metadata=[{}])
    payload.update(change)
    with pytest.raises(ValidationError):
        VectorUpsert(**payload)


def test_upsert_vectors_and_collection_dimension(boundary):
    b = boundary
    result = r.upsert_vectors(
        VectorUpsert(texts=["x"], vectors=[[1, 0]], metadata=[{"file_path": "p"}]),
        "store",
        b.db,
        b.auth,
    )
    assert result.points_inserted == 1
    b.runtime.add_to_store.assert_called_once_with(
        "collection", ["x"], [[1, 0]], [{"file_path": "p"}], vector_name=None
    )
    with pytest.raises(HTTPException) as exc:
        r.upsert_vectors(
            VectorUpsert(texts=["x"], vectors=[[1]], metadata=[{}]),
            "store",
            b.db,
            b.auth,
        )
    assert exc.value.status_code == 422


def test_file_metadata_failure_compensates_only_new_points(boundary):
    b = boundary
    b.service.create_vector_store_file.side_effect = r.VectorStoreDBError("db failure")
    request = VectorUpsert(
        texts=["x"],
        vectors=[[1, 0]],
        metadata=[{"file_path": "forged"}],
        file={"file_id": "f", "file_name": "p", "file_path": "p"},
    )
    with pytest.raises(HTTPException):
        r.upsert_vectors(request, "store", b.db, b.auth)
    b.runtime.delete_points.assert_called_once_with("collection", ["point"])
    assert b.runtime.add_to_store.call_args.args[3][0]["file_path"] == "p"


def test_search_vector_contract_and_hits(boundary):
    b = boundary
    b.runtime.query_store.return_value = [
        {"id": "p", "score": 0.9, "text": "x", "metadata": {"file_path": "p"}}
    ]
    hits = r.search_vectors(
        VectorSearch(query_vector=[1, 0], top_k=3, filters={"must": []}, offset=2),
        "store",
        b.db,
        b.auth,
    )
    assert hits[0]["vector_id"] == "p"
    assert hits[0]["meta_data"] == hits[0]["metadata"]
    assert b.runtime.query_store.call_args.kwargs["query_vector"] == [1, 0]
    assert b.runtime.query_store.call_args.kwargs["offset"] == 2
    with pytest.raises(ValidationError):
        VectorSearch(query_text="embed on server")
    with pytest.raises(ValidationError):
        VectorSearch(query_vector=[1, 0], vector_store_host="localhost")


def test_soft_and_hard_delete(boundary):
    b = boundary
    r.delete_vector_store("store", False, b.db, b.auth)
    b.runtime.delete_store.assert_not_called()
    b.service.mark_vector_store_deleted.assert_called_once_with("store")
    b.store.status = V.StatusEnum.deleted
    r.delete_vector_store("store", True, b.db, b.auth)
    b.runtime.delete_store.assert_called_once_with("collection")
    b.service.permanently_delete_vector_store.assert_called_once_with("store")


def test_file_delete_storage_before_metadata_and_missing_idempotent(boundary):
    b = boundary
    events = []
    b.runtime.delete_file_from_store.side_effect = lambda *a: events.append("vectors")
    b.service.delete_vector_store_file_by_path.side_effect = lambda *a: events.append(
        "metadata"
    )
    r.delete_file("store", "p", b.db, b.auth)
    assert events == ["vectors", "metadata"]
    b.service.delete_vector_store_file_by_path.side_effect = (
        r.VectorStoreFileNotFoundError()
    )
    r.delete_file("store", "p", b.db, b.auth)


def test_deleted_store_cannot_upsert(boundary):
    b = boundary
    b.store.status = V.StatusEnum.deleted
    with pytest.raises(HTTPException):
        r.upsert_vectors(
            VectorUpsert(texts=["x"], vectors=[[1, 0]], metadata=[{}]),
            "store",
            b.db,
            b.auth,
        )
    b.runtime.add_to_store.assert_not_called()


def test_adapter_uses_persisted_collections_and_named_vectors(tmp_path):
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    manager = VectorStoreManager.__new__(VectorStoreManager)
    manager.client = client
    manager.create_store("collection", vector_size=2)
    manager.add_to_store("collection", ["text"], [[1.0, 0.0]], [{"file_path": "p"}])
    assert manager.list_store_files("collection") == ["p"]
    assert manager.query_store("collection", [1.0, 0.0])[0]["text"] == "text"
    client.close()
    restarted = VectorStoreManager.__new__(VectorStoreManager)
    restarted.client = QdrantClient(path=str(tmp_path / "qdrant"))
    assert restarted.get_store_info("collection")["vectors_count"] == 1
    restarted.delete_file_from_store("collection", "p")
    assert restarted.get_store_info("collection")["vectors_count"] == 0
    assert restarted.delete_store("collection")["status"] == "deleted"
    assert restarted.delete_store("collection")["status"] == "absent"
    from qdrant_client.models import Distance, VectorParams

    restarted.create_store(
        "named", vectors_config={"text": VectorParams(size=2, distance=Distance.COSINE)}
    )
    restarted.add_to_store("named", ["named"], [[1.0, 0.0]], [{}])
    assert (
        restarted.query_store("named", [1.0, 0.0], vector_field="text")[0]["text"]
        == "named"
    )
    restarted.client.close()


def test_core_source_has_no_sdk_storage_import():
    for path in Path("src").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            if isinstance(node, ast.ImportFrom):
                assert node.module != "projectdavid.clients.vector_store_manager", str(
                    path
                )
            elif isinstance(node, ast.Import):
                assert all(
                    n.name != "projectdavid.clients.vector_store_manager"
                    for n in node.names
                ), str(path)


def test_http_routes_auth_and_typed_validation(boundary):
    b = boundary
    app = FastAPI()
    app.include_router(r.router, prefix="/v1")
    app.dependency_overrides[r.get_db] = lambda: b.db

    def deny():
        raise HTTPException(status_code=401, detail="Missing API key")

    app.dependency_overrides[r.get_api_key] = deny
    client = TestClient(app)
    assert (
        client.post(
            "/v1/vector-stores/store/search", json={"query_vector": [1, 0]}
        ).status_code
        == 401
    )
    app.dependency_overrides[r.get_api_key] = lambda: b.auth
    assert (
        client.post(
            "/v1/vector-stores/store/search", json={"query_text": "x"}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/v1/vector-stores/store/vectors",
            json={"texts": ["x"], "vectors": [[]], "metadata": [{}]},
        ).status_code
        == 422
    )


def test_file_db_decrements_count_once_and_not_on_repeat():
    from src.api.entities_api.services.vectors_service import (
        VectorStoreDBService,
        VectorStoreFileNotFoundError,
    )

    db = Mock()
    store = SimpleNamespace(file_count=2, updated_at=0)
    db.get.return_value = store
    db.query.return_value.filter.return_value.first.side_effect = [
        SimpleNamespace(id="file"),
        None,
    ]
    service = VectorStoreDBService(db)
    service.delete_vector_store_file_by_path("store", "p")
    assert store.file_count == 1
    with pytest.raises(VectorStoreFileNotFoundError):
        service.delete_vector_store_file_by_path("store", "p")
    assert store.file_count == 1
    db.commit.assert_called_once()


def test_db_validation_failure_is_before_commit(monkeypatch):
    from src.api.entities_api.services.vectors_service import (
        VectorStoreDBError,
        VectorStoreDBService,
    )

    db = Mock()
    monkeypatch.setattr(
        V.VectorStoreRead,
        "model_validate",
        Mock(side_effect=ValueError("schema failed")),
    )
    with pytest.raises(VectorStoreDBError):
        VectorStoreDBService(db).create_vector_store(
            "store", "test", "owner", 2, "COSINE"
        )
    db.flush.assert_called_once()
    db.commit.assert_not_called()
    db.rollback.assert_called_once()


def test_physical_delete_failure_preserves_metadata(boundary):
    b = boundary
    b.runtime.delete_store.side_effect = RuntimeError("storage down")
    with pytest.raises(HTTPException):
        r.delete_vector_store("store", True, b.db, b.auth)
    b.service.permanently_delete_vector_store.assert_not_called()


def test_file_success_uses_single_coordinated_operation(boundary):
    b = boundary
    b.service.create_vector_store_file.return_value = V.VectorStoreFileRead(
        id="file",
        vector_store_id="store",
        file_name="p",
        file_path="p",
        status="completed",
    )
    data = VectorUpsert(
        texts=["x"],
        vectors=[[1, 0]],
        metadata=[{}],
        file={"file_id": "file", "file_name": "p", "file_path": "p"},
    )
    result = r.upsert_vectors(data, "store", b.db, b.auth)
    assert result.file.id == "file"
    assert result.points_inserted == 1
    b.runtime.delete_points.assert_not_called()


def test_adapter_duplicate_create_preserves_existing_data():
    from src.api.entities_api.services.vector_runtime.base_vector_store import (
        StoreExistsError,
    )

    manager = VectorStoreManager.__new__(VectorStoreManager)
    manager.client = QdrantClient(":memory:")
    manager.create_store("store", vector_size=2)
    manager.add_to_store("store", ["keep"], [[1.0, 0.0]], [{}])
    with pytest.raises(StoreExistsError):
        manager.create_store("store", vector_size=2)
    assert manager.get_store_info("store")["vectors_count"] == 1
    manager.client.close()


def test_core_storage_does_not_require_embedding_packages():
    import tomllib

    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert not any(
        d.startswith(("sentence-transformers", "torch"))
        for d in data["project"]["dependencies"]
    )
    for path in Path("src/api/entities_api/services/vector_runtime").glob("*.py"):
        assert "sentence_transformers" not in path.read_text(encoding="utf-8")
