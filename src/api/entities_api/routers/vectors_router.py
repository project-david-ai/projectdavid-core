from functools import lru_cache
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from fastapi import Path as FastApiPath
from fastapi import Query, status
from projectdavid_common import UtilsInterface, ValidationInterface
from projectdavid_common.schemas.vectors_schema import VectorStoreRead
from sqlalchemy.orm import Session

from src.api.entities_api.dependencies import get_api_key, get_db
from src.api.entities_api.models.models import ApiKey as ApiKeyModel
from src.api.entities_api.services.vector_runtime.base_vector_store import (
    StoreExistsError,
)
from src.api.entities_api.services.vector_runtime.schemas import (
    VectorHit,
    VectorSearch,
    VectorUpsert,
    VectorUpsertResult,
)
from src.api.entities_api.services.vector_runtime.vector_store_manager import (
    VectorStoreManager,
)
from src.api.entities_api.services.vectors_service import (
    DatabaseConflictError,
    VectorStoreDBError,
    VectorStoreDBService,
    VectorStoreFileNotFoundError,
)
from src.api.entities_api.utilities.check_admin_status import _is_admin

router = APIRouter()
log = UtilsInterface.LoggingUtility()


@lru_cache(maxsize=1)
def get_vector_runtime():
    return VectorStoreManager()


def _storage_call(operation, *args, **kwargs):
    try:
        return operation(*args, **kwargs)
    except StoreExistsError as exc:
        raise HTTPException(
            status_code=409, detail="Vector store already exists."
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail="Invalid vector configuration or dimensions."
        ) from exc
    except Exception as exc:
        log.error("Vector storage operation failed: %s", exc)
        raise HTTPException(
            status_code=503, detail="Vector storage unavailable."
        ) from exc


def _require_store_access(
    store_id: str, db: Session, auth_key: ApiKeyModel, service: VectorStoreDBService
):
    store = service.get_vector_store_by_id(store_id)
    if not store or (
        store.user_id != auth_key.user_id and not _is_admin(auth_key.user_id, db)
    ):
        raise HTTPException(status_code=404, detail="Vector store not found.")
    if store.status == ValidationInterface.StatusEnum.deleted:
        raise HTTPException(status_code=404, detail="Vector store not found.")
    return store


# ── Vector Store CRUD ─────────────────────────────────────────────────────────


@router.post(
    "/vector-stores",
    response_model=ValidationInterface.VectorStoreRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create Vector Store",
    description=(
        "Creates a new vector-store.\n\n"
        "- **Regular callers** → the store is assigned to *their* user-id.\n"
        "- **Admins** → may pass the optional `owner_id` query-param to create "
        "the store for a different user."
    ),
)
def create_vector_store(
    data: ValidationInterface.VectorStoreCreateWithSharedId,
    owner_id: str | None = Query(
        default=None,
        description="Target user-id (admin-only). If omitted, the store is created for the caller.",
    ),
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    if owner_id is None:
        owner_id = auth_key.user_id
    elif owner_id != auth_key.user_id and not _is_admin(auth_key.user_id, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins may specify owner_id.",
        )

    log.info(
        "Create vector-store %s  owner=%s  requested_by=%s",
        data.shared_id,
        owner_id,
        auth_key.user_id,
    )
    service = VectorStoreDBService(db)
    runtime = get_vector_runtime()
    _storage_call(
        runtime.create_store,
        store_name=data.shared_id,
        vector_size=data.vector_size,
        distance=data.distance_metric,
    )
    try:
        return service.create_vector_store(
            shared_id=data.shared_id,
            name=data.name,
            user_id=owner_id,
            vector_size=data.vector_size,
            distance_metric=data.distance_metric,
            config=data.config,
        )
    except Exception as exc:
        try:
            runtime.delete_store(data.shared_id)
        except Exception:
            log.exception("Failed to compensate vector collection creation")
        if isinstance(exc, DatabaseConflictError):
            raise HTTPException(
                status_code=409, detail="Vector store already exists."
            ) from exc
        raise HTTPException(
            status_code=500, detail="Vector store metadata creation failed."
        ) from exc


@router.get(
    "/vector-stores",
    response_model=List[ValidationInterface.VectorStoreRead],
    summary="List current user's Vector Stores",
)
def list_my_vector_stores(
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    service = VectorStoreDBService(db)
    return service.get_stores_by_user(auth_key.user_id)


@router.get(
    "/vector-stores/admin/by-user",
    response_model=List[VectorStoreRead],
    summary="(Admin) List vector-stores for a given user",
)
def list_vector_stores_by_user(
    owner_id: str = Query(..., description="Target user-id"),
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    if not _is_admin(auth_key.user_id, db):
        raise HTTPException(status_code=403, detail="Admin privilege required.")
    service = VectorStoreDBService(db)
    return service.get_stores_by_user(owner_id)


@router.get(
    "/vector-stores/lookup/collection",
    response_model=ValidationInterface.VectorStoreRead,
    summary="Get Vector Store by Collection Name",
    description="Retrieves vector-store metadata using its unique collection name.",
)
def get_vector_store_by_collection(
    name: str = Query(..., description="Collection name to look up"),
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    service = VectorStoreDBService(db)
    store = service.get_vector_store_by_collection_name(name)
    if not store or (
        store.user_id != auth_key.user_id and not _is_admin(auth_key.user_id, db)
    ):
        raise HTTPException(status_code=404, detail="Collection not found.")
    if store.status == ValidationInterface.StatusEnum.deleted:
        raise HTTPException(status_code=404, detail="Vector store not found.")
    return store


@router.get(
    "/vector-stores/{vector_store_id}",
    response_model=ValidationInterface.VectorStoreRead,
)
def get_vector_store(
    vector_store_id: str = FastApiPath(...),
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    service = VectorStoreDBService(db)
    store = service.get_vector_store_by_id(vector_store_id)
    if not store or (
        store.user_id != auth_key.user_id and not _is_admin(auth_key.user_id, db)
    ):
        raise HTTPException(status_code=404, detail="Vector store not found.")
    if store.status == ValidationInterface.StatusEnum.deleted:
        raise HTTPException(status_code=404, detail="Vector store not found.")
    return store


@router.delete(
    "/vector-stores/{vector_store_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Vector Store",
    description=(
        "Deletes a vector store.\n\n"
        "- **Soft-delete** (default, `permanent=false`) → stamps `deleted_at` and sets "
        "`status=deleted`. The DB record and Qdrant collection are preserved and can be "
        "restored. The store becomes invisible to all normal read/write paths immediately.\n\n"
        "- **Hard-delete** (`permanent=true`) → permanently destroys the DB record and the "
        "backing Qdrant collection. This action is irreversible."
    ),
)
def delete_vector_store(
    vector_store_id: str = FastApiPath(...),
    permanent: bool = Query(
        False,
        description="Set true to permanently destroy the DB record and Qdrant collection.",
    ),
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    log.info(
        "User '%s' – delete store %s permanent=%s",
        auth_key.user_id,
        vector_store_id,
        permanent,
    )
    service = VectorStoreDBService(db)
    store = service.get_vector_store_by_id(vector_store_id)
    if not store or (
        store.user_id != auth_key.user_id and not _is_admin(auth_key.user_id, db)
    ):
        raise HTTPException(status_code=404, detail="Vector store not found.")
    try:
        if permanent:
            _storage_call(get_vector_runtime().delete_store, store.collection_name)
            service.permanently_delete_vector_store(vector_store_id)
        else:
            service.mark_vector_store_deleted(vector_store_id)
    except VectorStoreDBError as exc:
        raise HTTPException(
            status_code=500, detail="Vector store metadata deletion failed."
        ) from exc


# ── Vector Store Files ────────────────────────────────────────────────────────


@router.post(
    "/vector-stores/{vector_store_id}/files",
    response_model=ValidationInterface.VectorStoreFileRead,
    status_code=status.HTTP_201_CREATED,
)
def add_file(
    file_data: ValidationInterface.VectorStoreFileCreate,
    vector_store_id: str = FastApiPath(...),
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    service = VectorStoreDBService(db)
    _require_store_access(vector_store_id, db, auth_key, service)
    return service.create_vector_store_file(
        vector_store_id=vector_store_id,
        file_id=file_data.file_id,
        file_name=file_data.file_name,
        file_path=file_data.file_path,
        status=file_data.status or ValidationInterface.StatusEnum.completed,
        meta_data=file_data.meta_data,
    )


@router.get(
    "/vector-stores/{vector_store_id}/files",
    response_model=List[ValidationInterface.VectorStoreFileRead],
)
def list_files(
    vector_store_id: str = FastApiPath(...),
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    service = VectorStoreDBService(db)
    _require_store_access(vector_store_id, db, auth_key, service)
    return service.list_vector_store_files(vector_store_id)


@router.delete(
    "/vector-stores/{vector_store_id}/files",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_file(
    vector_store_id: str = FastApiPath(...),
    file_path: str = Query(...),
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    service = VectorStoreDBService(db)
    store = _require_store_access(vector_store_id, db, auth_key, service)
    _storage_call(
        get_vector_runtime().delete_file_from_store, store.collection_name, file_path
    )
    try:
        service.delete_vector_store_file_by_path(vector_store_id, file_path)
    except VectorStoreFileNotFoundError:
        return
    except VectorStoreDBError as exc:
        raise HTTPException(
            status_code=500, detail="File metadata deletion failed."
        ) from exc


@router.patch(
    "/vector-stores/{vector_store_id}/files/{file_id}",
    response_model=ValidationInterface.VectorStoreFileRead,
)
def update_file_status(
    file_status: ValidationInterface.VectorStoreFileUpdateStatus,
    vector_store_id: str = FastApiPath(...),
    file_id: str = FastApiPath(...),
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    service = VectorStoreDBService(db)
    _require_store_access(vector_store_id, db, auth_key, service)
    if not any(
        f.id == file_id for f in service.list_vector_store_files(vector_store_id)
    ):
        raise HTTPException(status_code=404, detail="Vector store file not found.")
    return service.update_vector_store_file_status(
        file_id, file_status.status, file_status.error_message
    )


@router.post(
    "/vector-stores/{vector_store_id}/vectors", response_model=VectorUpsertResult
)
def upsert_vectors(
    data: VectorUpsert,
    vector_store_id: str,
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    service = VectorStoreDBService(db)
    store = _require_store_access(vector_store_id, db, auth_key, service)
    if any(len(v) != store.vector_size for v in data.vectors):
        raise HTTPException(
            status_code=422, detail="Vector dimensions do not match store."
        )
    metadata = data.metadata
    if data.file:
        # File identity is server-enforced so vector deletion matches the DB record.
        metadata = [
            {
                **md,
                "file_id": data.file.file_id,
                "file_path": data.file.file_path,
                "file_name": data.file.file_name,
            }
            for md in metadata
        ]
    runtime = get_vector_runtime()
    result = _storage_call(
        runtime.add_to_store,
        store.collection_name,
        data.texts,
        data.vectors,
        metadata,
        vector_name=data.vector_name,
    )
    record = None
    if data.file:
        try:
            record = service.create_vector_store_file(
                vector_store_id=vector_store_id,
                **data.file.model_dump(exclude={"status"}),
                status=data.file.status or ValidationInterface.StatusEnum.completed,
            )
        except Exception as exc:
            try:
                runtime.delete_points(store.collection_name, result["point_ids"])
            except Exception:
                log.exception("Failed to compensate file vector insertion")
            raise HTTPException(
                status_code=500, detail="File metadata creation failed."
            ) from exc
    return VectorUpsertResult(
        status="success", points_inserted=result["points_inserted"], file=record
    )


@router.post("/vector-stores/{vector_store_id}/search", response_model=list[VectorHit])
def search_vectors(
    data: VectorSearch,
    vector_store_id: str,
    db: Session = Depends(get_db),
    auth_key: ApiKeyModel = Depends(get_api_key),
):
    service = VectorStoreDBService(db)
    store = _require_store_access(vector_store_id, db, auth_key, service)
    if len(data.query_vector) != store.vector_size:
        raise HTTPException(
            status_code=422, detail="Query dimensions do not match store."
        )
    hits = _storage_call(
        get_vector_runtime().query_store, store.collection_name, **data.model_dump()
    )
    return [
        {
            **hit,
            "vector_id": hit["id"],
            "meta_data": hit["metadata"],
            "store_id": vector_store_id,
        }
        for hit in hits
    ]
