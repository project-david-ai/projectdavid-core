from typing import List

from fastapi import APIRouter, Depends, HTTPException
from fastapi import Path as FastApiPath
from fastapi import Query, status
from projectdavid_common import UtilsInterface, ValidationInterface
from projectdavid_common.schemas.vectors_schema import VectorStoreRead
from sqlalchemy.orm import Session

from src.api.entities_api.dependencies import get_api_key, get_db
from src.api.entities_api.models.models import ApiKey as ApiKeyModel
from src.api.entities_api.services.vectors_service import (
    DatabaseConflictError,
    VectorStoreDBError,
    VectorStoreDBService,
)
from src.api.entities_api.utils.check_admin_status import _is_admin

router = APIRouter()
log = UtilsInterface.LoggingUtility()


def _require_store_access(
    store_id: str, db: Session, auth_key: ApiKeyModel, service: VectorStoreDBService
):
    store = service.get_vector_store_by_id(store_id)
    if not store or (store.user_id != auth_key.user_id and not _is_admin(auth_key.user_id, db)):
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
    try:
        return service.create_vector_store(
            shared_id=data.shared_id,
            name=data.name,
            user_id=owner_id,
            vector_size=data.vector_size,
            distance_metric=data.distance_metric,
            config=data.config,
        )
    except DatabaseConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except VectorStoreDBError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {exc}",
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
    if not store or (store.user_id != auth_key.user_id and not _is_admin(auth_key.user_id, db)):
        raise HTTPException(status_code=404, detail="Collection not found.")
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
    if not store or (store.user_id != auth_key.user_id and not _is_admin(auth_key.user_id, db)):
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
    if not store or (store.user_id != auth_key.user_id and not _is_admin(auth_key.user_id, db)):
        raise HTTPException(status_code=404, detail="Vector store not found.")
    try:
        if permanent:
            service.permanently_delete_vector_store(vector_store_id)
        else:
            service.mark_vector_store_deleted(vector_store_id)
    except VectorStoreDBError as exc:
        raise HTTPException(status_code=500, detail=f"Delete failed: {exc}") from exc


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
    _require_store_access(vector_store_id, db, auth_key, service)
    service.delete_vector_store_file_by_path(vector_store_id, file_path)


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
    return service.update_vector_store_file_status(
        file_id, file_status.status, file_status.error_message
    )
