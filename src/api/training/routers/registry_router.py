"""
registry_router.py

REST API endpoints for the Base Model Registry.

Prefix:  /v1/registry
Tags:    registry

Endpoints:
  POST   /v1/registry/base-models                        — Register a base model        [admin only]
  GET    /v1/registry/base-models                        — List all base models
  GET    /v1/registry/base-models/{model_ref:path}       — Retrieve by ID or HF path
  DELETE /v1/registry/base-models/{model_id}             — Deregister by ID             [admin only]
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.api.entities_api.utils.check_admin_status import _is_admin
from src.api.training.db.database import get_db
from src.api.training.dependencies import get_current_user_id
from src.api.training.schemas.registry_schemas import (
    BaseModelDeleted, BaseModelList, BaseModelRead, BaseModelRegisterRequest)
from src.api.training.services.registry_service import RegistryService

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_admin(current_user_id: str, db: Session) -> None:
    """
    Raise 403 if the current user does not have admin privileges.
    Centralised so all admin-gated endpoints stay consistent.
    """
    if not _is_admin(current_user_id, db):
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required for this operation.",
        )


# ---------------------------------------------------------------------------
# Registration                                                   [admin only]
# ---------------------------------------------------------------------------


@router.post(
    "/base-models",
    response_model=BaseModelRead,
    summary="Register a base model",
    description=(
        "Register a HuggingFace base model in the catalog. "
        "Generates a clean `bm_...` prefixed ID and stores the HF path in `endpoint`. "
        "Idempotent — re-registering the same HF path returns the existing record. "
        "**Admin only.**"
    ),
    status_code=201,
)
def register_base_model(
    payload: BaseModelRegisterRequest,
    db: Session = Depends(get_db),
    current_user_id: str = Depends(get_current_user_id),
) -> BaseModelRead:
    _require_admin(current_user_id, db)
    service = RegistryService(db)
    model = service.register_base_model(
        hf_model_id=payload.hf_model_id,
        name=payload.name,
        family=payload.family,
        parameter_count=payload.parameter_count,
        is_multimodal=payload.is_multimodal,
    )
    return BaseModelRead.model_validate(model)


# ---------------------------------------------------------------------------
# Listing                                                        [any user]
# ---------------------------------------------------------------------------


@router.get(
    "/base-models",
    response_model=BaseModelList,
    summary="List registered base models",
    description="Return a paginated list of all base models in the catalog.",
)
def list_base_models(
    limit: int = Query(default=50, ge=1, le=200, description="Page size."),
    offset: int = Query(default=0, ge=0, description="Pagination offset."),
    db: Session = Depends(get_db),
    current_user_id: str = Depends(get_current_user_id),
) -> BaseModelList:
    service = RegistryService(db)
    items = service.list_base_models(limit=limit, offset=offset)
    return BaseModelList(
        items=[BaseModelRead.model_validate(m) for m in items],
        total=len(items),
        limit=limit,
        offset=offset,
    )


# ---------------------------------------------------------------------------
# Retrieval                                                      [any user]
# ---------------------------------------------------------------------------


@router.get(
    "/base-models/{model_ref:path}",
    response_model=BaseModelRead,
    summary="Retrieve a base model",
    description=(
        "Fetch a base model by either its `bm_...` prefixed ID "
        "or its HuggingFace path (e.g. `unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit`). "
        "The `:path` converter handles slashes in HF model IDs."
    ),
)
def get_base_model(
    model_ref: str,
    db: Session = Depends(get_db),
    current_user_id: str = Depends(get_current_user_id),
) -> BaseModelRead:
    service = RegistryService(db)
    model = service.resolve(model_ref)
    return BaseModelRead.model_validate(model)


# ---------------------------------------------------------------------------
# Deregistration                                                 [admin only]
# ---------------------------------------------------------------------------


@router.delete(
    "/base-models/{model_id}",
    response_model=BaseModelDeleted,
    summary="Deregister a base model",
    description=(
        "Remove a base model from the catalog by its `bm_...` prefixed ID. "
        "Hard delete — use with caution if active deployments reference this model. "
        "**Admin only.**"
    ),
)
def deregister_base_model(
    model_id: str,
    db: Session = Depends(get_db),
    current_user_id: str = Depends(get_current_user_id),
) -> BaseModelDeleted:
    _require_admin(current_user_id, db)
    service = RegistryService(db)
    result = service.deregister_base_model(model_id)
    return BaseModelDeleted(**result)
