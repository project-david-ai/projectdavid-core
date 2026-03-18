# src/api/training/routers/datasets_router.py

import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from projectdavid_common import UtilsInterface, ValidationInterface
from projectdavid_common.schemas.enums import StatusEnum
from sqlalchemy.orm import Session

from src.api.training.db.database import get_db
from src.api.training.dependencies import get_current_user_id
from src.api.training.services.dataset_service import (create_dataset,
                                                       delete_dataset,
                                                       get_dataset,
                                                       list_datasets,
                                                       prepare_dataset)

logging_utility = UtilsInterface.LoggingUtility()

router = APIRouter()

API_BASE_URL = os.getenv("ASSISTANTS_BASE_URL", "http://api:9000")
WORKER_API_KEY = os.getenv("WORKER_API_KEY", "")


# ---------------------------------------------------------------------------
# POST /v1/datasets
# ---------------------------------------------------------------------------


@router.post(
    "/",
    response_model=ValidationInterface.DatasetRead,
    status_code=201,
    summary="Register a dataset by file_id",
)
def create_dataset_endpoint(
    payload: ValidationInterface.DatasetCreate,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    logging_utility.info(
        "POST /datasets — user=%s name=%s format=%s file_id=%s",
        user_id,
        payload.name,
        payload.format,
        payload.file_id,
    )
    dataset = create_dataset(
        db=db,
        user_id=user_id,
        name=payload.name,
        fmt=payload.format,
        file_id=payload.file_id,
        description=payload.description,
        filename=getattr(payload, "filename", None),
    )
    return ValidationInterface.DatasetRead.model_validate(dataset)


# ---------------------------------------------------------------------------
# GET /v1/datasets
# ---------------------------------------------------------------------------


@router.get(
    "/",
    response_model=ValidationInterface.DatasetList,
    summary="List datasets for the current user",
)
def list_datasets_endpoint(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    status_filter = None
    if status:
        try:
            status_filter = StatusEnum(status)
        except ValueError:
            raise HTTPException(status_code=422, detail=f"Invalid status value '{status}'.")

    datasets = list_datasets(
        db=db, user_id=user_id, status=status_filter, limit=limit, offset=offset
    )
    return ValidationInterface.DatasetList(
        data=[ValidationInterface.DatasetRead.model_validate(d) for d in datasets],
        total=len(datasets),
    )


# ---------------------------------------------------------------------------
# GET /v1/datasets/{dataset_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{dataset_id}",
    response_model=ValidationInterface.DatasetRead,
    summary="Retrieve a dataset record",
)
def get_dataset_endpoint(
    dataset_id: str,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    dataset = get_dataset(db=db, dataset_id=dataset_id, user_id=user_id)
    return ValidationInterface.DatasetRead.model_validate(dataset)


# ---------------------------------------------------------------------------
# POST /v1/datasets/{dataset_id}/prepare
# ---------------------------------------------------------------------------


@router.post(
    "/{dataset_id}/prepare",
    summary="Validate format and compute train/eval split",
)
async def prepare_dataset_endpoint(
    dataset_id: str,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    return prepare_dataset(
        db=db,
        dataset_id=dataset_id,
        user_id=user_id,
        api_base_url=API_BASE_URL,
        api_key=WORKER_API_KEY,
    )


# ---------------------------------------------------------------------------
# DELETE /v1/datasets/{dataset_id}
# ---------------------------------------------------------------------------


@router.delete(
    "/{dataset_id}",
    response_model=ValidationInterface.DatasetDeleted,
    summary="Soft delete a dataset",
)
def delete_dataset_endpoint(
    dataset_id: str,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    result = delete_dataset(db=db, dataset_id=dataset_id, user_id=user_id)
    return ValidationInterface.DatasetDeleted(**result)
