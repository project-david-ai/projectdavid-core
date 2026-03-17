# src/api/training/routers/datasets_router.py

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from projectdavid_common import UtilsInterface, ValidationInterface
from sqlalchemy.orm import Session

from src.api.training.db.database import get_db
from src.api.training.dependencies import get_current_user_id
from src.api.training.models.models import StatusEnum
from src.api.training.services.dataset_service import (create_dataset,
                                                       delete_dataset,
                                                       get_dataset,
                                                       list_datasets,
                                                       prepare_dataset)

logging_utility = UtilsInterface.LoggingUtility()
validator = ValidationInterface()

router = APIRouter()

# ---------------------------------------------------------------------------
# Samba client dependency
# ---------------------------------------------------------------------------
# Injected so the router stays testable — swap out for a mock in tests.


def get_samba_client():
    from src.api.entities_api.utils.samba_client import SambaClient

    return SambaClient()


# ---------------------------------------------------------------------------
# POST /v1/datasets
# ---------------------------------------------------------------------------


@router.post(
    "/",
    response_model=ValidationInterface.DatasetRead,
    status_code=201,
    summary="Register a new dataset",
)
async def create_dataset_endpoint(
    name: str = Form(...),
    format: str = Form(...),
    description: Optional[str] = Form(default=None),
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
    samba_client=Depends(get_samba_client),
):
    logging_utility.info("POST /datasets — user=%s name=%s format=%s", user_id, name, format)
    dataset = create_dataset(
        db=db,
        samba_client=samba_client,
        user_id=user_id,
        name=name,
        fmt=format,
        file=file,
        description=description,
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
            raise HTTPException(
                status_code=422,
                detail=f"Invalid status value '{status}'.",
            )

    datasets = list_datasets(
        db=db,
        user_id=user_id,
        status=status_filter,
        limit=limit,
        offset=offset,
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
    summary="Trigger format validation and train/eval split",
)
async def prepare_dataset_endpoint(
    dataset_id: str,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
    samba_client=Depends(get_samba_client),
):
    result = prepare_dataset(
        db=db,
        samba_client=samba_client,
        dataset_id=dataset_id,
        user_id=user_id,
    )
    return result


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
