# src/api/training/routers/training_jobs_router.py

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from projectdavid_common import ValidationInterface
from projectdavid_common.schemas.enums import StatusEnum
from sqlalchemy.orm import Session

from src.api.training.db.database import get_db
from src.api.training.dependencies import get_current_user_id
from src.api.training.services.training_service import TrainingService

router = APIRouter()


@router.post(
    "/",
    response_model=ValidationInterface.TrainingJobRead,
    status_code=201,
    summary="Create and queue a new training job",
)
def create_training_job_endpoint(
    payload: ValidationInterface.TrainingJobCreate,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    service = TrainingService(db)
    job = service.create_training_job(
        user_id=user_id,
        dataset_id=payload.dataset_id,
        base_model=payload.base_model,
        framework=payload.framework,
        config=payload.config,
    )
    return ValidationInterface.TrainingJobRead.model_validate(job)


@router.get(
    "/",
    response_model=ValidationInterface.TrainingJobList,
    summary="List training jobs for the current user",
)
def list_training_jobs_endpoint(
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
                status_code=422, detail=f"Invalid status value '{status}'."
            )

    service = TrainingService(db)
    jobs = service.list_training_jobs(
        user_id=user_id, status=status_filter, limit=limit, offset=offset
    )
    return ValidationInterface.TrainingJobList(
        data=[ValidationInterface.TrainingJobRead.model_validate(j) for j in jobs],
        total=len(jobs),
    )


@router.get(
    "/{job_id}",
    response_model=ValidationInterface.TrainingJobRead,
    summary="Retrieve a training job record",
)
def get_training_job_endpoint(
    job_id: str,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    service = TrainingService(db)
    job = service.get_training_job(job_id=job_id, user_id=user_id)
    return ValidationInterface.TrainingJobRead.model_validate(job)


@router.post(
    "/{job_id}/cancel",
    summary="Cancel a running or pending training job",
)
def cancel_training_job_endpoint(
    job_id: str,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    service = TrainingService(db)
    return service.cancel_training_job(job_id=job_id, user_id=user_id)


@router.get(
    "/queue/peek",
    response_model=ValidationInterface.TrainingQueueList,
    summary="Diagnostic: See pending jobs in the Redis queue for this user",
)
def peek_queue_endpoint(
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    service = TrainingService(db)
    items = service.peek_training_queue(user_id=user_id)
    return ValidationInterface.TrainingQueueList(
        total_in_queue=len(items),
        data=[ValidationInterface.TrainingQueueItem(**item) for item in items],
    )
