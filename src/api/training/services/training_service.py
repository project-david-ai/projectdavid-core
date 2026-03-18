# src/api/training/services/training_service.py
import json
import os
import time
from typing import List, Optional

import redis
from fastapi import HTTPException
from projectdavid_common import UtilsInterface
from projectdavid_common.schemas.enums import StatusEnum
from projectdavid_common.utilities.identifier_service import IdentifierService
from sqlalchemy.orm import Session

from src.api.training.models.models import TrainingJob
from src.api.training.services.dataset_service import get_dataset

logging_utility = UtilsInterface.LoggingUtility()


def get_redis_client() -> redis.Redis:
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    return redis.from_url(redis_url)


def create_training_job(
    db: Session,
    user_id: str,
    dataset_id: str,
    base_model: str,
    framework: str,
    config: Optional[dict] = None,
) -> TrainingJob:
    # 1. Verify dataset exists and is ready for training
    dataset = get_dataset(db, dataset_id, user_id)
    if dataset.status != StatusEnum.active:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset {dataset_id} is not ready. Current status: {dataset.status.value}",
        )

    # 2. Create TrainingJob record
    job_id = IdentifierService.generate_prefixed_id("job")
    now = int(time.time())

    job = TrainingJob(
        id=job_id,
        user_id=user_id,
        dataset_id=dataset_id,
        base_model=base_model,
        framework=framework,
        config=config or {},
        status=StatusEnum.pending,
        created_at=now,
        updated_at=now,
    )

    try:
        db.add(job)
        db.commit()
        db.refresh(job)
        logging_utility.info("Training job %s registered for user %s", job_id, user_id)
    except Exception as e:
        db.rollback()
        logging_utility.error("DB commit failed for training job %s: %s", job_id, e)
        raise HTTPException(status_code=500, detail="Failed to save training job to database.")

    # 3. Push to Redis Queue for the Worker
    try:
        r = get_redis_client()
        payload = json.dumps({"job_id": job.id, "user_id": user_id})
        r.lpush("training_jobs", payload)
        logging_utility.info("Training job %s enqueued to Redis successfully.", job_id)
    except Exception as e:
        logging_utility.error("Failed to enqueue job %s to Redis: %s", job_id, e)
        # We flag the job as failed if Redis is unreachable
        job.status = StatusEnum.failed
        job.config = {**(job.config or {}), "queue_error": str(e)}
        db.commit()
        raise HTTPException(status_code=500, detail="Failed to enqueue training job to worker.")

    return job


def get_training_job(db: Session, job_id: str, user_id: str) -> TrainingJob:
    job = (
        db.query(TrainingJob)
        .filter(
            TrainingJob.id == job_id,
            TrainingJob.user_id == user_id,
            TrainingJob.deleted_at.is_(None),
        )
        .first()
    )

    if not job:
        raise HTTPException(status_code=404, detail=f"Training Job '{job_id}' not found.")
    return job


def list_training_jobs(
    db: Session,
    user_id: str,
    status: Optional[StatusEnum] = None,
    limit: int = 50,
    offset: int = 0,
) -> List[TrainingJob]:
    query = db.query(TrainingJob).filter(
        TrainingJob.user_id == user_id, TrainingJob.deleted_at.is_(None)
    )
    if status:
        query = query.filter(TrainingJob.status == status)
    return query.order_by(TrainingJob.created_at.desc()).offset(offset).limit(limit).all()


def cancel_training_job(db: Session, job_id: str, user_id: str) -> dict:
    job = get_training_job(db, job_id, user_id)

    if job.status in [StatusEnum.completed, StatusEnum.failed, StatusEnum.cancelled]:
        raise HTTPException(
            status_code=400, detail=f"Cannot cancel job in status: {job.status.value}"
        )

    job.status = StatusEnum.cancelling
    job.updated_at = int(time.time())

    try:
        db.commit()
        logging_utility.info("Training job %s marked for cancellation by user %s", job_id, user_id)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    return {"status": "cancelling", "job_id": job_id}
