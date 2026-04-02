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
    """Connect to Redis with response decoding enabled for cluster strings."""
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    return redis.from_url(redis_url, decode_responses=True)


class TrainingService:
    """
    Service layer for training job lifecycle management.

    Node selection is intentionally deferred — training jobs are created
    without a node binding and picked up from Redis by the worker. Actual
    GPU resource assignment occurs at activation time via ModelRegistryService
    and the InferenceReconciler.
    """

    def __init__(self, db: Session) -> None:
        self.db = db
        self.r = get_redis_client()

    # ------------------------------------------------------------------
    # Job creation
    # ------------------------------------------------------------------

    def create_training_job(
        self,
        user_id: str,
        dataset_id: str,
        base_model: str,
        framework: str,
        config: Optional[dict] = None,
    ) -> TrainingJob:
        # 1. Verify dataset exists and is ready
        dataset = get_dataset(self.db, dataset_id, user_id)

        current_status = (
            dataset.status.value
            if hasattr(dataset.status, "value")
            else str(dataset.status)
        )

        if current_status != StatusEnum.active.value:
            raise HTTPException(
                status_code=400,
                detail=f"Dataset {dataset_id} is not ready. Current status: {current_status}",
            )

        # 2. Create job record — node binding deferred to activation
        job_id = IdentifierService.generate_prefixed_id("job")
        now = int(time.time())

        job = TrainingJob(
            id=job_id,
            user_id=user_id,
            dataset_id=dataset_id,
            base_model=base_model,
            framework=framework,
            config=config or {},
            status=StatusEnum.queued,
            node_id=None,
            created_at=now,
            updated_at=now,
        )

        try:
            self.db.add(job)
            self.db.commit()
            self.db.refresh(job)
            logging_utility.info(
                "Training job %s registered. Node binding deferred to activation.",
                job_id,
            )
        except Exception as e:
            self.db.rollback()
            logging_utility.error("DB commit failed for training job %s: %s", job_id, e)
            raise HTTPException(
                status_code=500, detail="Failed to save training job to database."
            )

        # 3. Enqueue to Redis
        self._enqueue(job)

        return job

    # ------------------------------------------------------------------
    # Job retrieval
    # ------------------------------------------------------------------

    def get_training_job(self, job_id: str, user_id: str) -> TrainingJob:
        job = (
            self.db.query(TrainingJob)
            .filter(
                TrainingJob.id == job_id,
                TrainingJob.user_id == user_id,
                TrainingJob.deleted_at.is_(None),
            )
            .first()
        )
        if not job:
            raise HTTPException(
                status_code=404, detail=f"Training job '{job_id}' not found."
            )
        return job

    def list_training_jobs(
        self,
        user_id: str,
        status: Optional[StatusEnum] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[TrainingJob]:
        query = self.db.query(TrainingJob).filter(
            TrainingJob.user_id == user_id,
            TrainingJob.deleted_at.is_(None),
        )
        if status:
            query = query.filter(TrainingJob.status == status)
        return (
            query.order_by(TrainingJob.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

    # ------------------------------------------------------------------
    # Job lifecycle
    # ------------------------------------------------------------------

    def cancel_training_job(self, job_id: str, user_id: str) -> dict:
        job = self.get_training_job(job_id, user_id)

        if job.status in [
            StatusEnum.completed,
            StatusEnum.failed,
            StatusEnum.cancelled,
        ]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel job in status: {job.status.value}",
            )

        job.status = StatusEnum.cancelling
        job.updated_at = int(time.time())

        try:
            self.db.commit()
            logging_utility.info(
                "Training job %s marked for cancellation by user %s", job_id, user_id
            )
        except Exception as e:
            self.db.rollback()
            raise HTTPException(
                status_code=500, detail="Database error during cancellation."
            )

        return {"status": "cancelling", "job_id": job_id}

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def peek_training_queue(self, user_id: str, limit: int = 10) -> List[dict]:
        """Peek at the Redis queue and return jobs belonging to this user."""
        try:
            items = self.r.lrange("training_jobs", 0, 100)
        except Exception:
            return []

        user_jobs = []
        for item in items:
            try:
                payload = json.loads(item)
                if payload.get("user_id") == user_id:
                    user_jobs.append(payload)
                    if len(user_jobs) >= limit:
                        break
            except Exception:
                continue

        return user_jobs

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _enqueue(self, job: TrainingJob) -> None:
        try:
            payload = json.dumps(
                {
                    "job_id": job.id,
                    "user_id": job.user_id,
                    "target_node": job.node_id,
                }
            )
            self.r.lpush("training_jobs", payload)
            logging_utility.info("Training job %s enqueued to Redis.", job.id)
        except Exception as e:
            logging_utility.error("Failed to enqueue job %s to Redis: %s", job.id, e)
            job.status = StatusEnum.failed
            job.config = {**(job.config or {}), "queue_error": str(e)}
            self.db.commit()
            raise HTTPException(
                status_code=500, detail="Failed to enqueue training job to worker."
            )
