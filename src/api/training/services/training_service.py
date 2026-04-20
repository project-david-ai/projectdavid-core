# src/api/training/services/training_service.py
import json
import os
import time
from pathlib import Path
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

# ─── CANCELLATION SIGNALING ──────────────────────────────────────────────────
# The worker polls this Redis key on every stdout read from the training
# subprocess. Presence of the key = user requested cancellation.
# TTL prevents orphaned keys from accumulating if the worker misses a read
# (crashed, disconnected, etc.) and the job completes on its own.
CANCEL_KEY_PREFIX = "cancel:job:"
CANCEL_KEY_TTL_SECONDS = 3600  # 1 hour — generous; training rarely outlasts this


def _cancel_key(job_id: str) -> str:
    return f"{CANCEL_KEY_PREFIX}{job_id}"


# ─────────────────────────────────────────────────────────────────────────────


def get_redis_client() -> redis.Redis:
    """Connect to Redis with response decoding enabled for cluster strings."""
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    return redis.from_url(redis_url, decode_responses=True)


def _is_model_in_hf_cache(model_id: str) -> bool:
    """
    Check whether a HuggingFace model is present in the local cache.

    Uses HF_HOME (the HuggingFace-native env var) with a fallback to the
    standard default cache location. HF_HOME is the canonical variable
    set by transformers/huggingface_hub itself.
    """
    hf_home = os.getenv("HF_HOME", "/root/.cache/huggingface")
    safe_name = "models--" + model_id.replace("/", "--")
    snapshots_path = Path(hf_home) / "hub" / safe_name / "snapshots"

    if not snapshots_path.exists():
        return False

    for snapshot_dir in snapshots_path.iterdir():
        if snapshot_dir.is_dir() and any(snapshot_dir.iterdir()):
            return True

    return False


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

        # 2. HF cache guard — reject immediately if model is not cached locally.
        #    This prevents the training worker from attempting a download mid-job,
        #    which would fail silently after claiming the GPU and consuming queue time.
        #    Users must pre-cache models via the model registry before fine-tuning.
        if not _is_model_in_hf_cache(base_model):
            logging_utility.warning(
                "Training job rejected — model '%s' not found in HF cache for user %s",
                base_model,
                user_id,
            )
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Model '{base_model}' is not available in the local HuggingFace cache. "
                    f"Register and activate the base model via the model registry before "
                    f"submitting a fine-tuning job."
                ),
            )

        # 3. Create job record — node binding deferred to activation
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

        # 4. Enqueue to Redis
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
        """
        Cancel a queued or in-progress training job.

        Behaviour by current status:
            queued       → flip to CANCELLED immediately (worker skips on claim)
            in_progress  → flip to CANCELLING + signal Redis; worker unwinds
            cancelling   → idempotent, return current status
            cancelled    → idempotent, return current status
            completed    → idempotent, return current status ("already finished")
            failed       → idempotent, return current status ("already failed")
        """
        job = self.get_training_job(job_id, user_id)

        current_status = (
            job.status.value if hasattr(job.status, "value") else str(job.status)
        )

        # ── Idempotent short-circuits for terminal / in-flight-cancel states ──
        if job.status == StatusEnum.cancelled:
            return {
                "job_id": job_id,
                "status": current_status,
                "cancelled_at": job.cancelled_at,
                "message": "Job was already cancelled.",
            }

        if job.status == StatusEnum.cancelling:
            return {
                "job_id": job_id,
                "status": current_status,
                "cancelled_at": job.cancelled_at,
                "message": "Cancellation already in progress.",
            }

        if job.status == StatusEnum.completed:
            return {
                "job_id": job_id,
                "status": current_status,
                "cancelled_at": None,
                "message": "Job already completed — nothing to cancel.",
            }

        if job.status == StatusEnum.failed:
            return {
                "job_id": job_id,
                "status": current_status,
                "cancelled_at": None,
                "message": "Job already failed — nothing to cancel.",
            }

        # ── Actionable cancellations ────────────────────────────────────────
        now = int(time.time())

        if job.status == StatusEnum.queued:
            # Queued jobs: flip straight to CANCELLED.
            # The worker inspects DB status on claim and will skip.
            job.status = StatusEnum.cancelled
            job.cancelled_at = now
            job.updated_at = now

            try:
                self.db.commit()
                logging_utility.info(
                    "Training job %s cancelled while queued (user %s)",
                    job_id,
                    user_id,
                )
            except Exception as e:
                self.db.rollback()
                logging_utility.error(
                    "DB commit failed cancelling queued job %s: %s", job_id, e
                )
                raise HTTPException(
                    status_code=500, detail="Database error during cancellation."
                )

            return {
                "job_id": job_id,
                "status": StatusEnum.cancelled.value,
                "cancelled_at": now,
                "message": "Queued job cancelled before execution started.",
            }

        if job.status == StatusEnum.in_progress:
            # In-progress: flip to CANCELLING, set Redis signal.
            # The worker polls the Redis key between subprocess stdout reads
            # and initiates two-stage subprocess shutdown (SIGTERM → grace → SIGKILL).
            job.status = StatusEnum.cancelling
            job.cancelled_at = now
            job.updated_at = now

            try:
                self.db.commit()
            except Exception as e:
                self.db.rollback()
                logging_utility.error(
                    "DB commit failed flagging job %s for cancellation: %s", job_id, e
                )
                raise HTTPException(
                    status_code=500, detail="Database error during cancellation."
                )

            # Set the cancel signal. If Redis is unreachable, the DB flip is
            # authoritative — worker does a periodic DB check as a backstop,
            # but Redis is the fast path.
            try:
                self.r.set(
                    _cancel_key(job_id),
                    "1",
                    ex=CANCEL_KEY_TTL_SECONDS,
                )
                logging_utility.info(
                    "Training job %s marked for cancellation (user %s), Redis signal set",
                    job_id,
                    user_id,
                )
            except Exception as e:
                logging_utility.warning(
                    "Could not set Redis cancel signal for %s (falling back to DB poll): %s",
                    job_id,
                    e,
                )

            return {
                "job_id": job_id,
                "status": StatusEnum.cancelling.value,
                "cancelled_at": now,
                "message": "Cancellation initiated — worker is unwinding subprocess.",
            }

        # Fallthrough — unexpected state
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job in unexpected status: {current_status}",
        )

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
