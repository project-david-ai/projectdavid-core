# src/api/training/services/dataset_service.py
import asyncio
import json
import os
import time
from typing import List, Optional

from fastapi import HTTPException
from projectdavid_common import UtilsInterface
from projectdavid_common.schemas.enums import StatusEnum
from projectdavid_common.utilities.identifier_service import IdentifierService
from projectdavid_orm.projectdavid_orm.models import File as FileModel
from projectdavid_orm.projectdavid_orm.models import FileStorage
from sqlalchemy.orm import Session

from src.api.training.models.models import Dataset, TrainingJob
from src.api.training.services.file_service import SambaClient

logging_utility = UtilsInterface.LoggingUtility()

SUPPORTED_FORMATS = {"chatml", "alpaca", "sharegpt", "jsonl"}
BLOCKING_JOB_STATUSES = (
    StatusEnum.queued,
    StatusEnum.in_progress,
    StatusEnum.cancelling,
)


# ---------------------------------------------------------------------------
# Module-level privates (stateless helpers)
# ---------------------------------------------------------------------------


def _get_samba_client() -> SambaClient:
    """Instantiate a Samba client for background prep (download from share)."""
    return SambaClient(
        server=os.getenv("SMBCLIENT_SERVER", "samba"),
        share=os.getenv("SMBCLIENT_SHARE", "cosmic_share"),
        username=os.getenv("SMBCLIENT_USERNAME", "samba_user"),
        password=os.getenv("SMBCLIENT_PASSWORD"),
    )


def _split_and_validate(
    file_bytes: bytes,
    fmt: str,
    eval_ratio: float = 0.1,
) -> tuple[int, int]:
    """
    Validate file is well-formed for the given format.
    Returns (train_samples, eval_samples). Raises ValueError on malformed input.
    """
    lines = [
        line.strip() for line in file_bytes.decode("utf-8").splitlines() if line.strip()
    ]

    if not lines:
        raise ValueError("Dataset file is empty.")

    parsed = []
    for i, line in enumerate(lines, 1):
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"Line {i} is not valid JSON: {e}")

    if fmt == "chatml":
        for i, record in enumerate(parsed, 1):
            if "messages" not in record:
                raise ValueError(f"chatml record {i} missing 'messages' field.")
    elif fmt == "alpaca":
        for i, record in enumerate(parsed, 1):
            if "instruction" not in record:
                raise ValueError(f"alpaca record {i} missing 'instruction' field.")
    elif fmt == "sharegpt":
        for i, record in enumerate(parsed, 1):
            if "conversations" not in record:
                raise ValueError(f"sharegpt record {i} missing 'conversations' field.")

    total = len(parsed)
    eval_samples = max(1, int(total * eval_ratio))
    train_samples = total - eval_samples
    return train_samples, eval_samples


# ---------------------------------------------------------------------------
# DatasetService
# ---------------------------------------------------------------------------


class DatasetService:
    """
    Service layer for dataset lifecycle: create, list, retrieve, prepare, delete.

    Delete supports two modes:
      - soft (default): sets deleted_at and flips status to StatusEnum.deleted.
        DB rows preserved. Listed queries exclude it.
      - hard (hard=True): soft-delete + cascade remove of File + FileStorage
        DB rows. Physical bytes on Samba are reaped later by the existing
        purge daemons (purge_expired_files, purge_soft_deleted_files).

    Both modes share race guards:
      - Refuse if dataset is mid-preparation (status=processing)
      - Refuse if dataset is referenced by any non-terminal training job
        (queued, in_progress, cancelling)
    """

    def __init__(self, db: Session) -> None:
        self.db = db

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create(
        self,
        user_id: str,
        name: str,
        fmt: str,
        file_id: str,
        description: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> Dataset:
        if fmt not in SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported format '{fmt}'. Must be one of: {sorted(SUPPORTED_FORMATS)}",
            )

        dataset_id = IdentifierService.generate_prefixed_id("ds")
        now = int(time.time())

        dataset = Dataset(
            id=dataset_id,
            user_id=user_id,
            name=name,
            description=description,
            format=fmt,
            file_id=file_id,
            storage_path=None,
            status=StatusEnum.pending,
            created_at=now,
            updated_at=now,
        )

        try:
            self.db.add(dataset)
            self.db.commit()
            self.db.refresh(dataset)
            logging_utility.info(
                "Dataset %s registered — file_id=%s user=%s",
                dataset_id,
                file_id,
                user_id,
            )
        except Exception as e:
            self.db.rollback()
            logging_utility.error("DB commit failed for dataset %s: %s", dataset_id, e)
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

        return dataset

    def get(self, dataset_id: str, user_id: str) -> Dataset:
        dataset = (
            self.db.query(Dataset)
            .filter(
                Dataset.id == dataset_id,
                Dataset.user_id == user_id,
                Dataset.deleted_at.is_(None),
            )
            .first()
        )
        if not dataset:
            raise HTTPException(
                status_code=404, detail=f"Dataset '{dataset_id}' not found."
            )
        return dataset

    def list(
        self,
        user_id: str,
        status: Optional[StatusEnum] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dataset]:
        query = self.db.query(Dataset).filter(
            Dataset.user_id == user_id,
            Dataset.deleted_at.is_(None),
        )
        if status:
            query = query.filter(Dataset.status == status)
        return (
            query.order_by(Dataset.created_at.desc()).offset(offset).limit(limit).all()
        )

    # ------------------------------------------------------------------
    # Delete — soft + hard
    # ------------------------------------------------------------------

    def delete(
        self,
        dataset_id: str,
        user_id: str,
        hard: bool = False,
    ) -> dict:
        """
        Delete a dataset. Default is soft-delete; hard=True cascades to File
        and FileStorage DB rows. Physical bytes are reaped by purge daemons.

        Race guards (applied to both modes):
          - 409 if dataset is mid-preparation
          - 409 if referenced by any non-terminal training job
        """
        dataset = self.get(dataset_id, user_id)
        self._assert_deletable(dataset)

        if not hard:
            return self._soft_delete(dataset, user_id)
        return self._hard_delete(dataset, user_id)

    def _assert_deletable(self, dataset: Dataset) -> None:
        """Apply race guards. Raises HTTPException(409) on violation."""
        if dataset.status == StatusEnum.processing:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Dataset '{dataset.id}' is currently being prepared. "
                    f"Wait for preparation to complete before deleting."
                ),
            )

        active_jobs = (
            self.db.query(TrainingJob.id, TrainingJob.status)
            .filter(
                TrainingJob.dataset_id == dataset.id,
                TrainingJob.status.in_(BLOCKING_JOB_STATUSES),
                TrainingJob.deleted_at.is_(None),
            )
            .all()
        )
        if active_jobs:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": f"Dataset '{dataset.id}' is in use by active training jobs.",
                    "blocking_jobs": [
                        {"job_id": job_id, "status": str(status)}
                        for job_id, status in active_jobs
                    ],
                    "hint": "Cancel or wait for these jobs before deleting the dataset.",
                },
            )

    def _soft_delete(self, dataset: Dataset, user_id: str) -> dict:
        now = int(time.time())
        dataset.deleted_at = now
        dataset.status = StatusEnum.deleted
        dataset.updated_at = now
        try:
            self.db.commit()
            logging_utility.info(
                "Dataset %s soft-deleted by user %s", dataset.id, user_id
            )
        except Exception as e:
            self.db.rollback()
            raise HTTPException(status_code=500, detail=f"Database error: {e}")
        return {"deleted": True, "dataset_id": dataset.id}

    def _hard_delete(self, dataset: Dataset, user_id: str) -> dict:
        """
        Cascade hard-delete: Dataset row, FileStorage rows, File row.
        Physical bytes on Samba are reaped later by purge_soft_deleted_files
        once the File row's deleted_at passes the retention window.

        For immediate hard-delete semantics, we mark the File as deleted
        (sets deleted_at on the File row), which the purge daemon sweeps.
        """
        dataset_id = dataset.id
        file_id = dataset.file_id
        now = int(time.time())

        try:
            # Mark the backing File row as soft-deleted so the purge daemon
            # reaps the physical bytes on its next pass.
            file_row = self.db.query(FileModel).filter(FileModel.id == file_id).first()
            if file_row and file_row.deleted_at is None:
                file_row.deleted_at = now

            # Hard-remove the Dataset row from the table — not a soft flip.
            self.db.delete(dataset)
            self.db.commit()
            logging_utility.info(
                "Dataset %s hard-deleted by user %s (file_id=%s, File marked for purge)",
                dataset_id,
                user_id,
                file_id,
            )
        except Exception as e:
            self.db.rollback()
            logging_utility.error(
                "Hard-delete failed for dataset %s: %s", dataset_id, e
            )
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

        return {"deleted": True, "dataset_id": dataset_id, "hard": True}

    # ------------------------------------------------------------------
    # Prepare
    # ------------------------------------------------------------------

    def prepare(self, dataset_id: str, user_id: str) -> dict:
        """Trigger background preparation. Ownership is verified via self.get."""
        dataset = self.get(dataset_id, user_id)

        if dataset.status == StatusEnum.active:
            return {"status": "active", "dataset_id": dataset_id}
        if dataset.status == StatusEnum.processing:
            return {"status": "processing", "dataset_id": dataset_id}

        dataset.status = StatusEnum.processing
        dataset.updated_at = int(time.time())

        try:
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            raise HTTPException(status_code=500, detail=f"Database error: {e}")

        asyncio.create_task(
            _run_preparation(dataset.id, dataset.file_id, dataset.format)
        )

        return {"status": "processing", "dataset_id": dataset_id}


# ---------------------------------------------------------------------------
# Background prep coroutine
# ---------------------------------------------------------------------------
# Stays module-level: manages its own DB session lifecycle, not tied to a
# request-scoped DatasetService instance.


async def _run_preparation(
    dataset_id: str,
    file_id: str,
    fmt: str,
) -> None:
    """
    Background task: query shared DB for storage path, fetch from Samba,
    validate, compute splits.
    """
    from src.api.training.db.database import SessionLocal

    db_bg = SessionLocal()
    bg_dataset = None
    try:
        bg_dataset = db_bg.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not bg_dataset:
            logging_utility.error("Background prep: dataset %s not found.", dataset_id)
            return

        storage_record = (
            db_bg.query(FileStorage).filter(FileStorage.file_id == file_id).first()
        )
        if not storage_record:
            raise ValueError(f"No storage record found for file_id {file_id}")

        logging_utility.info(
            "🚀 Internal Prep: Fetching %s from Samba", storage_record.storage_path
        )

        smb = _get_samba_client()
        loop = asyncio.get_event_loop()
        file_bytes = await loop.run_in_executor(
            None, smb.download_file_to_bytes, storage_record.storage_path
        )

        train_samples, eval_samples = _split_and_validate(
            file_bytes, fmt=fmt, eval_ratio=0.1
        )

        bg_dataset.train_samples = train_samples
        bg_dataset.eval_samples = eval_samples
        bg_dataset.status = StatusEnum.active
        bg_dataset.updated_at = int(time.time())

        logging_utility.info(
            "✅ Dataset %s prepared: %d train / %d eval samples",
            dataset_id,
            train_samples,
            eval_samples,
        )

    except Exception as e:
        logging_utility.error("❌ Dataset %s preparation failed: %s", dataset_id, e)
        if bg_dataset:
            bg_dataset.status = StatusEnum.failed
            bg_dataset.config = {
                **(bg_dataset.config or {}),
                "preparation_error": str(e),
            }
            bg_dataset.updated_at = int(time.time())

    finally:
        try:
            db_bg.commit()
        except Exception as e:
            logging_utility.error(
                "Failed to commit preparation result for %s: %s", dataset_id, e
            )
            db_bg.rollback()
        finally:
            db_bg.close()
