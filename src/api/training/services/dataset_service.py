# src/api/training/services/dataset_service.py

import asyncio
import time
from typing import List, Optional

from fastapi import HTTPException, UploadFile
from projectdavid_common import UtilsInterface
from projectdavid_common.utilities.identifier_service import IdentifierService
from sqlalchemy.orm import Session

from src.api.training.models.models import Dataset, StatusEnum

logging_utility = UtilsInterface.LoggingUtility()

SUPPORTED_FORMATS = {"chatml", "alpaca", "sharegpt", "jsonl"}


# ---------------------------------------------------------------------------
# Samba helpers
# ---------------------------------------------------------------------------


def _samba_upload(samba_client, file_bytes: bytes, path: str) -> None:
    samba_client.upload(file_bytes, path)


def _samba_download(samba_client, path: str) -> bytes:
    return samba_client.download(path)


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------


def _split_and_validate(
    file_bytes: bytes,
    fmt: str,
    eval_ratio: float = 0.1,
) -> tuple[int, int]:
    """
    Validate file is well-formed for the given format.
    Returns (train_samples, eval_samples).
    Raises ValueError on malformed input.
    """
    import json

    lines = [l.strip() for l in file_bytes.decode("utf-8").splitlines() if l.strip()]

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
# Service functions
# ---------------------------------------------------------------------------


def create_dataset(
    db: Session,
    samba_client,
    user_id: str,
    name: str,
    fmt: str,
    file: UploadFile,
    description: Optional[str] = None,
) -> Dataset:
    if fmt not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported format '{fmt}'. Must be one of: {sorted(SUPPORTED_FORMATS)}",
        )

    dataset_id = IdentifierService.generate_prefixed_id("ds")
    storage_path = f"datasets/{user_id}/{dataset_id}/{file.filename}"

    file_bytes = file.file.read()
    if not file_bytes:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    try:
        _samba_upload(samba_client, file_bytes, storage_path)
        logging_utility.info("Dataset %s uploaded to Samba at %s", dataset_id, storage_path)
    except Exception as e:
        logging_utility.error("Samba upload failed for dataset %s: %s", dataset_id, e)
        raise HTTPException(status_code=500, detail=f"File storage failed: {e}")

    now = int(time.time())
    dataset = Dataset(
        id=dataset_id,
        user_id=user_id,
        name=name,
        description=description,
        format=fmt,
        storage_path=storage_path,
        status=StatusEnum.pending,
        created_at=now,
        updated_at=now,
    )

    try:
        db.add(dataset)
        db.commit()
        db.refresh(dataset)
        logging_utility.info("Dataset %s registered in DB for user %s", dataset_id, user_id)
    except Exception as e:
        db.rollback()
        logging_utility.error("DB commit failed for dataset %s: %s", dataset_id, e)
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    return dataset


def get_dataset(db: Session, dataset_id: str, user_id: str) -> Dataset:
    dataset = (
        db.query(Dataset)
        .filter(
            Dataset.id == dataset_id,
            Dataset.user_id == user_id,
            Dataset.deleted_at.is_(None),
        )
        .first()
    )
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    return dataset


def list_datasets(
    db: Session,
    user_id: str,
    status: Optional[StatusEnum] = None,
    limit: int = 50,
    offset: int = 0,
) -> List[Dataset]:
    query = db.query(Dataset).filter(
        Dataset.user_id == user_id,
        Dataset.deleted_at.is_(None),
    )
    if status:
        query = query.filter(Dataset.status == status)

    return query.order_by(Dataset.created_at.desc()).offset(offset).limit(limit).all()


def delete_dataset(db: Session, dataset_id: str, user_id: str) -> dict:
    dataset = get_dataset(db, dataset_id, user_id)

    dataset.deleted_at = int(time.time())
    dataset.status = StatusEnum.deleted
    dataset.updated_at = int(time.time())

    try:
        db.commit()
        logging_utility.info("Dataset %s soft-deleted by user %s", dataset_id, user_id)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    return {"deleted": True, "dataset_id": dataset_id}


def prepare_dataset(
    db: Session,
    samba_client,
    dataset_id: str,
    user_id: str,
) -> dict:
    dataset = get_dataset(db, dataset_id, user_id)

    if dataset.status == StatusEnum.active:
        return {"status": "active", "dataset_id": dataset_id}

    if dataset.status == StatusEnum.processing:
        return {"status": "processing", "dataset_id": dataset_id}

    dataset.status = StatusEnum.processing
    dataset.updated_at = int(time.time())

    try:
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    asyncio.create_task(_run_preparation(db, samba_client, dataset))

    return {"status": "processing", "dataset_id": dataset_id}


async def _run_preparation(
    db: Session,
    samba_client,
    dataset: Dataset,
) -> None:
    """
    Background task: validate, split, and mark dataset active.
    Opens a fresh DB session to avoid DetachedInstanceError — the
    session from the originating request will be closed by the time
    this task runs.
    """
    from src.api.training.db.database import SessionLocal

    db_bg = SessionLocal()
    try:
        # Re-fetch inside the background session
        bg_dataset = db_bg.query(Dataset).filter(Dataset.id == dataset.id).first()
        if not bg_dataset:
            logging_utility.error("Background prep: dataset %s not found.", dataset.id)
            return

        logging_utility.info(
            "Starting preparation for dataset %s (format=%s)",
            bg_dataset.id,
            bg_dataset.format,
        )

        file_bytes = _samba_download(samba_client, bg_dataset.storage_path)

        train_samples, eval_samples = _split_and_validate(
            file_bytes, fmt=bg_dataset.format, eval_ratio=0.1
        )

        prepared_path = bg_dataset.storage_path.replace("/datasets/", "/datasets_prepared/")
        _samba_upload(samba_client, file_bytes, prepared_path)

        bg_dataset.train_samples = train_samples
        bg_dataset.eval_samples = eval_samples
        bg_dataset.storage_path = prepared_path
        bg_dataset.status = StatusEnum.active
        bg_dataset.updated_at = int(time.time())

        logging_utility.info(
            "Dataset %s prepared — %d train / %d eval samples",
            bg_dataset.id,
            train_samples,
            eval_samples,
        )

    except ValueError as e:
        logging_utility.warning("Dataset %s validation failed: %s", dataset.id, e)
        bg_dataset.status = StatusEnum.failed
        bg_dataset.config = {**(bg_dataset.config or {}), "preparation_error": str(e)}
        bg_dataset.updated_at = int(time.time())

    except Exception as e:
        logging_utility.error("Dataset %s preparation failed unexpectedly: %s", dataset.id, e)
        bg_dataset.status = StatusEnum.failed
        bg_dataset.config = {**(bg_dataset.config or {}), "preparation_error": str(e)}
        bg_dataset.updated_at = int(time.time())

    finally:
        try:
            db_bg.commit()
        except Exception as e:
            logging_utility.error(
                "Failed to commit preparation result for dataset %s: %s",
                dataset.id,
                e,
            )
            db_bg.rollback()
        finally:
            db_bg.close()
