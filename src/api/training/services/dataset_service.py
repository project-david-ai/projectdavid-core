# src/api/training/services/dataset_service.py

import asyncio
import time
from typing import List, Optional

import httpx
from fastapi import HTTPException
from projectdavid_common import UtilsInterface
from projectdavid_common.schemas.enums import StatusEnum
from projectdavid_common.utilities.identifier_service import IdentifierService
from sqlalchemy.orm import Session

from src.api.training.models.models import Dataset

logging_utility = UtilsInterface.LoggingUtility()

SUPPORTED_FORMATS = {"chatml", "alpaca", "sharegpt", "jsonl"}


# ---------------------------------------------------------------------------
# Core API client helper
# ---------------------------------------------------------------------------


async def _fetch_file_as_bytes(file_id: str, api_base_url: str, api_key: str) -> bytes:
    """
    Retrieve file content from the core API as bytes.
    Uses GET /v1/files/{file_id}/base64 — no direct Samba access needed.
    """
    import base64

    url = f"{api_base_url}/v1/files/{file_id}/base64"
    async with httpx.AsyncClient() as client:
        response = await client.get(
            url,
            headers={"X-API-Key": api_key},
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()
        b64 = data.get("base64")
        if not b64:
            raise ValueError(f"No base64 content returned for file_id={file_id}")
        return base64.b64decode(b64)


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
    user_id: str,
    name: str,
    fmt: str,
    file_id: str,
    description: Optional[str] = None,
    filename: Optional[str] = None,
) -> Dataset:
    """
    Register a dataset record. The file has already been uploaded to the
    core API — we just store the reference here.
    """
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
        storage_path=None,  # resolved by worker at training time
        status=StatusEnum.pending,
        created_at=now,
        updated_at=now,
    )

    try:
        db.add(dataset)
        db.commit()
        db.refresh(dataset)
        logging_utility.info(
            "Dataset %s registered — file_id=%s user=%s", dataset_id, file_id, user_id
        )
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
    dataset_id: str,
    user_id: str,
    api_base_url: str,
    api_key: str,
) -> dict:
    """
    Trigger background preparation — fetch file from core API, validate,
    compute splits, update DB record.
    """
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

    asyncio.create_task(
        _run_preparation(dataset.id, dataset.file_id, dataset.format, api_base_url, api_key)
    )

    return {"status": "processing", "dataset_id": dataset_id}


async def _run_preparation(
    dataset_id: str,
    file_id: str,
    fmt: str,
    api_base_url: str,
    api_key: str,
) -> None:
    """
    Background task: fetch file from core API, validate format, compute splits.
    Opens its own DB session — safe to outlive the originating request.
    """
    from src.api.training.db.database import SessionLocal

    db_bg = SessionLocal()
    try:
        bg_dataset = db_bg.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not bg_dataset:
            logging_utility.error("Background prep: dataset %s not found.", dataset_id)
            return

        logging_utility.info("Fetching file %s from core API for dataset %s", file_id, dataset_id)

        file_bytes = await _fetch_file_as_bytes(file_id, api_base_url, api_key)

        train_samples, eval_samples = _split_and_validate(file_bytes, fmt=fmt, eval_ratio=0.1)

        bg_dataset.train_samples = train_samples
        bg_dataset.eval_samples = eval_samples
        bg_dataset.status = StatusEnum.active
        bg_dataset.updated_at = int(time.time())

        logging_utility.info(
            "Dataset %s prepared — %d train / %d eval samples",
            dataset_id,
            train_samples,
            eval_samples,
        )

    except ValueError as e:
        logging_utility.warning("Dataset %s validation failed: %s", dataset_id, e)
        bg_dataset.status = StatusEnum.failed
        bg_dataset.config = {**(bg_dataset.config or {}), "preparation_error": str(e)}
        bg_dataset.updated_at = int(time.time())

    except Exception as e:
        logging_utility.error("Dataset %s preparation failed: %s", dataset_id, e)
        bg_dataset.status = StatusEnum.failed
        bg_dataset.config = {**(bg_dataset.config or {}), "preparation_error": str(e)}
        bg_dataset.updated_at = int(time.time())

    finally:
        try:
            db_bg.commit()
        except Exception as e:
            logging_utility.error(
                "Failed to commit preparation result for dataset %s: %s", dataset_id, e
            )
            db_bg.rollback()
        finally:
            db_bg.close()
