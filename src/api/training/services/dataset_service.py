import asyncio
import os
import time
from typing import List, Optional

from fastapi import HTTPException
from projectdavid_common import UtilsInterface
from projectdavid_common.schemas.enums import StatusEnum
from projectdavid_common.utilities.identifier_service import IdentifierService
from projectdavid_orm.projectdavid_orm.models import \
    FileStorage  # Accessing the shared ORM
from sqlalchemy.orm import Session

from src.api.training.models.models import Dataset
from src.api.training.services.file_service import \
    SambaClient  # Using your provided wrapper

logging_utility = UtilsInterface.LoggingUtility()

SUPPORTED_FORMATS = {"chatml", "alpaca", "sharegpt", "jsonl"}

# ---------------------------------------------------------------------------
# Infrastructure Helpers
# ---------------------------------------------------------------------------


def _get_samba_client() -> SambaClient:
    """
    Initialize SambaClient using environment variables.
    """
    return SambaClient(
        server=os.getenv("SMBCLIENT_SERVER", "samba"),
        share=os.getenv("SMBCLIENT_SHARE", "cosmic_share"),
        username=os.getenv("SMBCLIENT_USERNAME", "samba_user"),
        password=os.getenv("SMBCLIENT_PASSWORD"),
    )


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

    lines = [line.strip() for line in file_bytes.decode("utf-8").splitlines() if line.strip()]

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
    # Removed api_base_url and api_key as they are no longer needed for internal fetches
) -> dict:
    """
    Trigger background preparation. Ownership is verified via get_dataset.
    Preparation is now done via direct Samba access.
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

    # Kick off background task without needing user tokens
    asyncio.create_task(_run_preparation(dataset.id, dataset.file_id, dataset.format))

    return {"status": "processing", "dataset_id": dataset_id}


async def _run_preparation(
    dataset_id: str,
    file_id: str,
    fmt: str,
) -> None:
    """
    Background task: Query shared DB for storage path, fetch from Samba,
    validate, and compute splits.
    """
    from src.api.training.db.database import SessionLocal

    db_bg = SessionLocal()
    try:
        bg_dataset = db_bg.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not bg_dataset:
            logging_utility.error("Background prep: dataset %s not found.", dataset_id)
            return

        # 1. Lookup physical storage in the shared Core API tables
        storage_record = db_bg.query(FileStorage).filter(FileStorage.file_id == file_id).first()
        if not storage_record:
            raise ValueError(f"No storage record found for file_id {file_id}")

        logging_utility.info(
            "🚀 Internal Prep: Fetching %s from Samba", storage_record.storage_path
        )

        # 2. Download bytes directly from Samba
        # SMBConnection is synchronous, so we use run_in_executor to avoid blocking
        smb = _get_samba_client()
        loop = asyncio.get_event_loop()
        file_bytes = await loop.run_in_executor(
            None, smb.download_file_to_bytes, storage_record.storage_path
        )

        # 3. Validate and Split
        train_samples, eval_samples = _split_and_validate(file_bytes, fmt=fmt, eval_ratio=0.1)

        # 4. Success Update
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
            bg_dataset.config = {**(bg_dataset.config or {}), "preparation_error": str(e)}
            bg_dataset.updated_at = int(time.time())

    finally:
        try:
            db_bg.commit()
        except Exception as e:
            logging_utility.error("Failed to commit preparation result for %s: %s", dataset_id, e)
            db_bg.rollback()
        finally:
            db_bg.close()
