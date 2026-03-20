import time
from typing import List, Optional

from fastapi import HTTPException
from projectdavid_common.schemas.enums import StatusEnum
from sqlalchemy.orm import Session

from src.api.training.models.models import FineTunedModel


def list_fine_tuned_models(
    db: Session, user_id: str, limit: int = 50, offset: int = 0
) -> List[FineTunedModel]:
    return (
        db.query(FineTunedModel)
        .filter(FineTunedModel.user_id == user_id, FineTunedModel.deleted_at.is_(None))
        .order_by(FineTunedModel.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )


def get_fine_tuned_model(db: Session, model_id: str, user_id: str) -> FineTunedModel:
    model = (
        db.query(FineTunedModel)
        .filter(
            FineTunedModel.id == model_id,
            FineTunedModel.user_id == user_id,
            FineTunedModel.deleted_at.is_(None),
        )
        .first()
    )
    if not model:
        raise HTTPException(status_code=404, detail="Model not found.")
    return model


def soft_delete_model(db: Session, model_id: str, user_id: str) -> bool:
    model = get_fine_tuned_model(db, model_id, user_id)
    model.deleted_at = int(time.time())
    model.status = StatusEnum.deleted
    db.commit()
    return True


def activate_model(db: Session, model_id: str, user_id: str) -> dict:
    """
    Sets the target model as active in the database.
    Note: Real-world container restart is handled via the Platform CLI,
    but the API provides the 'Source of Truth' for which model to load.
    """
    # 1. Fetch the model
    model = get_fine_tuned_model(db, model_id, user_id)

    # 2. Deactivate all other models for this user
    db.query(FineTunedModel).filter(
        FineTunedModel.user_id == user_id, FineTunedModel.id != model_id
    ).update({"is_active": False})

    # 3. Activate target
    model.is_active = True
    db.commit()

    # 4. Return instructions for the CLI/Orchestrator
    # This matches the schema: ActivateModelResponse
    return {
        "activated": model.id,
        "vllm_model_id": model.id,  # The ID vLLM will use to reference the model
        "next_step": "Restart the inference service to load these weights: platform-api docker-manager --mode up --services vllm",
    }
