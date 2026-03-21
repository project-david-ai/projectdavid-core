import time
from typing import List, Optional

from fastapi import HTTPException
from projectdavid_common.schemas.enums import StatusEnum
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.api.training.models.models import (BaseModel, ComputeNode,
                                            FineTunedModel,
                                            InferenceDeployment)

# ---------------------------------------------------------------------------
# Registry Retrieval
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Cluster Resource Management (The New Logic)
# ---------------------------------------------------------------------------


def find_available_node(db: Session, required_vram: float = 4.0) -> str:
    """
    Finds the active node with the most free VRAM.
    Formula: total_vram - current_vram_usage
    """
    node = (
        db.query(ComputeNode)
        .filter(
            ComputeNode.status == StatusEnum.active,
            (ComputeNode.total_vram_gb - ComputeNode.current_vram_usage_gb) >= required_vram,
        )
        .order_by((ComputeNode.total_vram_gb - ComputeNode.current_vram_usage_gb).desc())
        .first()
    )

    if not node:
        raise HTTPException(
            status_code=503,
            detail="Resource Exhausted: No GPU nodes with sufficient VRAM available.",
        )
    return node.id


def activate_model(db: Session, model_id: str, user_id: str) -> dict:
    """
    Sets the target model as active AND identifies the best node for deployment.
    """
    # 1. Fetch the model
    model = get_fine_tuned_model(db, model_id, user_id)

    # 2. Mutex: Deactivate all other models for this user
    db.query(FineTunedModel).filter(
        FineTunedModel.user_id == user_id, FineTunedModel.id != model_id
    ).update({"is_active": False})

    # 3. Cluster Logic: Bind to the best available physical hardware
    # We assume a 1B model needs ~4GB for stable inference with LoRA
    node_id = find_available_node(db, required_vram=4.0)
    model.node_id = node_id
    model.is_active = True

    db.commit()

    # 4. Return instructions including the targeted Node
    return {
        "activated": model.id,
        "vllm_model_id": model.id,
        "target_node": node_id,
        "next_step": f"Restart vLLM on {node_id}: platform-api --node {node_id} up --services vllm",
    }


def deactivate_all_models(db: Session, user_id: str) -> dict:
    """
    Sets all fine-tuned models for this user to inactive.
    System resets to the base model defined in the CLI environment.
    """
    db.query(FineTunedModel).filter(
        FineTunedModel.user_id == user_id, FineTunedModel.is_active == True
    ).update({"is_active": False, "node_id": None})

    db.commit()
    return {
        "status": "success",
        "message": "All fine-tuned models deactivated. System reset to base model.",
    }


# ---------------------------------------------------------------------------
# Future: Unified Deployment Registry
# ---------------------------------------------------------------------------


def register_deployment(
    db: Session, node_id: str, base_model_id: str, ftm_id: Optional[str] = None
) -> InferenceDeployment:
    """
    Records a live vLLM process in the cluster ledger.
    """
    from projectdavid_common.utilities.identifier_service import \
        IdentifierService

    deployment_id = IdentifierService.generate_prefixed_id("dep")

    deployment = InferenceDeployment(
        id=deployment_id,
        node_id=node_id,
        base_model_id=base_model_id,
        fine_tuned_model_id=ftm_id,
        status=StatusEnum.active,
        last_seen=int(time.time()),
    )
    db.add(deployment)
    db.commit()
    return deployment
