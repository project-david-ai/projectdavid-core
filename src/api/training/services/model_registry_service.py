import time
from typing import List, Optional

from fastapi import HTTPException
from projectdavid_common.schemas.enums import StatusEnum
from projectdavid_common.utilities.identifier_service import IdentifierService
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.api.training.models.models import (BaseModel, ComputeNode,
                                            FineTunedModel, GPUAllocation,
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


# ---------------------------------------------------------------------------
# Cluster Resource Management & Scheduling
# ---------------------------------------------------------------------------


def find_available_node(db: Session, required_vram: float = 4.0) -> str:
    """
    Finds a node by calculating: Total VRAM - SUM(All Active Reservations).
    """
    nodes = db.query(ComputeNode).filter(ComputeNode.status == StatusEnum.active).all()

    for node in nodes:
        # Calculate current load from the Ledger
        reserved = (
            db.query(func.sum(GPUAllocation.vram_reserved_gb))
            .filter(GPUAllocation.node_id == node.id)
            .scalar()
            or 0.0
        )

        if (node.total_vram_gb - reserved) >= required_vram:
            return node.id

    # 🎯 REQUIREMENT 2: Graceful feedback
    raise HTTPException(
        status_code=507, detail=f"Insufficient VRAM in cluster. Required: {required_vram}GB."
    )


# ---------------------------------------------------------------------------
# Deployment Logic (The Mesh Implementation)
# ---------------------------------------------------------------------------


def deactivate_all_models(db: Session, user_id: str) -> dict:
    """
    CLEAN SLATE: Physically removes deployments and allocations to
    satisfy UniqueConstraints and free up hardware ports.
    """
    # 1. Reset Metadata flags for the user — E712 fix: use is_active directly
    db.query(FineTunedModel).filter(
        FineTunedModel.user_id == user_id, FineTunedModel.is_active
    ).update({"is_active": False, "node_id": None}, synchronize_session=False)

    # 2. HARD DELETE Deployments: This clears the uq_node_port_deployment constraint
    # so we can insert new models on the same port immediately.
    # E711 fix: use is_not(None) instead of != None
    db.query(InferenceDeployment).filter(InferenceDeployment.node_id.is_not(None)).delete(
        synchronize_session=False
    )

    # 3. Release VRAM allocations physically
    db.query(GPUAllocation).delete(synchronize_session=False)

    db.commit()
    return {"status": "success", "message": "Cluster resources released."}


def activate_model(
    db: Session, model_id: str, user_id: str, target_node_id: Optional[str] = None
) -> dict:
    """
    DEPLOYS A FINE-TUNED MODEL (Base + LoRA).
    """
    model = get_fine_tuned_model(db, model_id, user_id)

    # 1. Clear existing slots first
    deactivate_all_models(db, user_id)

    # 2. Schedule
    node_id = target_node_id or find_available_node(db, required_vram=5.0)

    # 3. Create Ticket
    deployment_id = IdentifierService.generate_prefixed_id("dep")
    deployment = InferenceDeployment(
        id=deployment_id,
        node_id=node_id,
        base_model_id=model.base_model,
        fine_tuned_model_id=model.id,
        port=8001,
        status=StatusEnum.pending,
        last_seen=int(time.time()),
    )

    # 4. Lock VRAM
    allocation = GPUAllocation(node_id=node_id, model_id=model.id, vram_reserved_gb=5.0)

    # 5. Persist
    model.is_active = True
    model.node_id = node_id
    db.add(deployment)
    db.add(allocation)
    db.commit()

    return {
        "status": "deploying",
        "model_id": model.id,
        "node": node_id,
        "next_step": "Worker is provisioning LoRA weights.",
    }


def activate_base_model(
    db: Session, base_model_id: str, user_id: str, target_node_id: Optional[str] = None
) -> dict:
    """
    DEPLOYS A STANDARD MODEL (Backbone only).
    """
    # 1. Verify model exists in our seeded catalog
    base = db.query(BaseModel).filter(BaseModel.id == base_model_id).first()
    if not base:
        raise HTTPException(status_code=404, detail=f"Base model {base_model_id} not found.")

    # 2. Mutex: Clear existing deployments to free Node Port 8001
    deactivate_all_models(db, user_id)

    # 3. Scheduler: Find hardware
    node_id = target_node_id or find_available_node(db, required_vram=4.0)

    # 4. Create the Deployment Ticket (fine_tuned_model_id=None signals Standard Mode)
    deployment_id = IdentifierService.generate_prefixed_id("dep")
    deployment = InferenceDeployment(
        id=deployment_id,
        node_id=node_id,
        base_model_id=base.id,
        fine_tuned_model_id=None,
        port=8001,
        status=StatusEnum.pending,
        last_seen=int(time.time()),
    )

    # 5. Lock VRAM (Standard backbone)
    allocation = GPUAllocation(
        node_id=node_id, model_id=None, vram_reserved_gb=4.0  # Indicates base model lock
    )

    db.add(deployment)
    db.add(allocation)
    db.commit()

    return {
        "status": "deploying_standard",
        "model_id": base.id,
        "node": node_id,
        "next_step": f"Standard backbone {base.id} is being provisioned.",
    }
