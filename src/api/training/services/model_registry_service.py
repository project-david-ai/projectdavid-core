import time
from typing import List, Optional

from fastapi import HTTPException
from projectdavid_common.schemas.enums import StatusEnum
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


def soft_delete_model(db: Session, model_id: str, user_id: str) -> bool:
    model = get_fine_tuned_model(db, model_id, user_id)
    model.deleted_at = int(time.time())
    model.status = StatusEnum.deleted
    db.commit()
    return True


# ---------------------------------------------------------------------------
# Cluster Resource Management & Scheduling
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
    v2.0 Cluster Activation Logic:
    1. Identifies the best physical node via the Scheduler.
    2. Mutex: Sets existing deployments for this user to 'offline'.
    3. Creates a 'Deployment Ticket' (Status: pending) for the Node Watcher.
    4. Reserves VRAM in the Cluster Ledger.
    """
    # 1. Fetch Model and Base Model metadata
    model = get_fine_tuned_model(db, model_id, user_id)

    # 2. Mutex: Deactivate all other fine-tuned models and deployments for this user
    db.query(FineTunedModel).filter(
        FineTunedModel.user_id == user_id, FineTunedModel.id != model_id
    ).update({"is_active": False})

    db.query(InferenceDeployment).filter(
        InferenceDeployment.fine_tuned_model_id.in_(
            db.query(FineTunedModel.id).filter(FineTunedModel.user_id == user_id)
        )
    ).update({"status": StatusEnum.offline})

    # 3. Scheduling: Find a home (Assume 5GB required for 1B model + LoRA overhead)
    node_id = find_available_node(db, required_vram=5.0)

    # 4. Create the Deployment Ticket (Signal to the physical Node)
    deployment_id = f"dep_{model.id[4:]}"
    deployment = InferenceDeployment(
        id=deployment_id,
        node_id=node_id,
        base_model_id=model.base_model,
        fine_tuned_model_id=model.id,
        port=8001,  # Standard port; could be made dynamic in v3.0
        status=StatusEnum.pending,  # Worker 'start_deployment_watcher' picks this up
        last_seen=int(time.time()),
    )

    # 5. Lock VRAM in Ledger (The Accounting Layer)
    allocation = GPUAllocation(node_id=node_id, model_id=model.id, vram_reserved_gb=5.0)

    # 6. Persist changes
    model.is_active = True
    model.node_id = node_id

    db.merge(deployment)  # Merge handles potential existing record updates
    db.add(allocation)

    db.commit()

    return {
        "activated": model.id,
        "deployment_id": deployment_id,
        "target_node": node_id,
        "url": f"http://{node_id}:8001",
        "next_step": "DEPLOY_SIGNAL_SENT",
    }


def deactivate_all_models(db: Session, user_id: str) -> dict:
    """
    Resets the system to the standard base model.
    Clears all active flags and releases cluster deployments.
    """
    # Clear model flags
    db.query(FineTunedModel).filter(
        FineTunedModel.user_id == user_id, FineTunedModel.is_active == True
    ).update({"is_active": False, "node_id": None})

    # Clear deployment ledger
    db.query(InferenceDeployment).filter(
        InferenceDeployment.fine_tuned_model_id.in_(
            db.query(FineTunedModel.id).filter(FineTunedModel.user_id == user_id)
        )
    ).update({"status": StatusEnum.offline})

    # Release VRAM allocations
    db.query(GPUAllocation).filter(
        GPUAllocation.model_id.in_(
            db.query(FineTunedModel.id).filter(FineTunedModel.user_id == user_id)
        )
    ).delete(synchronize_session=False)

    db.commit()
    return {
        "status": "success",
        "message": "All fine-tuned models deactivated. System reset to base model.",
    }


# ---------------------------------------------------------------------------
# Utility Helpers
# ---------------------------------------------------------------------------


def register_deployment(
    db: Session, node_id: str, base_model_id: str, ftm_id: Optional[str] = None
) -> InferenceDeployment:
    """
    Manually record a live vLLM process in the cluster ledger.
    Used by internal agents to sync physical state to the DB.
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


def activate_base_model(db: Session, base_model_id: str, user_id: str) -> dict:
    """
    v2.0 Cluster Logic for Standard Models:
    1. Finds the model in the catalog (base_models table).
    2. Schedules it to the healthiest node.
    3. Creates a 'Deployment Ticket' with NO adapter path.
    """
    # 1. Verify model exists in our seeded catalog
    base = db.query(BaseModel).filter(BaseModel.id == base_model_id).first()
    if not base:
        raise HTTPException(
            status_code=404, detail=f"Base model {base_model_id} not found in catalog."
        )

    # 2. Mutex: Set other active deployments for this user to offline
    db.query(InferenceDeployment).filter(
        InferenceDeployment.node_id == ComputeNode.id  # Optional: scope to user if needed
    ).update({"status": StatusEnum.offline})

    # 3. Scheduler: Find hardware (Assume 4GB for base 1.5B model)
    node_id = find_available_node(db, required_vram=4.0)

    # 4. Create the Deployment Ticket (NULL fine_tuned_model_id = Standard Mode)
    deployment_id = IdentifierService.generate_prefixed_id("dep")
    deployment = InferenceDeployment(
        id=deployment_id,
        node_id=node_id,
        base_model_id=base.id,
        fine_tuned_model_id=None,  # <--- THIS makes it a standard model
        port=8000,
        status=StatusEnum.pending,
    )

    # 5. Lock VRAM
    allocation = GPUAllocation(node_id=node_id, vram_reserved_gb=4.0)

    db.add(deployment)
    db.add(allocation)
    db.commit()

    return {
        "status": "deploying_standard",
        "model": base.id,
        "node": node_id,
        "next_step": f"platform-api --node {node_id} up --services vllm",
    }
