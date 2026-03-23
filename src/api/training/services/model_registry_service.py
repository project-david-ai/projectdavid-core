import time
from typing import List, Optional

import ray
from fastapi import HTTPException
from projectdavid_common.schemas.enums import StatusEnum
from projectdavid_common.utilities.identifier_service import IdentifierService
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


# ---------------------------------------------------------------------------
# Cluster Resource Management & Scheduling
# ---------------------------------------------------------------------------


def find_available_node(db: Session, required_vram: float = 4.0) -> str:
    """
    Phase 2: Queries the Ray cluster for live resource availability instead
    of summing rows in the gpu_allocations ledger.

    Ray tracks reservations implicitly as tasks and actors consume resources —
    available_resources() always reflects the true current free capacity.

    Falls back to HTTP 507 if the cluster has insufficient GPU or memory.
    """
    try:
        available = ray.available_resources()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Ray cluster unreachable — cannot schedule: {exc}",
        )

    available_gpu = available.get("GPU", 0.0)
    # Ray reports memory in bytes — convert to GB for comparison
    available_memory_gb = available.get("memory", 0.0) / (1024**3)

    if available_gpu < 1.0:
        raise HTTPException(
            status_code=507,
            detail=f"No GPU available in Ray cluster. Available: {available_gpu} GPU(s).",
        )

    if available_memory_gb < required_vram:
        raise HTTPException(
            status_code=507,
            detail=(
                f"Insufficient memory in Ray cluster. "
                f"Required: {required_vram:.1f} GB, "
                f"Available: {available_memory_gb:.1f} GB."
            ),
        )

    # Resolve the matching compute_node record.
    # compute_nodes still drives the deployment ticket (node_id FK).
    # Phase 4 will drop this table once Ray owns node discovery entirely.
    nodes = db.query(ComputeNode).filter(ComputeNode.status == StatusEnum.active).all()
    if not nodes:
        raise HTTPException(
            status_code=507,
            detail="No active compute nodes registered in the ledger.",
        )

    # Return the first active node — Ray has already confirmed resources
    # are available across the cluster.
    return nodes[0].id


# ---------------------------------------------------------------------------
# Deployment Logic (The Mesh Implementation)
# ---------------------------------------------------------------------------


def deactivate_all_models(db: Session, user_id: str) -> dict:
    """
    CLEAN SLATE: Physically removes deployments to satisfy UniqueConstraints
    and free up hardware ports.

    Phase 2: GPUAllocation deletes removed — Ray releases reservations
    implicitly when tasks/actors complete. No manual ledger cleanup needed.
    """
    # 1. Reset metadata flags for the user
    db.query(FineTunedModel).filter(
        FineTunedModel.user_id == user_id, FineTunedModel.is_active
    ).update({"is_active": False, "node_id": None}, synchronize_session=False)

    # 2. HARD DELETE deployments — clears the uq_node_port_deployment constraint
    # so we can insert new models on the same port immediately.
    db.query(InferenceDeployment).filter(InferenceDeployment.node_id.is_not(None)).delete(
        synchronize_session=False
    )

    db.commit()
    return {"status": "success", "message": "Cluster resources released."}


def activate_model(
    db: Session, model_id: str, user_id: str, target_node_id: Optional[str] = None
) -> dict:
    """
    DEPLOYS A FINE-TUNED MODEL (Base + LoRA).

    Phase 2: GPUAllocation row removed — Ray tracks resource consumption
    implicitly. find_available_node() now queries ray.available_resources().
    """
    model = get_fine_tuned_model(db, model_id, user_id)

    # 1. Clear existing slots first
    deactivate_all_models(db, user_id)

    # 2. Schedule — Ray confirms capacity, returns node_id from compute_nodes
    node_id = target_node_id or find_available_node(db, required_vram=5.0)

    # 3. Create deployment ticket
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

    # 4. Persist — no GPUAllocation row, Ray owns the reservation
    model.is_active = True
    model.node_id = node_id
    db.add(deployment)
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

    Phase 2: GPUAllocation row removed — Ray tracks resource consumption
    implicitly. find_available_node() now queries ray.available_resources().
    """
    # 1. Verify model exists in our seeded catalog
    base = db.query(BaseModel).filter(BaseModel.id == base_model_id).first()
    if not base:
        raise HTTPException(status_code=404, detail=f"Base model {base_model_id} not found.")

    # 2. Mutex: clear existing deployments to free Node Port 8001
    deactivate_all_models(db, user_id)

    # 3. Scheduler: Ray confirms capacity
    node_id = target_node_id or find_available_node(db, required_vram=4.0)

    # 4. Create deployment ticket (fine_tuned_model_id=None signals standard mode)
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

    # 5. Persist — no GPUAllocation row, Ray owns the reservation
    db.add(deployment)
    db.commit()

    return {
        "status": "deploying_standard",
        "model_id": base.id,
        "node": node_id,
        "next_step": f"Standard backbone {base.id} is being provisioned.",
    }
