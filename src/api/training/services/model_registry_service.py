import time
from typing import List, Optional

import httpx
from fastapi import HTTPException
from projectdavid_common.schemas.enums import StatusEnum
from projectdavid_common.utilities.identifier_service import IdentifierService
from sqlalchemy.orm import Session

from src.api.training.models.models import (
    BaseModel,
    FineTunedModel,
    InferenceDeployment,
)

# ---------------------------------------------------------------------------
# Ray Dashboard HTTP API
# ---------------------------------------------------------------------------

RAY_DASHBOARD_URL = "http://training_worker:8265"


def _get_ray_nodes() -> list:
    """
    Queries the Ray dashboard REST API for live node state.
    Returns a list of ALIVE node dicts with resources_total.
    """
    try:
        resp = httpx.get(
            f"{RAY_DASHBOARD_URL}/api/v0/nodes?detail=True",
            timeout=5.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Ray cluster unreachable — cannot schedule: {exc}",
        )

    nodes = data.get("data", {}).get("result", {}).get("result", [])
    return [n for n in nodes if isinstance(n, dict) and n.get("state") == "ALIVE"]


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
def find_available_node(
    db: Session,
    required_vram: float = 4.0,
    tensor_parallel_size: int = 1,
) -> str:
    """
    Phase 4 + sharding: Selects a node from Ray cluster state.

    tensor_parallel_size > 1 means the deployment will span multiple GPUs.
    The scheduler checks that the selected node has at least
    tensor_parallel_size GPUs available before confirming the slot.

    Returns the Ray node ID (hex string).
    """
    nodes = _get_ray_nodes()

    if not nodes:
        raise HTTPException(
            status_code=507,
            detail="No active nodes found in Ray cluster.",
        )

    nodes_sorted = sorted(
        nodes,
        key=lambda n: n.get("resources_total", {}).get("memory", 0.0),
        reverse=True,
    )

    for node in nodes_sorted:
        resources = node.get("resources_total", {})
        available_gpu = resources.get("GPU", 0.0)
        available_memory_gb = resources.get("memory", 0.0) / (1024**3)

        # For tensor parallel deployments we need N GPUs on the same node
        if available_gpu < tensor_parallel_size:
            continue

        if available_memory_gb < required_vram:
            continue

        node_id = node.get("node_id", "")
        if not node_id:
            continue

        return node_id

    raise HTTPException(
        status_code=507,
        detail=(
            f"No Ray node has sufficient resources. "
            f"Required: {tensor_parallel_size} GPU(s) + {required_vram:.1f} GB memory."
        ),
    )


# ---------------------------------------------------------------------------
# Deployment Logic
# ---------------------------------------------------------------------------


def deactivate_all_models(db: Session, user_id: str) -> dict:
    """
    CLEAN SLATE: removes deployments to satisfy port constraints.
    Phase 2+: GPUAllocation deletes removed — Ray releases reservations.
    Phase 4: compute_nodes not touched.
    Phase 5 candidate: node_id column removed from update — FK references
    compute_nodes which is a legacy table.
    """
    db.query(FineTunedModel).filter(
        FineTunedModel.user_id == user_id, FineTunedModel.is_active
    ).update({"is_active": False}, synchronize_session=False)

    db.query(InferenceDeployment).filter(
        InferenceDeployment.node_id.is_not(None)
    ).delete(synchronize_session=False)

    db.commit()
    return {"status": "success", "message": "Cluster resources released."}


def activate_model(
    db: Session,
    model_id: str,
    user_id: str,
    target_node_id: Optional[str] = None,
    tensor_parallel_size: int = 1,
) -> dict:
    """
    DEPLOYS A FINE-TUNED MODEL (Base + LoRA).

    tensor_parallel_size: number of GPUs to shard the model across.
    Default 1 = single GPU, backward compatible.
    """
    model = get_fine_tuned_model(db, model_id, user_id)

    deactivate_all_models(db, user_id)

    node_id = target_node_id or find_available_node(
        db,
        required_vram=5.0,
        tensor_parallel_size=tensor_parallel_size,
    )

    deployment_id = IdentifierService.generate_prefixed_id("dep")
    deployment = InferenceDeployment(
        id=deployment_id,
        node_id=node_id,
        base_model_id=model.base_model,
        fine_tuned_model_id=model.id,
        port=8001,
        status=StatusEnum.pending,
        last_seen=int(time.time()),
        tensor_parallel_size=tensor_parallel_size,
    )

    model.is_active = True
    # node_id removed from model update — FK references compute_nodes
    # which is a legacy table. Phase 5 will drop this column entirely.
    db.add(deployment)
    db.commit()

    return {
        "status": "deploying",
        "model_id": model.id,
        "node": node_id,
        "tensor_parallel_size": tensor_parallel_size,
        "next_step": "Worker is provisioning LoRA weights.",
    }


def activate_base_model(
    db: Session,
    base_model_id: str,
    user_id: str,
    target_node_id: Optional[str] = None,
    tensor_parallel_size: int = 1,
) -> dict:
    """
    DEPLOYS A STANDARD MODEL (Backbone only).

    tensor_parallel_size: number of GPUs to shard the model across.
    Default 1 = single GPU, backward compatible.
    """
    base = db.query(BaseModel).filter(BaseModel.id == base_model_id).first()
    if not base:
        raise HTTPException(
            status_code=404, detail=f"Base model {base_model_id} not found."
        )

    deactivate_all_models(db, user_id)

    node_id = target_node_id or find_available_node(
        db,
        required_vram=4.0,
        tensor_parallel_size=tensor_parallel_size,
    )

    deployment_id = IdentifierService.generate_prefixed_id("dep")
    deployment = InferenceDeployment(
        id=deployment_id,
        node_id=node_id,
        base_model_id=base.id,
        fine_tuned_model_id=None,
        port=8001,
        status=StatusEnum.pending,
        last_seen=int(time.time()),
        tensor_parallel_size=tensor_parallel_size,
    )

    db.add(deployment)
    db.commit()

    return {
        "status": "deploying_standard",
        "model_id": base.id,
        "node": node_id,
        "tensor_parallel_size": tensor_parallel_size,
        "next_step": f"Standard backbone {base.id} is being provisioned.",
    }
