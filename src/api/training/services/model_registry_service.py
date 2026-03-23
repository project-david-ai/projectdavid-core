import time
from typing import List, Optional

import httpx
from fastapi import HTTPException
from projectdavid_common.schemas.enums import StatusEnum
from projectdavid_common.utilities.identifier_service import IdentifierService
from sqlalchemy.orm import Session

from src.api.training.models.models import BaseModel, FineTunedModel, InferenceDeployment

# ---------------------------------------------------------------------------
# Ray Dashboard HTTP API
# ---------------------------------------------------------------------------

RAY_DASHBOARD_URL = "http://training_worker:8265"


def _get_ray_nodes() -> list:
    """
    Phase 4: Queries the Ray dashboard REST API for live node state.

    Returns a list of ALIVE node dicts, each containing:
      - node_id    : Ray hex node ID (stable identifier for this machine)
      - node_ip    : dotted-quad IP on the Docker network
      - resources_total: dict of GPU, CPU, memory (bytes)

    No GCS connection required — pure HTTP to the dashboard port.
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


def _get_ray_available_resources() -> dict:
    """
    Returns aggregated available resources across all ALIVE Ray nodes.
    Used by find_available_node() for cluster-level capacity checks.
    """
    nodes = _get_ray_nodes()
    total_gpu = sum(n.get("resources_total", {}).get("GPU", 0.0) for n in nodes)
    total_memory = sum(n.get("resources_total", {}).get("memory", 0.0) for n in nodes)
    return {"GPU": total_gpu, "memory": total_memory}


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
    Phase 4: Selects a node entirely from Ray cluster state.

    The compute_nodes DB table is no longer queried — Ray is the sole
    source of truth for node availability and resource capacity.

    Returns the Ray node ID (hex string) of the best available node,
    ordered by free memory descending for implicit load balancing.
    The node ID is written to inference_deployments.node_id for
    traceability and matches what appears in the Ray dashboard.

    Phase 5 will drop the compute_nodes table entirely.
    """
    nodes = _get_ray_nodes()

    if not nodes:
        raise HTTPException(
            status_code=507,
            detail="No active nodes found in Ray cluster.",
        )

    # Sort by available memory descending — pick the least loaded node
    def _free_memory(node: dict) -> float:
        return node.get("resources_total", {}).get("memory", 0.0)

    nodes_sorted = sorted(nodes, key=_free_memory, reverse=True)

    for node in nodes_sorted:
        resources = node.get("resources_total", {})
        available_gpu = resources.get("GPU", 0.0)
        available_memory_gb = resources.get("memory", 0.0) / (1024**3)

        if available_gpu < 1.0:
            continue

        if available_memory_gb < required_vram:
            continue

        node_id = node.get("node_id", "")
        node_ip = node.get("node_ip", "unknown")

        if not node_id:
            continue

        return node_id

    raise HTTPException(
        status_code=507,
        detail=(
            f"No Ray node has sufficient resources. "
            f"Required: 1 GPU + {required_vram:.1f} GB memory."
        ),
    )


# ---------------------------------------------------------------------------
# Deployment Logic (The Mesh Implementation)
# ---------------------------------------------------------------------------


def deactivate_all_models(db: Session, user_id: str) -> dict:
    """
    CLEAN SLATE: Physically removes deployments to satisfy UniqueConstraints
    and free up hardware ports.

    Phase 2+: GPUAllocation deletes removed — Ray releases reservations
    implicitly when tasks/actors complete.
    Phase 4: compute_nodes no longer touched.
    """
    db.query(FineTunedModel).filter(
        FineTunedModel.user_id == user_id, FineTunedModel.is_active
    ).update({"is_active": False, "node_id": None}, synchronize_session=False)

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

    Phase 4: node_id written to inference_deployments is now the Ray node ID
    (hex string) rather than the legacy compute_nodes primary key.
    """
    model = get_fine_tuned_model(db, model_id, user_id)

    deactivate_all_models(db, user_id)

    node_id = target_node_id or find_available_node(db, required_vram=5.0)

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

    Phase 4: node_id written to inference_deployments is now the Ray node ID.
    """
    base = db.query(BaseModel).filter(BaseModel.id == base_model_id).first()
    if not base:
        raise HTTPException(status_code=404, detail=f"Base model {base_model_id} not found.")

    deactivate_all_models(db, user_id)

    node_id = target_node_id or find_available_node(db, required_vram=4.0)

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

    db.add(deployment)
    db.commit()

    return {
        "status": "deploying_standard",
        "model_id": base.id,
        "node": node_id,
        "next_step": f"Standard backbone {base.id} is being provisioned.",
    }
