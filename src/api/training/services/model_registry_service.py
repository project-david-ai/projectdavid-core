import os
import socket
import time
from typing import List, Optional

import httpx
from fastapi import HTTPException
from projectdavid_common.schemas.enums import StatusEnum
from projectdavid_common.utilities.identifier_service import IdentifierService
from projectdavid_common.utilities.logging_service import LoggingUtility
from sqlalchemy.orm import Session

from src.api.training.models.models import (
    BaseModel,
    FineTunedModel,
    InferenceDeployment,
)
from src.api.training.services.registry_service import RegistryService

logging_utility = LoggingUtility()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAY_DASHBOARD_URL = "http://training_worker:8265"
RAY_SERVE_URL = "http://inference_worker:8000"


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class ModelRegistryService:
    """
    Service layer for fine-tuned model lifecycle management.

    Inference is managed by inference_worker.py via Ray Serve.
    GPU resources are reserved natively within the Ray cluster —
    activation creates an InferenceDeployment record which the
    InferenceReconciler picks up and deploys as a Ray Serve application.

    Ray dashboard connectivity is treated as optional — if the dashboard
    is unreachable the service fails open and lets the InferenceReconciler
    handle all scheduling decisions.
    """

    def __init__(self, db: Session) -> None:
        self.db = db

    # ------------------------------------------------------------------
    # Ray cluster interface
    # ------------------------------------------------------------------

    def _get_ray_nodes(self) -> list:
        """
        Queries the Ray dashboard REST API for live node state.
        Returns a list of ALIVE node dicts with resources_total.

        Fails open — returns empty list if the dashboard is unreachable
        so that activation always succeeds and the InferenceReconciler
        handles scheduling.
        """
        try:
            resp = httpx.get(
                f"{RAY_DASHBOARD_URL}/api/v0/nodes?detail=True",
                timeout=5.0,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logging_utility.warning(
                "Ray dashboard unreachable — skipping node check, "
                "proceeding with activation. Error: %s",
                exc,
            )
            return []

        nodes = data.get("data", {}).get("result", {}).get("result", [])
        return [n for n in nodes if isinstance(n, dict) and n.get("state") == "ALIVE"]

    def _find_available_node(
        self,
        required_vram: float = 4.0,
        tensor_parallel_size: int = 1,
    ) -> str:
        """
        Selects the most resource-rich ALIVE node from the Ray cluster.

        Falls back to a default node ID if the dashboard is unreachable —
        the InferenceReconciler will handle actual placement.

        Returns the Ray node ID (hex string).
        """
        nodes = self._get_ray_nodes()

        if not nodes:
            # Dashboard unreachable or no nodes visible — return a default
            # node ID and let InferenceReconciler handle scheduling.
            default_node = os.getenv("DEFAULT_NODE_ID", f"node_{socket.gethostname()}")
            logging_utility.warning(
                "No Ray nodes visible — using default node ID: %s", default_node
            )
            return default_node

        nodes_sorted = sorted(
            nodes,
            key=lambda n: n.get("resources_total", {}).get("memory", 0.0),
            reverse=True,
        )

        for node in nodes_sorted:
            resources = node.get("resources_total", {})
            available_gpu = resources.get("GPU", 0.0)
            available_memory_gb = resources.get("memory", 0.0) / (1024**3)

            if available_gpu < tensor_parallel_size:
                continue
            if available_memory_gb < required_vram:
                continue

            node_id = node.get("node_id", "")
            if not node_id:
                continue

            return node_id

        # Nodes found but none meet requirements — still fail open with default
        default_node = os.getenv("DEFAULT_NODE_ID", f"node_{socket.gethostname()}")
        logging_utility.warning(
            "No Ray node meets resource requirements — using default node ID: %s",
            default_node,
        )
        return default_node

    def _check_node_capacity(self, node_id: str, tensor_parallel_size: int = 1) -> None:
        """
        Raises 507 if the target node cannot accommodate another deployment.

        Fails open if Ray dashboard is unreachable — the InferenceReconciler
        will handle deployment failure gracefully on its next poll cycle.
        """
        try:
            resp = httpx.get(
                f"{RAY_DASHBOARD_URL}/api/v0/nodes?detail=True",
                timeout=5.0,
            )
            resp.raise_for_status()
            nodes = resp.json().get("data", {}).get("result", {}).get("result", [])
        except Exception:
            # Fail open
            return

        for node in nodes:
            if node.get("node_id") != node_id:
                continue

            resources_available = node.get("resources_available", {})
            available_gpu = resources_available.get("GPU", 0.0)

            if available_gpu < tensor_parallel_size:
                raise HTTPException(
                    status_code=507,
                    detail=(
                        f"Node {node_id[:16]}... has insufficient available GPUs. "
                        f"Required: {tensor_parallel_size}, "
                        f"Available: {available_gpu:.0f}. "
                        f"Active Ray Serve deployments may be holding GPU resources."
                    ),
                )
            return

    def _get_serve_route(self, deployment_id: str) -> str:
        """Returns the Ray Serve HTTP route for a given deployment ID."""
        name = f"vllm_{deployment_id.replace('-', '_')}"
        return f"{RAY_SERVE_URL}/{name}"

    # ------------------------------------------------------------------
    # Registry retrieval
    # ------------------------------------------------------------------

    def list_fine_tuned_models(
        self, user_id: str, limit: int = 50, offset: int = 0
    ) -> List[FineTunedModel]:
        """Return paginated fine-tuned models for a user."""
        return (
            self.db.query(FineTunedModel)
            .filter(
                FineTunedModel.user_id == user_id,
                FineTunedModel.deleted_at.is_(None),
            )
            .order_by(FineTunedModel.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

    def get_fine_tuned_model(
        self, model_id: str, user_id: Optional[str] = None
    ) -> FineTunedModel:
        """
        Fetch a fine-tuned model by ID.

        user_id is optional — when omitted (admin context) the user scope
        filter is bypassed, allowing admins to activate any user's model.
        """
        query = self.db.query(FineTunedModel).filter(
            FineTunedModel.id == model_id,
            FineTunedModel.deleted_at.is_(None),
        )
        if user_id:
            query = query.filter(FineTunedModel.user_id == user_id)
        model = query.first()
        if not model:
            raise HTTPException(status_code=404, detail="Model not found.")
        return model

    def soft_delete_model(self, model_id: str, user_id: Optional[str] = None) -> dict:
        """Soft-delete a fine-tuned model."""
        model = self.get_fine_tuned_model(model_id, user_id)
        model.deleted_at = int(time.time())
        model.is_active = False
        self.db.commit()
        return {"status": "deleted", "model_id": model_id}

    # ------------------------------------------------------------------
    # Deployment lifecycle
    # ------------------------------------------------------------------

    def deactivate_all_models(self, user_id: str) -> dict:
        """
        CLEAN SLATE: deactivates all active deployments for a user.

        Removes InferenceDeployment records — the InferenceReconciler will
        detect the missing records on its next poll and delete the corresponding
        Ray Serve deployments, releasing GPU reservations back to the cluster.
        """
        self.db.query(FineTunedModel).filter(
            FineTunedModel.user_id == user_id,
            FineTunedModel.is_active,
        ).update({"is_active": False}, synchronize_session=False)

        self.db.query(InferenceDeployment).filter(
            InferenceDeployment.node_id.is_not(None)
        ).delete(synchronize_session=False)

        self.db.commit()
        return {"status": "success", "message": "Cluster resources released."}

    def activate_model(
        self,
        model_id: str,
        user_id: str,
        target_node_id: Optional[str] = None,
        tensor_parallel_size: int = 1,
    ) -> dict:
        """
        Schedule a fine-tuned model (base + LoRA adapter) for deployment.

        Flow:
          1. Fetch model record (admin bypass — no user_id scope filter)
          2. Deactivate all existing deployments for that user
          3. Select or validate target node (fails open if dashboard unreachable)
          4. Capacity guard (fails open if dashboard unreachable)
          5. Resolve HF path → bm_... ID via RegistryService
          6. Create InferenceDeployment record
          7. InferenceReconciler picks up on next poll and deploys via Ray Serve
        """
        model = self.get_fine_tuned_model(model_id)

        self.deactivate_all_models(model.user_id)

        node_id = target_node_id or self._find_available_node(
            required_vram=5.0,
            tensor_parallel_size=tensor_parallel_size,
        )

        self._check_node_capacity(node_id, tensor_parallel_size=tensor_parallel_size)

        registry = RegistryService(self.db)
        base = registry.resolve(model.base_model)

        deployment_id = IdentifierService.generate_prefixed_id("dep")
        serve_route = self._get_serve_route(deployment_id)

        deployment = InferenceDeployment(
            id=deployment_id,
            node_id=node_id,
            base_model_id=base.id,
            fine_tuned_model_id=model.id,
            port=8000,
            status=StatusEnum.pending,
            last_seen=int(time.time()),
            tensor_parallel_size=tensor_parallel_size,
            internal_hostname=serve_route,
        )

        model.is_active = True
        self.db.add(deployment)
        self.db.commit()

        return {
            "status": "deploying",
            "model_id": model.id,
            "node": node_id,
            "tensor_parallel_size": tensor_parallel_size,
            "serve_route": serve_route,
            "next_step": "InferenceReconciler will deploy via Ray Serve on next poll.",
        }

    def activate_base_model(
        self,
        base_model_id: str,
        user_id: str,
        target_node_id: Optional[str] = None,
        tensor_parallel_size: int = 1,
    ) -> dict:
        """Schedule a standard backbone model (no LoRA adapter) for deployment."""
        base = self.db.query(BaseModel).filter(BaseModel.id == base_model_id).first()
        if not base:
            raise HTTPException(
                status_code=404,
                detail=f"Base model {base_model_id} not found.",
            )

        self.deactivate_all_models(user_id)

        node_id = target_node_id or self._find_available_node(
            required_vram=4.0,
            tensor_parallel_size=tensor_parallel_size,
        )

        self._check_node_capacity(node_id, tensor_parallel_size=tensor_parallel_size)

        deployment_id = IdentifierService.generate_prefixed_id("dep")
        serve_route = self._get_serve_route(deployment_id)

        deployment = InferenceDeployment(
            id=deployment_id,
            node_id=node_id,
            base_model_id=base.id,
            fine_tuned_model_id=None,
            port=8000,
            status=StatusEnum.pending,
            last_seen=int(time.time()),
            tensor_parallel_size=tensor_parallel_size,
            internal_hostname=serve_route,
        )

        self.db.add(deployment)
        self.db.commit()

        return {
            "status": "deploying_standard",
            "model_id": base.id,
            "node": node_id,
            "tensor_parallel_size": tensor_parallel_size,
            "serve_route": serve_route,
            "next_step": "InferenceReconciler will deploy via Ray Serve on next poll.",
        }

    def deactivate_model(self, model_id: str, user_id: Optional[str] = None) -> dict:
        """Surgically deactivate a single fine-tuned model deployment."""
        model = self.get_fine_tuned_model(model_id, user_id)

        self.db.query(InferenceDeployment).filter(
            InferenceDeployment.fine_tuned_model_id == model.id
        ).delete(synchronize_session=False)

        model.is_active = False
        self.db.commit()

        return {"status": "deactivated", "model_id": model.id}

    def deactivate_base_model(self, base_model_id: str) -> dict:
        """Surgically deactivate a single base model deployment."""
        self.db.query(InferenceDeployment).filter(
            InferenceDeployment.base_model_id == base_model_id,
            InferenceDeployment.fine_tuned_model_id.is_(None),
        ).delete(synchronize_session=False)

        self.db.commit()
        return {"status": "deactivated", "base_model_id": base_model_id}
