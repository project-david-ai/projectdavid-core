# src/api/training/services/deployment_service.py
"""
deployment_service.py

Service layer for inference deployment lifecycle management.

Responsibilities:
  - Activating base models and fine-tuned models for inference
  - Deactivating deployments (single or all)
  - Querying Ray cluster state for node selection and capacity checks
  - Creating and managing InferenceDeployment records

Boundary rule:
  This service NEVER queries the BaseModel table directly.
  All BaseModel lookups are delegated to RegistryService.
  This service owns: InferenceDeployment, FineTunedModel.

Ray integration:
  Activation creates an InferenceDeployment record with status=pending.
  The InferenceReconciler (inference_worker.py) picks it up on its next
  poll cycle and deploys the corresponding Ray Serve application.
  This service does NOT communicate with Ray Serve directly.
"""

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

from src.api.training.models.models import FineTunedModel, InferenceDeployment
from src.api.training.services.registry_service import RegistryService

logging_utility = LoggingUtility()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAY_DASHBOARD_URL = "http://inference_worker:8265"
RAY_SERVE_URL = "http://inference_worker:8000"


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class DeploymentService:
    """
    Service layer for inference deployment lifecycle.

    All BaseModel lookups are delegated to RegistryService.
    This service owns InferenceDeployment and FineTunedModel records only.
    """

    def __init__(self, db: Session) -> None:
        self.db = db
        self.registry = RegistryService(db)

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

        default_node = os.getenv("DEFAULT_NODE_ID", f"node_{socket.gethostname()}")
        logging_utility.warning(
            "No Ray node meets resource requirements — using default node ID: %s",
            default_node,
        )
        return default_node

    def _check_node_capacity(self, node_id: str, tensor_parallel_size: int = 1) -> None:
        """
        Raises 507 if the target node cannot accommodate another deployment.

        Ray reports resources_available["GPU"] = 0 when the GPU is reserved
        at the cluster level but no active deployment currently holds it.
        When the DB has no active InferenceDeployment records we use
        resources_total as the effective GPU count — the GPU is physically
        free even if Ray's internal accounting shows 0 available.

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
            return  # Fail open

        for node in nodes:
            if node.get("node_id") != node_id:
                continue

            resources_available = node.get("resources_available", {})
            resources_total = node.get("resources_total", {})

            available_gpu = resources_available.get("GPU", 0.0)
            total_gpu = resources_total.get("GPU", 0.0)

            # When no deployments are active in the DB, Ray may still report
            # resources_available["GPU"] = 0 due to internal cluster reservations.
            # Fall back to resources_total which reflects actual hardware capacity.
            # If active deployments exist, use resources_available which correctly
            # accounts for live GPU allocations held by Ray Serve applications.
            active_deployments = self.db.query(InferenceDeployment).count()
            effective_gpu = available_gpu if active_deployments > 0 else total_gpu

            logging_utility.info(
                "GPU capacity check — node=%s available=%.1f total=%.1f "
                "active_deployments=%d effective=%.1f required=%d",
                node_id[:16],
                available_gpu,
                total_gpu,
                active_deployments,
                effective_gpu,
                tensor_parallel_size,
            )

            if effective_gpu < tensor_parallel_size:
                raise HTTPException(
                    status_code=507,
                    detail=(
                        f"Node {node_id[:16]}... has insufficient available GPUs. "
                        f"Required: {tensor_parallel_size}, "
                        f"Available: {effective_gpu:.0f}. "
                        f"Active Ray Serve deployments may be holding GPU resources."
                    ),
                )
            return

    def _get_serve_route(self, deployment_id: str) -> str:
        """Returns the Ray Serve HTTP route for a given deployment ID."""
        name = f"vllm_{deployment_id.replace('-', '_')}"
        return f"{RAY_SERVE_URL}/{name}"

    # ------------------------------------------------------------------
    # Fine-tuned model retrieval
    # ------------------------------------------------------------------

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
            raise HTTPException(status_code=404, detail="Fine-tuned model not found.")
        return model

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

    def soft_delete_model(self, model_id: str, user_id: Optional[str] = None) -> dict:
        """Soft-delete a fine-tuned model."""
        model = self.get_fine_tuned_model(model_id, user_id)
        model.deleted_at = int(time.time())
        model.is_active = False
        self.db.commit()
        return {"status": "deleted", "model_id": model_id}

    # ------------------------------------------------------------------
    # Deployment listing
    # ------------------------------------------------------------------

    def list_deployments(self) -> List[InferenceDeployment]:
        """Return all active InferenceDeployment records."""
        return (
            self.db.query(InferenceDeployment)
            .order_by(InferenceDeployment.last_seen.desc())
            .all()
        )

    # ------------------------------------------------------------------
    # Deactivation
    # ------------------------------------------------------------------

    def _clear_all_deployments(self) -> None:
        """
        Internal: remove all InferenceDeployment records.

        The InferenceReconciler detects the missing records on its next
        poll and tears down the corresponding Ray Serve applications,
        releasing GPU reservations back to the cluster.
        """
        self.db.query(InferenceDeployment).delete(synchronize_session=False)
        self.db.commit()

    def deactivate_all_for_user(self, user_id: str) -> dict:
        """
        Deactivate all active fine-tuned model deployments for a user.
        Also clears all InferenceDeployment records (cluster clean slate).
        """
        self.db.query(FineTunedModel).filter(
            FineTunedModel.user_id == user_id,
            FineTunedModel.is_active,
        ).update({"is_active": False}, synchronize_session=False)

        self._clear_all_deployments()
        return {"status": "success", "message": "Cluster resources released."}

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
        """
        Surgically deactivate a single base model deployment.

        Accepts either a bm_... ID or an HF path — delegates resolution
        to RegistryService.
        """
        base = self.registry.resolve(base_model_id)

        self.db.query(InferenceDeployment).filter(
            InferenceDeployment.base_model_id == base.id,
            InferenceDeployment.fine_tuned_model_id.is_(None),
        ).delete(synchronize_session=False)

        self.db.commit()
        return {"status": "deactivated", "base_model_id": base.id}

    # ------------------------------------------------------------------
    # Activation
    # ------------------------------------------------------------------

    def activate_base_model(
        self,
        base_model_id: str,
        target_node_id: Optional[str] = None,
        tensor_parallel_size: int = 1,
    ) -> dict:
        """
        Schedule a standard backbone model (no LoRA adapter) for deployment.

        Accepts either a bm_... ID or an HF path — delegates resolution
        to RegistryService.

        Flow:
          1. Resolve base_model_id → BaseModel record via RegistryService
          2. Clear all existing deployments (cluster clean slate)
          3. Select target node (fails open if dashboard unreachable)
          4. Capacity guard (fails open if dashboard unreachable)
          5. Create InferenceDeployment record with status=pending
          6. InferenceReconciler picks up on next poll and deploys via Ray Serve
        """
        base = self.registry.resolve(base_model_id)

        self._clear_all_deployments()

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

        logging_utility.info(
            "DeploymentService: base model activation scheduled. "
            "base_model_id=%s deployment_id=%s node=%s",
            base.id,
            deployment_id,
            node_id,
        )

        return {
            "status": "deploying",
            "model_id": base.id,
            "hf_path": base.endpoint,
            "node": node_id,
            "tensor_parallel_size": tensor_parallel_size,
            "serve_route": serve_route,
            "next_step": "InferenceReconciler will deploy via Ray Serve on next poll.",
        }

    def activate_fine_tuned_model(
        self,
        model_id: str,
        target_node_id: Optional[str] = None,
        tensor_parallel_size: int = 1,
    ) -> dict:
        """
        Schedule a fine-tuned model (base + LoRA adapter) for deployment.

        Flow:
          1. Fetch FineTunedModel record (admin bypass — no user_id scope)
          2. Deactivate all existing deployments for that user
          3. Resolve base model via RegistryService
          4. Select target node (fails open if dashboard unreachable)
          5. Capacity guard (fails open if dashboard unreachable)
          6. Create InferenceDeployment record with status=pending
          7. InferenceReconciler picks up on next poll and deploys via Ray Serve
        """
        model = self.get_fine_tuned_model(model_id)

        self.deactivate_all_for_user(model.user_id)

        base = self.registry.resolve(model.base_model)

        node_id = target_node_id or self._find_available_node(
            required_vram=5.0,
            tensor_parallel_size=tensor_parallel_size,
        )

        self._check_node_capacity(node_id, tensor_parallel_size=tensor_parallel_size)

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

        logging_utility.info(
            "DeploymentService: fine-tuned model activation scheduled. "
            "model_id=%s base_model_id=%s deployment_id=%s node=%s",
            model.id,
            base.id,
            deployment_id,
            node_id,
        )

        return {
            "status": "deploying",
            "model_id": model.id,
            "base_model_id": base.id,
            "hf_path": base.endpoint,
            "node": node_id,
            "tensor_parallel_size": tensor_parallel_size,
            "serve_route": serve_route,
            "next_step": "InferenceReconciler will deploy via Ray Serve on next poll.",
        }
