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
from src.api.training.services.registry_service import RegistryService

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAY_DASHBOARD_URL = "http://training_worker:8265"


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class ModelRegistryService:
    """
    Service layer for fine-tuned model lifecycle management.

    Responsibilities:
      - Listing and retrieving fine-tuned model records
      - Scheduling deployments against the Ray cluster
      - Capacity enforcement before deployment
      - Activating and deactivating fine-tuned and base model deployments

    The DB session is injected at construction time and reused across all
    operations — callers do not need to pass it to individual methods.
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
        Raises 503 if the cluster is unreachable.
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

    def _find_available_node(
        self,
        required_vram: float = 4.0,
        tensor_parallel_size: int = 1,
    ) -> str:
        """
        Selects the most resource-rich ALIVE node from the Ray cluster.

        For tensor parallel deployments, verifies the node has at least
        tensor_parallel_size GPUs before confirming the slot.

        Returns the Ray node ID (hex string).
        Raises 507 if no suitable node is found.
        """
        nodes = self._get_ray_nodes()

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

    def _check_node_capacity(self, node_id: str, required_vram_gb: float = 5.0) -> None:
        """
        Raises 507 if the target node cannot accommodate another deployment.

        Called after deactivate_all_models() so that resources freed by
        eviction are accounted for before the capacity check fires.

        Fails open if Ray is unreachable — the DeploymentSupervisor will
        handle container failure gracefully if the node is actually OOM.
        """
        try:
            resp = httpx.get(
                f"{RAY_DASHBOARD_URL}/api/v0/nodes?detail=True",
                timeout=5.0,
            )
            resp.raise_for_status()
            nodes = resp.json().get("data", {}).get("result", {}).get("result", [])
        except Exception:
            # Fail open — do not block activation if Ray is temporarily unreachable
            return

        for node in nodes:
            if node.get("node_id") != node_id:
                continue

            resources = node.get("resources_total", {})
            total_gpu = resources.get("GPU", 0.0)
            total_memory_gb = resources.get("memory", 0.0) / (1024**3)

            if total_gpu < 1:
                raise HTTPException(
                    status_code=507,
                    detail=f"Node {node_id[:16]}... has no GPU resources available.",
                )

            if total_memory_gb < required_vram_gb:
                raise HTTPException(
                    status_code=507,
                    detail=(
                        f"Node {node_id[:16]}... has insufficient memory. "
                        f"Required: {required_vram_gb:.1f} GB, "
                        f"Available: {total_memory_gb:.1f} GB."
                    ),
                )
            return

        raise HTTPException(
            status_code=507,
            detail=f"Node {node_id[:16]}... not found in Ray cluster.",
        )

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
        When provided (user context) only models owned by that user are returned.
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

    # ------------------------------------------------------------------
    # Deployment lifecycle
    # ------------------------------------------------------------------

    def deactivate_all_models(self, user_id: str) -> dict:
        """
        CLEAN SLATE: deactivates all active deployments for a user.

        Removes InferenceDeployment records and marks FineTunedModels
        as inactive. The DeploymentSupervisor reconciles container state
        on its next poll cycle.

        Phase 5 candidate: node_id column removal from bulk update.
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
        Deploy a fine-tuned model (base + LoRA adapter).

        Flow:
          1. Fetch model record (admin bypass — no user_id scope filter)
          2. Deactivate all existing deployments for that user
          3. Select or validate target node
          4. Capacity guard — raises 507 if node is over-subscribed
          5. Resolve HF path → bm_... ID via RegistryService
          6. Create InferenceDeployment record
          7. DeploymentSupervisor picks up on next poll and spawns vLLM

        tensor_parallel_size: number of GPUs to shard across. Default 1.
        """
        model = self.get_fine_tuned_model(model_id)

        self.deactivate_all_models(model.user_id)

        node_id = target_node_id or self._find_available_node(
            required_vram=5.0,
            tensor_parallel_size=tensor_parallel_size,
        )

        # ── Capacity guard ────────────────────────────────────────────
        # Run after deactivation so freed resources are reflected.
        # Prevents over-subscription when vLLM containers consume GPU
        # resources outside Ray's task graph accounting.
        self._check_node_capacity(node_id, required_vram_gb=5.0)

        # ── Resolve HF path → bm_... ID ──────────────────────────────
        # fine_tuned_models.base_model stores the raw HF path.
        # inference_deployments.base_model_id FK requires a bm_... ID.
        registry = RegistryService(self.db)
        base = registry.resolve(model.base_model)

        deployment_id = IdentifierService.generate_prefixed_id("dep")
        deployment = InferenceDeployment(
            id=deployment_id,
            node_id=node_id,
            base_model_id=base.id,
            fine_tuned_model_id=model.id,
            port=8001,
            status=StatusEnum.pending,
            last_seen=int(time.time()),
            tensor_parallel_size=tensor_parallel_size,
        )

        model.is_active = True
        # node_id not written to FineTunedModel — FK references compute_nodes
        # which is a legacy table. Phase 5 will drop this column entirely.
        self.db.add(deployment)
        self.db.commit()

        return {
            "status": "deploying",
            "model_id": model.id,
            "node": node_id,
            "tensor_parallel_size": tensor_parallel_size,
            "next_step": "Worker is provisioning LoRA weights.",
        }

    def activate_base_model(
        self,
        base_model_id: str,
        user_id: str,
        target_node_id: Optional[str] = None,
        tensor_parallel_size: int = 1,
    ) -> dict:
        """
        Deploy a standard backbone model (no LoRA adapter).

        tensor_parallel_size: number of GPUs to shard across. Default 1.
        """
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

        # ── Capacity guard ────────────────────────────────────────────
        self._check_node_capacity(node_id, required_vram_gb=4.0)

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

        self.db.add(deployment)
        self.db.commit()

        return {
            "status": "deploying_standard",
            "model_id": base.id,
            "node": node_id,
            "tensor_parallel_size": tensor_parallel_size,
            "next_step": f"Standard backbone {base.id} is being provisioned.",
        }

    def deactivate_model(self, model_id: str, user_id: Optional[str] = None) -> dict:
        """
        Surgically deactivate a single fine-tuned model deployment.
        Admin bypass available via omitting user_id.
        """
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
        """
        self.db.query(InferenceDeployment).filter(
            InferenceDeployment.base_model_id == base_model_id,
            InferenceDeployment.fine_tuned_model_id.is_(None),
        ).delete(synchronize_session=False)

        self.db.commit()
        return {"status": "deactivated", "base_model_id": base_model_id}
