import os
import socket
import time
from typing import Any, Dict, List, Optional

import httpx
from fastapi import HTTPException
from projectdavid_common.schemas.enums import StatusEnum
from projectdavid_common.utilities.identifier_service import IdentifierService
from projectdavid_common.utilities.logging_service import LoggingUtility
from sqlalchemy.orm import Session

from src.api.training.models.models import FineTunedModel, InferenceDeployment
from src.api.training.services.registry_service import RegistryService

logging_utility = LoggingUtility()

RAY_DASHBOARD_URL = "http://inference_worker:8265"
RAY_SERVE_URL = "http://inference_worker:8000"

# Reconciler poll cadence — used to describe unload latency in deactivate responses.
# Must match INFERENCE_POLL_INTERVAL in inference_worker.py for accurate messaging.
_POLL_INTERVAL = int(os.getenv("INFERENCE_POLL_INTERVAL", "20"))


class DeploymentService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.registry = RegistryService(db)

    def _get_ray_nodes(self) -> list:
        try:
            resp = httpx.get(
                f"{RAY_DASHBOARD_URL}/api/v0/nodes?detail=True",
                timeout=5.0,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logging_utility.warning("Ray dashboard unreachable: %s", exc)
            return []

        nodes = data.get("data", {}).get("result", {}).get("result", [])
        return [n for n in nodes if isinstance(n, dict) and n.get("state") == "ALIVE"]

    def _find_available_node(
        self,
        required_vram: float = 4.0,
        tensor_parallel_size: int = 1,
    ) -> str:
        nodes = self._get_ray_nodes()

        if not nodes:
            default_node = os.getenv(
                "DEFAULT_NODE_ID",
                f"node_{socket.gethostname()}",
            )
            logging_utility.warning(
                "No Ray nodes visible — using default: %s",
                default_node,
            )
            return default_node

        nodes_sorted = sorted(
            nodes,
            key=lambda n: n.get("resources_total", {}).get("memory", 0.0),
            reverse=True,
        )

        for node in nodes_sorted:
            resources = node.get("resources_total", {})

            if resources.get("GPU", 0.0) < tensor_parallel_size:
                continue

            if resources.get("memory", 0.0) / (1024**3) < required_vram:
                continue

            node_id = node.get("node_id", "")
            if node_id:
                return node_id

        default_node = os.getenv(
            "DEFAULT_NODE_ID",
            f"node_{socket.gethostname()}",
        )
        logging_utility.warning(
            "No qualifying node found — using default: %s",
            default_node,
        )
        return default_node

    def _check_node_capacity(
        self,
        node_id: str,
        tensor_parallel_size: int = 1,
    ) -> None:
        try:
            resp = httpx.get(
                f"{RAY_DASHBOARD_URL}/api/v0/nodes?detail=True",
                timeout=5.0,
            )
            resp.raise_for_status()
            nodes = resp.json().get("data", {}).get("result", {}).get("result", [])
        except Exception:
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
                        f"Node {node_id[:16]}... has insufficient GPUs. "
                        f"Required: {tensor_parallel_size}, "
                        f"Available: {available_gpu:.0f}."
                    ),
                )

            return

    def _get_serve_route(self, deployment_id: str) -> str:
        name = f"vllm_{deployment_id.replace('-', '_')}"
        return f"{RAY_SERVE_URL}/{name}"

    def _get_blocking_deployments(self) -> List[InferenceDeployment]:
        return (
            self.db.query(InferenceDeployment)
            .filter(
                InferenceDeployment.status.in_(
                    [
                        StatusEnum.pending,
                        StatusEnum.active,
                        StatusEnum.cancelling,
                    ]
                )
            )
            .all()
        )

    def _ensure_activation_allowed(self) -> None:
        blocking = self._get_blocking_deployments()

        if not blocking:
            return

        raise HTTPException(
            status_code=409,
            detail={
                "message": (
                    "A local inference deployment is still active, pending, "
                    "or cancelling. Wait for teardown to complete before "
                    "activating another model."
                ),
                "deployments": [
                    {
                        "deployment_id": dep.id,
                        "status": (
                            dep.status.value
                            if hasattr(dep.status, "value")
                            else str(dep.status)
                        ),
                    }
                    for dep in blocking
                ],
            },
        )

    def _mark_deployments_cancelling(self, query) -> List[str]:
        deployments = query.filter(
            InferenceDeployment.status.in_(
                [
                    StatusEnum.pending,
                    StatusEnum.active,
                    StatusEnum.cancelling,
                ]
            )
        ).all()

        now = int(time.time())

        for deployment in deployments:
            deployment.status = StatusEnum.cancelling
            deployment.last_seen = now

        self.db.commit()

        return [deployment.id for deployment in deployments]

    def get_fine_tuned_model(
        self,
        model_id: str,
        user_id: Optional[str] = None,
    ) -> FineTunedModel:
        query = self.db.query(FineTunedModel).filter(
            FineTunedModel.id == model_id,
            FineTunedModel.deleted_at.is_(None),
        )

        if user_id:
            query = query.filter(FineTunedModel.user_id == user_id)

        model = query.first()

        if not model:
            raise HTTPException(
                status_code=404,
                detail="Fine-tuned model not found.",
            )

        return model

    def list_fine_tuned_models(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[FineTunedModel]:
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

    def soft_delete_model(
        self,
        model_id: str,
        user_id: Optional[str] = None,
    ) -> dict:
        model = self.get_fine_tuned_model(model_id, user_id)
        model.deleted_at = int(time.time())
        model.is_active = False
        self.db.commit()

        return {
            "status": "deleted",
            "model_id": model_id,
        }

    def list_deployments(self) -> List[InferenceDeployment]:
        return (
            self.db.query(InferenceDeployment)
            .order_by(InferenceDeployment.last_seen.desc())
            .all()
        )

    def deactivate_all_for_user(self, user_id: str) -> dict:
        self.db.query(FineTunedModel).filter(
            FineTunedModel.user_id == user_id,
            FineTunedModel.is_active,
        ).update(
            {"is_active": False},
            synchronize_session=False,
        )

        deployment_ids = self._mark_deployments_cancelling(
            self.db.query(InferenceDeployment)
        )

        if not deployment_ids:
            return {
                "status": "cancelled",
                "deployment_ids": [],
                "message": "No active local deployments require teardown.",
            }

        return {
            "status": "cancelling",
            "deployment_ids": deployment_ids,
            "message": (
                "Local deployments are being removed from Ray Serve. "
                "Project David will mark them cancelled only after runtime "
                "teardown and GPU release are confirmed."
            ),
        }

    def deactivate_model(
        self,
        model_id: str,
        user_id: Optional[str] = None,
    ) -> dict:
        model = self.get_fine_tuned_model(model_id, user_id)

        deployment_ids = self._mark_deployments_cancelling(
            self.db.query(InferenceDeployment).filter(
                InferenceDeployment.fine_tuned_model_id == model.id
            )
        )

        model.is_active = False
        self.db.commit()

        return {
            "status": "cancelling" if deployment_ids else "cancelled",
            "model_id": model.id,
            "deployment_ids": deployment_ids,
        }

    def deactivate_base_model(self, base_model_id: str) -> dict:
        base = self.registry.resolve(base_model_id)

        deployment_ids = self._mark_deployments_cancelling(
            self.db.query(InferenceDeployment).filter(
                InferenceDeployment.base_model_id == base.id,
                InferenceDeployment.fine_tuned_model_id.is_(None),
            )
        )

        return {
            "status": "cancelling" if deployment_ids else "cancelled",
            "base_model_id": base.id,
            "deployment_ids": deployment_ids,
        }

    def activate_base_model(
        self,
        base_model_id: str,
        target_node_id: Optional[str] = None,
        tensor_parallel_size: int = 1,
        # --- vLLM engine hyperparam overrides ---
        gpu_memory_utilization: Optional[float] = None,
        max_model_len: Optional[int] = None,
        max_num_seqs: Optional[int] = None,
        quantization: Optional[str] = None,
        dtype: Optional[str] = None,
        enforce_eager: Optional[bool] = None,
        limit_mm_per_prompt: Optional[Dict[str, int]] = None,
        mm_processor_kwargs: Optional[Dict[str, Any]] = None,
    ) -> dict:
        base = self.registry.resolve(base_model_id)

        self._ensure_activation_allowed()

        node_id = target_node_id or self._find_available_node(
            required_vram=4.0,
            tensor_parallel_size=tensor_parallel_size,
        )

        self._check_node_capacity(
            node_id,
            tensor_parallel_size=tensor_parallel_size,
        )

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
            # hyperparam overrides — None values fall back to reconciler defaults
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            quantization=quantization,
            dtype=dtype,
            enforce_eager=enforce_eager,
            limit_mm_per_prompt=limit_mm_per_prompt,
            mm_processor_kwargs=mm_processor_kwargs,
        )

        self.db.add(deployment)
        self.db.commit()

        logging_utility.info(
            "DeploymentService: base model activation scheduled — "
            "base_model_id=%s deployment_id=%s node=%s gpu_mem_util=%s "
            "max_model_len=%s quantization=%s limit_mm=%s mm_kwargs=%s",
            base.id,
            deployment_id,
            node_id,
            gpu_memory_utilization,
            max_model_len,
            quantization,
            limit_mm_per_prompt,
            mm_processor_kwargs,
        )

        return {
            "status": "deploying",
            "model_id": base.id,
            "hf_path": base.endpoint,
            "node": node_id,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "quantization": quantization,
            "dtype": dtype,
            "serve_route": serve_route,
            "next_step": (
                "InferenceReconciler will deploy via Ray Serve " "on next poll."
            ),
        }

    def activate_fine_tuned_model(
        self,
        model_id: str,
        target_node_id: Optional[str] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: Optional[float] = None,
        max_model_len: Optional[int] = None,
        max_num_seqs: Optional[int] = None,
        quantization: Optional[str] = None,
        dtype: Optional[str] = None,
        enforce_eager: Optional[bool] = None,
        limit_mm_per_prompt: Optional[Dict[str, int]] = None,
        mm_processor_kwargs: Optional[Dict[str, Any]] = None,
    ) -> dict:
        model = self.get_fine_tuned_model(model_id)

        self._ensure_activation_allowed()

        base = self.registry.resolve(model.base_model)

        node_id = target_node_id or self._find_available_node(
            required_vram=5.0,
            tensor_parallel_size=tensor_parallel_size,
        )

        self._check_node_capacity(
            node_id,
            tensor_parallel_size=tensor_parallel_size,
        )

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
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            quantization=quantization,
            dtype=dtype,
            enforce_eager=enforce_eager,
            limit_mm_per_prompt=limit_mm_per_prompt,
            mm_processor_kwargs=mm_processor_kwargs,
        )

        model.is_active = True
        self.db.add(deployment)
        self.db.commit()

        logging_utility.info(
            "DeploymentService: fine-tuned model activation scheduled — "
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
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "quantization": quantization,
            "dtype": dtype,
            "serve_route": serve_route,
            "next_step": (
                "InferenceReconciler will deploy via Ray Serve " "on next poll."
            ),
        }

    def update_deployment(
        self,
        deployment_id: str,
        update: "DeploymentUpdateRequest",
    ) -> dict:
        dep = (
            self.db.query(InferenceDeployment)
            .filter(InferenceDeployment.id == deployment_id)
            .first()
        )

        if not dep:
            raise HTTPException(
                status_code=404,
                detail="Deployment not found.",
            )

        data = update.model_dump(exclude_unset=True)

        for field, value in data.items():
            setattr(dep, field, value)

        dep.last_seen = int(time.time())
        self.db.commit()

        logging_utility.info(
            "DeploymentService: deployment %s updated — fields=%s",
            deployment_id,
            list(data.keys()),
        )

        return {
            "status": "updated",
            "deployment_id": deployment_id,
            "updated_fields": list(data.keys()),
        }
