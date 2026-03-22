import logging
import os
from typing import Optional

from projectdavid_common import UtilsInterface
from projectdavid_common.schemas.enums import StatusEnum
from projectdavid_orm.projectdavid_orm.models import (ComputeNode,
                                                      InferenceDeployment)
from sqlalchemy.orm import Session

# Initialize logging for Stage 6 observability
logging_utility = UtilsInterface.LoggingUtility()


class InferenceResolver:
    """
    STAGE 6: Global Mesh Resolver.
    Determines the physical or internal endpoint for a specific model ID
    by interrogating the Mesh Ledger.
    """

    @staticmethod
    def resolve_vllm_url(db: Session, model_tag: str) -> Optional[str]:
        """
        Finds the healthiest endpoint for a requested model.

        Routing Priority:
          1. Internal Container Hostname: Stays inside the Docker Bridge (Port 8000).
          2. Physical Node IP: Facilitates cross-host routing (Port 8001).
        """
        # 1. Standardize the ID (Strip vllm/ prefix if the SDK provided one)
        target_id = model_tag.replace("vllm/", "")

        # 2. Query the Mesh Ledger
        # Join InferenceDeployment with ComputeNode to verify the hardware is online.
        # Order by free VRAM (Total - Usage) descending to provide automatic load balancing.
        deployment = (
            db.query(InferenceDeployment)
            .join(ComputeNode, InferenceDeployment.node_id == ComputeNode.id)
            .filter(
                InferenceDeployment.status == StatusEnum.active,
                ComputeNode.status == StatusEnum.active,
                (InferenceDeployment.fine_tuned_model_id == target_id)
                | (InferenceDeployment.base_model_id == target_id),
            )
            .order_by((ComputeNode.total_vram_gb - ComputeNode.current_vram_usage_gb).desc())
            .first()
        )

        if not deployment:
            logging_utility.warning(
                f"🌐 Resolver: No active deployment found for model '{target_id}'"
            )
            return None

        # 3. 🎯 STAGE 6: MESH ROUTING
        # If the worker has recorded an internal container name, use it.
        # This is the 'Gold Standard' for Docker-to-Docker communication.
        if hasattr(deployment, "internal_hostname") and deployment.internal_hostname:
            internal_url = f"http://{deployment.internal_hostname}:8000"
            logging_utility.info(
                f"🌐 Resolver: Resolved '{target_id}' to internal host -> {internal_url}"
            )
            return internal_url

        # 4. Fallback: Physical Node IP + External Mapped Port (e.g. 8001)
        # Used when the Core API and vLLM are on separate physical hosts.
        host = deployment.node.ip_address or deployment.node.hostname
        external_url = f"http://{host}:{deployment.port}"

        logging_utility.info(
            f"🌐 Resolver: Resolved '{target_id}' to physical node -> {external_url}"
        )
        return external_url
