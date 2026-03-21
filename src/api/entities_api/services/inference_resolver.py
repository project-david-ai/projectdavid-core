import os

from projectdavid_common.schemas.enums import StatusEnum
from projectdavid_orm.projectdavid_orm.models import (ComputeNode,
                                                      InferenceDeployment)
from sqlalchemy.orm import Session


class InferenceResolver:
    @staticmethod
    def resolve_vllm_url(db: Session, model_tag: str) -> str | None:
        """
        STAGE 6: Global Mesh Resolver.
        Finds the healthiest physical endpoint for a specific model ID.
        """
        # 1. Standardize the ID (Strip vllm/ prefix)
        target_id = model_tag.replace("vllm/", "")

        # 2. Find Active Deployments on Active Nodes
        # We sort by (Total VRAM - Usage) descending to pick the least-loaded node.
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
            return None

        # 3. Construct physical URL
        host = deployment.node.ip_address or deployment.node.hostname
        return f"http://{host}:{deployment.port}"
