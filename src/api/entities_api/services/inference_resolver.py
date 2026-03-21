import os

from sqlalchemy.orm import Session

from src.api.training.models.models import (ComputeNode, InferenceDeployment,
                                            StatusEnum)


class InferenceResolver:
    @staticmethod
    def resolve_vllm_url(db: Session, model_tag: str) -> str:
        """
        STAGE 6: Dynamic Mesh Resolution.
        Finds the physical URL (IP:Port) for a requested model.
        """
        # 1. Strip 'vllm/' prefix if present in the tag
        search_id = model_tag.replace("vllm/", "")

        # 2. Query the Deployment Ledger
        # We join with ComputeNode to ensure we only route to ONLINE hardware
        deployment = (
            db.query(InferenceDeployment)
            .join(ComputeNode)
            .filter(
                InferenceDeployment.status == StatusEnum.active,
                ComputeNode.status == StatusEnum.active,
                # Resolve by either Base Model ID or Fine-Tuned ID
                (InferenceDeployment.fine_tuned_model_id == search_id)
                | (InferenceDeployment.base_model_id == search_id),
            )
            # Simple Load Balancing: Route to the node with the least throughput
            .order_by(InferenceDeployment.current_throughput.asc())
            .first()
        )

        if deployment:
            host = deployment.node.ip_address or deployment.node.hostname
            target = f"http://{host}:{deployment.port}"
            return target

        # 3. Fallback: If no deployment found in Mesh, use legacy ENV default
        fallback = os.getenv("VLLM_BASE_URL", "http://vllm_server:8000")
        return fallback
