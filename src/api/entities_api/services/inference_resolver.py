import os
from typing import Optional

from projectdavid_common import UtilsInterface
from projectdavid_common.schemas.enums import StatusEnum
from projectdavid_orm.projectdavid_orm.models import InferenceDeployment
from sqlalchemy.orm import Session

logging_utility = UtilsInterface.LoggingUtility()


class InferenceResolver:
    """
    STAGE 6: Global Mesh Resolver.

    Phase 4 update: ComputeNode JOIN removed. inference_deployments.node_id
    now stores a Ray node ID (hex string) rather than a compute_nodes FK,
    so the JOIN would always return zero rows. Node health is now verified
    by the Ray cluster — the presence of an active InferenceDeployment row
    with a populated internal_hostname is sufficient to route the request.
    """

    @staticmethod
    def resolve_vllm_url(db: Session, model_tag: str) -> Optional[str]:
        """
        Finds the active endpoint for a requested model.

        Routing:
          internal_hostname (dotted-quad IP written by DeploymentSupervisor)
          → http://<internal_hostname>:8000
          This is always a Docker-internal address — port 8000 is the
          container's native vLLM port, not the host-mapped 8001.

        Falls back to None with a warning if no active deployment is found
        or if internal_hostname has not yet been populated by the supervisor.
        """
        # 1. Normalise model tag — strip vllm/ prefix if SDK provided one
        target_id = model_tag.replace("vllm/", "")

        # 2. Query inference_deployments directly — no ComputeNode JOIN.
        #    Order by last_seen descending so the most recently confirmed
        #    active deployment is preferred when multiple rows exist.
        deployment = (
            db.query(InferenceDeployment)
            .filter(
                InferenceDeployment.status == StatusEnum.active,
                (InferenceDeployment.fine_tuned_model_id == target_id)
                | (InferenceDeployment.base_model_id == target_id),
            )
            .order_by(InferenceDeployment.last_seen.desc())
            .first()
        )

        if not deployment:
            logging_utility.warning(
                f"🌐 Resolver: No active deployment found for model '{target_id}'"
            )
            return None

        # 3. Route via internal container IP written by DeploymentSupervisor.
        #    This is the only routing path in Phase 4 — the node relationship
        #    and fallback to node.ip_address are removed because node_id is
        #    no longer a FK to compute_nodes.
        if deployment.internal_hostname:
            internal_url = f"http://{deployment.internal_hostname}:8000"
            logging_utility.info(
                f"🌐 Resolver: Resolved '{target_id}' → {internal_url} "
                f"(node: {deployment.node_id[:16]}...)"
            )
            return internal_url

        # 4. Deployment exists but supervisor hasn't written the IP yet.
        #    This is a transient state — the supervisor reconciles every 20s.
        logging_utility.warning(
            f"🌐 Resolver: Deployment found for '{target_id}' but "
            f"internal_hostname not yet populated — supervisor reconciling."
        )
        return None
