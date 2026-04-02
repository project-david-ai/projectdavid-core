# src/api/entities_api/services/inference_resolver.py

import os
from typing import Optional

from projectdavid_common import UtilsInterface
from projectdavid_common.schemas.enums import StatusEnum
from projectdavid_orm.projectdavid_orm.models import InferenceDeployment
from sqlalchemy.orm import Session

logging_utility = UtilsInterface.LoggingUtility()

# Prefix written by InferenceReconciler when creating Ray Serve deployments.
# Used to distinguish Sovereign Forge deployments from legacy IP-based ones.
_SOVEREIGN_FORGE_DEP_PREFIX = "vllm_dep_"


class InferenceResolver:
    """
    STAGE 6: Global Mesh Resolver.

    Resolves a model identifier to a live vLLM endpoint URL.

    Supports two deployment architectures:

    Legacy (DeploymentSupervisor):
        internal_hostname = bare IP, e.g. "172.18.0.14"
        URL constructed as: http://{ip}:8000

    Sovereign Forge (Ray Serve / InferenceReconciler):
        internal_hostname = full URL, e.g. "http://inference_worker:8000/vllm_dep_{id}"
        URL returned as-is — no wrapping needed.

    Model lookup supports three identifier formats:
        ftm_...      fine_tuned_model_id column
        bm_...       base_model_id column (HF path stored here)
        vllm_dep_... InferenceDeployment.id (primary key)
    """

    @staticmethod
    def resolve_vllm_url(db: Session, model_tag: str) -> Optional[str]:
        """
        Find the active inference endpoint for a requested model tag.

        Args:
            model_tag: One of:
                - "ftm_..."         fine-tuned model ID
                - "bm_..." or HF path   base model
                - "vllm_dep_..."    deployment primary key (Sovereign Forge)

        Returns:
            Full URL string ready to use as base_url in VLLMRawStream,
            or None if no active deployment is found.
        """
        # Strip provider prefix if present (e.g. "vllm/vllm_dep_..." → "vllm_dep_...")
        target_id = model_tag.replace("vllm/", "")

        deployment = (
            db.query(InferenceDeployment)
            .filter(
                InferenceDeployment.status == StatusEnum.active,
                (InferenceDeployment.fine_tuned_model_id == target_id)
                | (InferenceDeployment.base_model_id == target_id)
                # Sovereign Forge: target_id is the deployment primary key
                | (InferenceDeployment.id == target_id),
            )
            .order_by(InferenceDeployment.last_seen.desc())
            .first()
        )

        if not deployment:
            logging_utility.warning(
                "Resolver: no active deployment found for model '%s'", target_id
            )
            return None

        if not deployment.internal_hostname:
            # Deployment exists but reconciler has not written the hostname yet.
            # Transient state — reconciler runs every 20s.
            logging_utility.warning(
                "Resolver: deployment '%s' found for model '%s' but "
                "internal_hostname not yet populated — reconciler pending.",
                deployment.id,
                target_id,
            )
            return None

        hostname = deployment.internal_hostname

        # Sovereign Forge / Ray Serve: internal_hostname is a complete URL.
        # Return it directly — do not wrap with http://{...}:8000.
        if hostname.startswith(("http://", "https://")):
            logging_utility.info(
                "Resolver: '%s' → %s (Ray Serve / Sovereign Forge, node=%s...)",
                target_id,
                hostname,
                deployment.node_id[:16] if deployment.node_id else "unknown",
            )
            return hostname

        # Legacy / DeploymentSupervisor: internal_hostname is a bare IP.
        # Wrap with scheme and vLLM default port.
        legacy_url = f"http://{hostname}:8000"
        logging_utility.info(
            "Resolver: '%s' → %s (legacy IP routing, node=%s...)",
            target_id,
            legacy_url,
            deployment.node_id[:16] if deployment.node_id else "unknown",
        )
        return legacy_url
