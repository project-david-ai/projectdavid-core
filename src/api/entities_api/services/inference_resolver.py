# src/api/entities_api/services/inference_resolver.py

import os
from typing import Optional

from projectdavid_common import UtilsInterface
from projectdavid_common.schemas.enums import StatusEnum
from projectdavid_orm.projectdavid_orm.models import BaseModel, InferenceDeployment
from sqlalchemy.orm import Session

logging_utility = UtilsInterface.LoggingUtility()

# Prefix written by InferenceReconciler when creating Ray Serve deployments.
_SOVEREIGN_FORGE_DEP_PREFIX = "vllm_dep_"


class InferenceResolver:
    """
    STAGE 6: Global Mesh Resolver.

    Resolves a model identifier to a live vLLM endpoint URL.

    Supports four lookup strategies (in priority order):

    1. Fine-tuned model ID (ftm_...)
       Matches InferenceDeployment.fine_tuned_model_id directly.

    2. Deployment primary key (dep_... / vllm_dep_...)
       Matches InferenceDeployment.id directly.
       Used by fine-tuned model callers who know the deployment ID:
           MODEL_ID = "vllm/vllm_dep_XBY3Xnx3rBiUGG89iJIaFw"

    3. Base model ID (bm_...)
       Matches InferenceDeployment.base_model_id directly.

    4. HF path (e.g. "unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit")
       Implicit base model call. Joins BaseModel table and matches
       BaseModel.endpoint — the HF path stored at registration time.
           MODEL_ID = "vllm/unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit"

    Deployment architectures supported:

    Legacy (DeploymentSupervisor):
        internal_hostname = bare IP, e.g. "172.18.0.14"
        URL constructed as: http://{ip}:8000

    Sovereign Forge (Ray Serve / InferenceReconciler):
        internal_hostname = full URL, e.g. "http://inference_worker:8000/vllm_dep_{id}"
        URL returned as-is.
    """

    @staticmethod
    def resolve_vllm_url(db: Session, model_tag: str) -> Optional[str]:
        """
        Find the active inference endpoint for a requested model tag.

        Args:
            model_tag: One of:
                - "vllm/unsloth/..."        HF path — implicit base model call
                - "vllm/vllm_dep_..."       Deployment PK — fine-tuned or base
                - "vllm/ftm_..."            Fine-tuned model ID
                - "vllm/bm_..."             Base model ID

        Returns:
            Full URL string ready to use as base_url in VLLMRawStream,
            or None if no active deployment is found.
        """
        # Strip provider prefix
        target = model_tag.replace("vllm/", "").strip()

        # Normalise deployment PK: "vllm_dep_XYZ" → "dep_XYZ"
        dep_pk = (
            target.replace("vllm_dep_", "dep_")
            if target.startswith("vllm_dep_")
            else target
        )

        logging_utility.info(
            "Resolver: looking up '%s' (normalised: '%s')", model_tag, target
        )

        deployment = (
            db.query(InferenceDeployment)
            .outerjoin(BaseModel, BaseModel.id == InferenceDeployment.base_model_id)
            .filter(
                InferenceDeployment.status == StatusEnum.active,
                (
                    # Strategy 1: fine-tuned model ID
                    (InferenceDeployment.fine_tuned_model_id == target)
                    # Strategy 2: deployment primary key (normalised)
                    | (InferenceDeployment.id == dep_pk)
                    # Strategy 3: base model ID (bm_...)
                    | (InferenceDeployment.base_model_id == target)
                    # Strategy 4: HF path via BaseModel.endpoint join
                    | (BaseModel.endpoint == target)
                ),
            )
            .order_by(InferenceDeployment.last_seen.desc())
            .first()
        )

        if not deployment:
            logging_utility.warning(
                "Resolver: no active deployment found for model '%s'", target
            )
            return None

        if not deployment.internal_hostname:
            logging_utility.warning(
                "Resolver: deployment '%s' found for model '%s' but "
                "internal_hostname not yet populated — reconciler pending.",
                deployment.id,
                target,
            )
            return None

        hostname = deployment.internal_hostname

        # Sovereign Forge / Ray Serve: internal_hostname is a complete URL.
        if hostname.startswith(("http://", "https://")):
            logging_utility.info(
                "Resolver: '%s' → %s (Ray Serve / Sovereign Forge, node=%s...)",
                target,
                hostname,
                deployment.node_id[:16] if deployment.node_id else "unknown",
            )
            return hostname

        # Legacy / DeploymentSupervisor: bare IP.
        legacy_url = f"http://{hostname}:8000"
        logging_utility.info(
            "Resolver: '%s' → %s (legacy IP routing, node=%s...)",
            target,
            legacy_url,
            deployment.node_id[:16] if deployment.node_id else "unknown",
        )
        return legacy_url
