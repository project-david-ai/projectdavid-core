# src/api/entities_api/services/inference_resolver.py

import os
import posixpath
import re
from typing import Optional

from projectdavid_common import UtilsInterface
from projectdavid_common.schemas.enums import StatusEnum
from projectdavid_orm.projectdavid_orm.models import BaseModel, InferenceDeployment
from sqlalchemy.orm import Session

logging_utility = UtilsInterface.LoggingUtility()

# Prefix written by InferenceReconciler when creating Ray Serve deployments.
_SOVEREIGN_FORGE_DEP_PREFIX = "vllm_dep_"


_MODEL_HUB_DEFAULT_ROOT = "/opt/projectdavid/model-hub/models"
_MODEL_HUB_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9._-]*", re.ASCII)
_MODEL_HUB_REVISION_KEY = re.compile(r"[0-9a-f]{24}", re.ASCII)


def _selector_target(model_tag: str) -> str:
    return model_tag.removeprefix("vllm/").strip()


def _model_hub_root() -> str:
    root = os.getenv("MODEL_HUB_RUNTIME_ROOT", _MODEL_HUB_DEFAULT_ROOT).strip()
    if (
        not root.startswith("/")
        or root.startswith("//")
        or root == "/"
        or "\\" in root
        or "\x00" in root
        or posixpath.normpath(root) != root
    ):
        raise ValueError(
            "MODEL_HUB_RUNTIME_ROOT must be a canonical non-root POSIX path"
        )
    return root


def _registered_variant(endpoint: str, root: str) -> Optional[str]:
    """Decode Q's validated runtime locator contract, never an arbitrary suffix.

    Q modelRuntimeLocator.js requires modelId/variantId/sha256(revision)[:24].
    Only endpoints already in the registry beneath the configured root qualify.
    """
    if not isinstance(endpoint, str) or not endpoint.startswith(root + "/"):
        return None
    parts = endpoint[len(root) + 1 :].split("/")
    if (
        len(parts) != 3
        or not all(_MODEL_HUB_IDENTIFIER.fullmatch(p) for p in parts[:2])
        or not _MODEL_HUB_REVISION_KEY.fullmatch(parts[2])
    ):
        return None
    return parts[1]


class InferenceResolver:
    """
    STAGE 6: Global Mesh Resolver.

    Resolves a model identifier to a live vLLM endpoint URL.

    Supports the existing identity lookups plus registered Model Hub variant IDs:

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

    5. Model Hub variant ID (e.g. "qwen3-8b-awq-4bit")
       Decoded from Q's canonical registered projection endpoint. Exactly one
       active base deployment must match; ambiguous variants fail closed.

    Deployment architectures supported:

    Legacy (DeploymentSupervisor):
        internal_hostname = bare IP, e.g. "172.18.0.14"
        URL constructed as: http://{ip}:8000

    Sovereign Forge (Ray Serve / InferenceReconciler):
        internal_hostname = full URL, e.g. "http://inference_worker:8000/vllm_dep_{id}"
        URL returned as-is.
    """

    @staticmethod
    def requires_registered_route(model_tag: str) -> bool:
        """Opaque IDs must resolve in the ledger, never a generic server fallback."""
        return bool(
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", _selector_target(model_tag))
        )

    @staticmethod
    def resolve_vllm_route(db: Session, model_tag: str) -> Optional[dict]:
        """Resolve a live endpoint together with its deployment context capacity.

        ``resolve_vllm_url`` remains the canonical routing decision. This helper
        deliberately derives capacity *after* that decision so existing exact,
        deployment-ID, Hugging Face, and Model Hub routing semantics stay
        unchanged.
        """
        url = InferenceResolver.resolve_vllm_url(db, model_tag)
        if not url:
            return None

        deployments = (
            db.query(InferenceDeployment)
            .filter(InferenceDeployment.status == StatusEnum.active)
            .order_by(InferenceDeployment.last_seen.desc())
            .all()
        )

        for deployment in deployments:
            hostname = deployment.internal_hostname
            if not hostname:
                continue
            candidate_url = (
                hostname
                if hostname.startswith(("http://", "https://"))
                else f"http://{hostname}:8000"
            )
            if candidate_url != url:
                continue

            raw_limit = getattr(deployment, "max_model_len", None)
            try:
                max_model_len = int(raw_limit) if raw_limit is not None else None
            except (TypeError, ValueError):
                max_model_len = None
            if max_model_len is not None and max_model_len <= 0:
                max_model_len = None

            return {
                "url": url,
                "deployment_id": deployment.id,
                "max_model_len": max_model_len,
            }

        # Legacy/custom ledgers may not expose a capacity column. Routing still
        # succeeds; callers simply retain their historical context policy.
        return {
            "url": url,
            "deployment_id": None,
            "max_model_len": None,
        }

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
        target = _selector_target(model_tag)

        # Normalise deployment PK: "vllm_dep_XYZ" → "dep_XYZ"
        dep_pk = (
            "dep_" + target[len("vllm_dep_") :]
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

        if not deployment and InferenceResolver.requires_registered_route(model_tag):
            root = _model_hub_root()
            candidates = (
                db.query(InferenceDeployment, BaseModel.endpoint)
                .join(BaseModel, BaseModel.id == InferenceDeployment.base_model_id)
                .filter(
                    InferenceDeployment.status == StatusEnum.active,
                    InferenceDeployment.fine_tuned_model_id.is_(None),
                    BaseModel.endpoint.startswith(root + "/", autoescape=True),
                )
                .all()
            )
            matches = [
                dep
                for dep, endpoint in candidates
                if _registered_variant(endpoint, root) == target
            ]
            if len(matches) > 1:
                raise ValueError(
                    "MODEL_HUB_ROUTE_AMBIGUOUS: multiple active base deployments "
                    "match the requested variant; select an exact deployment ID"
                )
            if matches:
                deployment = matches[0]

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
