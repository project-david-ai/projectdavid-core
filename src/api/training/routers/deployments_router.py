# src/api/training/routers/deployments_router.py
"""
deployments_router.py

REST API endpoints for inference deployment lifecycle management.

Prefix:  /v1/deployments
Tags:    deployments

All activation and deactivation operations are admin-scoped.
Listing deployments is available to any authenticated user.

Endpoints:
  POST   /v1/deployments/base                     Activate a base model          [admin]
  POST   /v1/deployments/fine-tuned               Activate a fine-tuned model    [admin]
  GET    /v1/deployments                          List all active deployments
  DELETE /v1/deployments/base/{model_ref}         Deactivate a base model        [admin]
  DELETE /v1/deployments/fine-tuned/{model_id}    Deactivate a fine-tuned model  [admin]
  DELETE /v1/deployments                          Deactivate all deployments      [admin]

Architecture note:
  Activation endpoints create an InferenceDeployment record with status=pending.
  The InferenceReconciler (inference_worker.py) picks it up on its next poll
  cycle and deploys the corresponding Ray Serve application.
  These endpoints do NOT communicate with Ray Serve directly.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from projectdavid_common.schemas.deployment_schemas import (
    ActivateBaseModelRequest,
    ActivateFineTunedModelRequest,
    DeactivateAllResponse,
    DeploymentActivationResponse,
    DeploymentDeactivationResponse,
    DeploymentListResponse,
)
from projectdavid_common.utilities.check_admin_status import _is_admin
from sqlalchemy.orm import Session

from src.api.training.db.database import get_db
from src.api.training.dependencies import get_current_user_id
from src.api.training.services.deployment_service import DeploymentService

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_admin(current_user_id: str, db: Session) -> None:
    if not _is_admin(current_user_id, db):
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required for this operation.",
        )


# ---------------------------------------------------------------------------
# Activation                                                     [admin only]
# ---------------------------------------------------------------------------


@router.post(
    "/base",
    response_model=DeploymentActivationResponse,
    summary="Activate a base model",
    description=(
        "Schedule a base model (no LoRA adapter) for inference deployment. "
        "Accepts either a `bm_...` prefixed ID or a raw HuggingFace model path. "
        "Clears all existing deployments before creating the new one. "
        "The InferenceReconciler deploys via Ray Serve on its next poll cycle. "
        "**Admin only.**"
    ),
    status_code=202,
)
def activate_base_model(
    payload: ActivateBaseModelRequest,
    db: Session = Depends(get_db),
    current_user_id: str = Depends(get_current_user_id),
) -> DeploymentActivationResponse:
    _require_admin(current_user_id, db)
    service = DeploymentService(db)
    result = service.activate_base_model(
        base_model_id=payload.base_model_id,
        target_node_id=payload.target_node_id,
        tensor_parallel_size=payload.tensor_parallel_size,
    )
    return DeploymentActivationResponse(**result)


@router.post(
    "/fine-tuned",
    response_model=DeploymentActivationResponse,
    summary="Activate a fine-tuned model",
    description=(
        "Schedule a fine-tuned model (base + LoRA adapter) for inference deployment. "
        "Deactivates all existing deployments for the model owner before creating the new one. "
        "The InferenceReconciler deploys via Ray Serve on its next poll cycle. "
        "**Admin only.**"
    ),
    status_code=202,
)
def activate_fine_tuned_model(
    payload: ActivateFineTunedModelRequest,
    db: Session = Depends(get_db),
    current_user_id: str = Depends(get_current_user_id),
) -> DeploymentActivationResponse:
    _require_admin(current_user_id, db)
    service = DeploymentService(db)
    result = service.activate_fine_tuned_model(
        model_id=payload.model_id,
        target_node_id=payload.target_node_id,
        tensor_parallel_size=payload.tensor_parallel_size,
    )
    return DeploymentActivationResponse(**result)


# ---------------------------------------------------------------------------
# Listing                                                        [any user]
# ---------------------------------------------------------------------------


@router.get(
    "/",
    response_model=DeploymentListResponse,
    summary="List active deployments",
    description=(
        "Return all InferenceDeployment records currently tracked by the system. "
        "Includes status, node assignment, base model, and serve route for each deployment."
    ),
)
def list_deployments(
    db: Session = Depends(get_db),
    current_user_id: str = Depends(get_current_user_id),
) -> DeploymentListResponse:
    service = DeploymentService(db)
    deployments = service.list_deployments()
    return DeploymentListResponse(
        items=deployments,
        total=len(deployments),
    )


# ---------------------------------------------------------------------------
# Deactivation                                                   [admin only]
# ---------------------------------------------------------------------------


@router.delete(
    "/base/{model_ref:path}",
    response_model=DeploymentDeactivationResponse,
    summary="Deactivate a base model deployment",
    description=(
        "Surgically remove the InferenceDeployment record for a base model. "
        "Accepts either a `bm_...` prefixed ID or a raw HuggingFace model path. "
        "The InferenceReconciler tears down the Ray Serve application on its next poll. "
        "**Admin only.**"
    ),
)
def deactivate_base_model(
    model_ref: str,
    db: Session = Depends(get_db),
    current_user_id: str = Depends(get_current_user_id),
) -> DeploymentDeactivationResponse:
    _require_admin(current_user_id, db)
    service = DeploymentService(db)
    result = service.deactivate_base_model(model_ref)
    return DeploymentDeactivationResponse(**result)


@router.delete(
    "/fine-tuned/{model_id}",
    response_model=DeploymentDeactivationResponse,
    summary="Deactivate a fine-tuned model deployment",
    description=(
        "Surgically remove the InferenceDeployment record for a fine-tuned model. "
        "The InferenceReconciler tears down the Ray Serve application on its next poll. "
        "**Admin only.**"
    ),
)
def deactivate_fine_tuned_model(
    model_id: str,
    db: Session = Depends(get_db),
    current_user_id: str = Depends(get_current_user_id),
) -> DeploymentDeactivationResponse:
    _require_admin(current_user_id, db)
    service = DeploymentService(db)
    result = service.deactivate_model(model_id)
    return DeploymentDeactivationResponse(**result)


@router.delete(
    "/",
    response_model=DeactivateAllResponse,
    summary="Deactivate all deployments",
    description=(
        "Remove all InferenceDeployment records — full cluster clean slate. "
        "The InferenceReconciler tears down all Ray Serve applications on its next poll, "
        "releasing all GPU reservations back to the cluster. "
        "**Admin only.**"
    ),
)
def deactivate_all(
    db: Session = Depends(get_db),
    current_user_id: str = Depends(get_current_user_id),
) -> DeactivateAllResponse:
    _require_admin(current_user_id, db)
    service = DeploymentService(db)
    service._clear_all_deployments()
    return DeactivateAllResponse(
        status="success",
        message="All deployments cleared. InferenceReconciler will release GPU resources on next poll.",
    )
