# src/api/training/routers/fine_tuned_models_router.py

from typing import Optional

from fastapi import APIRouter, Depends, Query
from projectdavid_common import ValidationInterface
from sqlalchemy.orm import Session

from src.api.training.db.database import get_db
from src.api.training.dependencies import get_current_user_id
from src.api.training.services.model_registry_service import ModelRegistryService

router = APIRouter()

# ──────────────────────────────────────────────────────────────────────────────
# REGISTRY MANAGEMENT (CRUD)
# ──────────────────────────────────────────────────────────────────────────────


@router.get("/", response_model=ValidationInterface.FineTunedModelList)
def list_models_endpoint(
    limit: int = 50,
    offset: int = 0,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """List all fine-tuned models belonging to the user."""
    service = ModelRegistryService(db)
    models = service.list_fine_tuned_models(user_id, limit, offset)
    return ValidationInterface.FineTunedModelList(
        data=[ValidationInterface.FineTunedModelRead.model_validate(m) for m in models],
        total=len(models),
    )


@router.get("/{model_id}", response_model=ValidationInterface.FineTunedModelRead)
def get_model_endpoint(
    model_id: str,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """Retrieve metadata for a specific fine-tuned model."""
    service = ModelRegistryService(db)
    model = service.get_fine_tuned_model(model_id, user_id)
    return ValidationInterface.FineTunedModelRead.model_validate(model)


@router.delete("/{model_id}", response_model=ValidationInterface.FineTunedModelDeleted)
def delete_model_endpoint(
    model_id: str,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """Soft-delete a model from the registry."""
    ModelRegistryService(db).soft_delete_model(model_id, user_id)
    return ValidationInterface.FineTunedModelDeleted(deleted=True, model_id=model_id)


# ──────────────────────────────────────────────────────────────────────────────
# FINE-TUNED MODEL LIFECYCLE (LoRA)
# ──────────────────────────────────────────────────────────────────────────────


@router.post(
    "/{model_id}/activate", response_model=ValidationInterface.ActivateModelResponse
)
def activate_model_endpoint(
    model_id: str,
    node_id: Optional[str] = None,
    tensor_parallel_size: int = Query(
        default=1,
        ge=1,
        description=(
            "Number of GPUs to shard this model across using vLLM tensor parallelism. "
            "1 = single GPU (default). N > 1 requires N GPUs available on the target node."
        ),
    ),
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """
    Promote a fine-tuned model to 'Active' status.
    Triggers the DeploymentSupervisor to provision vLLM with the LoRA adapters.

    Use tensor_parallel_size > 1 to shard the model across multiple GPUs
    for larger models that exceed single-GPU VRAM.
    """
    service = ModelRegistryService(db)
    return service.activate_model(
        model_id=model_id,
        user_id=user_id,
        target_node_id=node_id,
        tensor_parallel_size=tensor_parallel_size,
    )


@router.post("/{model_id}/deactivate")
def deactivate_model_endpoint(
    model_id: str,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """
    Surgically shutdown a specific fine-tuned model deployment.
    Releases VRAM on the associated node.
    """
    service = ModelRegistryService(db)
    return service.deactivate_model(model_id, user_id)


# ──────────────────────────────────────────────────────────────────────────────
# BASE MODEL LIFECYCLE (Factory Models)
# ──────────────────────────────────────────────────────────────────────────────


@router.post("/base/{base_model_id:path}/activate")
def activate_base_model_endpoint(
    base_model_id: str,
    node_id: Optional[str] = None,
    tensor_parallel_size: int = Query(
        default=1,
        ge=1,
        description=(
            "Number of GPUs to shard this model across using vLLM tensor parallelism. "
            "1 = single GPU (default). N > 1 requires N GPUs available on the target node."
        ),
    ),
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """
    Deploy a standard backbone model from the catalog (no LoRA).
    Schedules the model to the healthiest available GPU node.

    Use tensor_parallel_size > 1 to shard the model across multiple GPUs
    for larger models that exceed single-GPU VRAM.
    """
    service = ModelRegistryService(db)
    return service.activate_base_model(
        base_model_id=base_model_id,
        user_id=user_id,
        target_node_id=node_id,
        tensor_parallel_size=tensor_parallel_size,
    )


@router.post("/base/{base_model_id}/deactivate")
def deactivate_base_model_endpoint(
    base_model_id: str,
    db: Session = Depends(get_db),
):
    """Shutdown a specific standard backbone deployment."""
    service = ModelRegistryService(db)
    return service.deactivate_base_model(base_model_id)


# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL CLUSTER RESET
# ──────────────────────────────────────────────────────────────────────────────


@router.post("/deactivate-all")
def deactivate_all_endpoint(
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """
    Emergency Stop: Deactivates ALL deployments for the user.
    Reverts the cluster to an idle/clean state.
    """
    service = ModelRegistryService(db)
    return service.deactivate_all_models(user_id)
