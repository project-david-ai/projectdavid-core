from typing import Optional

from fastapi import APIRouter, Depends
from projectdavid_common import ValidationInterface
from sqlalchemy.orm import Session

from src.api.training.db.database import get_db
from src.api.training.dependencies import get_current_user_id
from src.api.training.services import model_registry_service

router = APIRouter()


@router.get("/", response_model=ValidationInterface.FineTunedModelList)
def list_models_endpoint(
    limit: int = 50,
    offset: int = 0,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    models = model_registry_service.list_fine_tuned_models(db, user_id, limit, offset)
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
    model = model_registry_service.get_fine_tuned_model(db, model_id, user_id)
    return ValidationInterface.FineTunedModelRead.model_validate(model)


@router.delete("/{model_id}", response_model=ValidationInterface.FineTunedModelDeleted)
def delete_model_endpoint(
    model_id: str,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    model_registry_service.soft_delete_model(db, model_id, user_id)
    return ValidationInterface.FineTunedModelDeleted(deleted=True, model_id=model_id)


@router.post("/{model_id}/activate", response_model=ValidationInterface.ActivateModelResponse)
def activate_model_endpoint(
    model_id: str,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    return model_registry_service.activate_model(db, model_id, user_id)


@router.post("/deactivate-all")
def deactivate_all_endpoint(
    user_id: str = Depends(get_current_user_id), db: Session = Depends(get_db)
):
    return model_registry_service.deactivate_all_models(db, user_id)
