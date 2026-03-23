from fastapi import APIRouter

from src.api.training.routers.datasets_router import router as datasets_router
from src.api.training.routers.fine_tuned_models_router import router as fine_tuned_models_router
from src.api.training.routers.training_jobs_router import router as training_jobs_router

training_router = APIRouter()

training_router.include_router(datasets_router, prefix="/datasets", tags=["Datasets"])
training_router.include_router(
    training_jobs_router, prefix="/training-jobs", tags=["Training Jobs"]
)
training_router.include_router(
    fine_tuned_models_router, prefix="/fine-tuned-models", tags=["Fine-Tuned Models"]
)
