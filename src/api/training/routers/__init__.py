from fastapi import APIRouter

from src.api.training.routers.datasets_router import router as datasets_router
from src.api.training.routers.training_jobs_router import \
    router as training_jobs_router

# The central aggregator
training_router = APIRouter()

# Include sub-routers.
# Note: prefixes here combine with the /v1 in app.py
training_router.include_router(datasets_router, prefix="/datasets", tags=["Datasets"])
training_router.include_router(
    training_jobs_router, prefix="/training-jobs", tags=["Training Jobs"]
)

# Add this later when ready:
# from src.api.training.routers.fine_tuned_models_router import router as models_router
# training_router.include_router(models_router, prefix="/fine-tuned-models", tags=["Models"])
