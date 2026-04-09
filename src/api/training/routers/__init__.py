from fastapi import APIRouter

from src.api.training.routers.datasets_router import router as datasets_router
from src.api.training.routers.deployments_router import router as deployments_router
from src.api.training.routers.fine_tuned_models_router import (
    router as fine_tuned_models_router,
)
from src.api.training.routers.registry_router import router as registry_router
from src.api.training.routers.training_jobs_router import router as training_jobs_router

training_router = APIRouter()

# ── New routers ──────────────────────────────────────────────────────────────
training_router.include_router(
    deployments_router, prefix="/deployments", tags=["deployments"]
)
training_router.include_router(registry_router, prefix="/registry", tags=["registry"])

# ── Legacy routers (deprecated — will be removed in v3.0.0) ──────────────────
# Migrate to /v1/deployments/* and /v1/registry/* equivalents.
training_router.include_router(datasets_router, prefix="/datasets", tags=["Datasets"])
training_router.include_router(
    training_jobs_router, prefix="/training-jobs", tags=["Training Jobs"]
)
training_router.include_router(
    fine_tuned_models_router,
    prefix="/fine-tuned-models",
    tags=["Fine-Tuned Models"],
    deprecated=True,
)
