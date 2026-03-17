# src/api/training/app.py


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from projectdavid_common import UtilsInterface

logging_utility = UtilsInterface.LoggingUtility()


def create_app() -> FastAPI:
    logging_utility.info("Creating Training Service FastAPI app")

    app = FastAPI(
        title="ProjectDavid — Training Service",
        description="API-driven fine-tuning pipeline: datasets, training jobs, and fine-tuned model registry.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---------------------------------------------------------------------------
    # Routers — uncomment as each is implemented
    # ---------------------------------------------------------------------------
    # from src.api.training.routers.datasets_router import router as datasets_router
    # from src.api.training.routers.training_jobs_router import router as training_jobs_router
    # from src.api.training.routers.fine_tuned_models_router import router as fine_tuned_models_router

    # app.include_router(datasets_router,          prefix="/v1/datasets",          tags=["Datasets"])
    # app.include_router(training_jobs_router,     prefix="/v1/training-jobs",     tags=["Training Jobs"])
    # app.include_router(fine_tuned_models_router, prefix="/v1/fine-tuned-models", tags=["Fine-Tuned Models"])

    @app.get("/", tags=["Health"])
    def read_root():
        return {"service": "training", "status": "online"}

    @app.get("/health", tags=["Health"])
    def health_check():
        return {"status": "ok"}

    logging_utility.info("Training Service ready.")
    return app


app = create_app()
