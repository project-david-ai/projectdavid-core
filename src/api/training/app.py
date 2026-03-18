# src/api/training/app.py
#
# Training Service — FastAPI application entry point.
#
# Architecture:
#   - Direct DB access via its own SQLAlchemy engine (shared MySQL instance).
#   - No local auth logic — JWT ticket method (wired when routers are added).
#   - No observability yet — circle back once the pipeline is functional.
#
# Run locally:
#   uvicorn src.api.training.app:app --host 0.0.0.0 --port 9001 --reload

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from projectdavid_common import UtilsInterface

from src.api.training.constants.banner import BANNER
from src.api.training.db.database import wait_for_db

logging_utility = UtilsInterface.LoggingUtility()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Print the neon green graphic right as the server spins up
    print(BANNER)
    logging_utility.info("Training Service ready.")
    yield
    # Any future cleanup/shutdown logic goes here
    logging_utility.info("Training Service shutting down.")


# Initialize DB Connection
wait_for_db()


def create_app() -> FastAPI:
    logging_utility.info("Creating Training Service FastAPI app")

    app = FastAPI(
        title="ProjectDavid — Training Service",
        description="API-driven fine-tuning pipeline: datasets, training jobs, and fine-tuned model registry.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,  # <--- Injected lifespan manager here
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---------------------------------------------------------------------------
    # Routers — Active
    # ---------------------------------------------------------------------------
    from src.api.training.routers.datasets_router import \
        router as datasets_router

    # from src.api.training.routers.training_jobs_router import router as training_jobs_router
    # from src.api.training.routers.fine_tuned_models_router import router as fine_tuned_models_router
    # This injects the /v1/datasets prefix.
    # So if your router has `@router.post("")`, it becomes `POST /v1/datasets`
    app.include_router(datasets_router, prefix="/v1/datasets", tags=["Datasets"])

    # app.include_router(training_jobs_router,     prefix="/v1/training-jobs",     tags=["Training Jobs"])
    # app.include_router(fine_tuned_models_router, prefix="/v1/fine-tuned-models", tags=["Fine-Tuned Models"])

    @app.get("/", tags=["Health"])
    def read_root():
        return {"service": "training", "status": "online"}

    @app.get("/health", tags=["Health"])
    def health_check():
        return {"status": "ok"}

    return app


app = create_app()
