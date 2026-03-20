from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from projectdavid_common import UtilsInterface
from projectdavid_orm.projectdavid_orm.base import Base
from sqlalchemy import text

# Training-specific imports
from src.api.training.db.database import engine, wait_for_db
from src.api.training.routers import training_router

logging_utility = UtilsInterface.LoggingUtility()

# Block until MySQL is ready
wait_for_db()


def create_app(init_db: bool = True) -> FastAPI:
    logging_utility.info("Creating Training API app")

    app = FastAPI(
        title="ProjectDavid — Training Service",
        description="Private OpenAI-in-a-box Fine-Tuning Pipeline",
        version="1.0.0",
        docs_url="/docs",
        openapi_url="/openapi.json",
    )

    # CORS — allows SDK and UI to interact with the API
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Inject the aggregated router with the version prefix
    # Result: /v1/datasets and /v1/training-jobs
    app.include_router(training_router, prefix="/v1")

    @app.get("/")
    def read_root():
        logging_utility.info("Training Root endpoint accessed")
        return {"service": "training", "status": "online"}

    @app.get("/health")
    def health_check():
        return {"status": "ok"}

    if init_db:
        logging_utility.info("Initializing Training database schema...")
        # This creates the datasets, training_jobs, and fine_tuned_models tables
        Base.metadata.create_all(bind=engine)

    return app


app = create_app()
