from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from projectdavid_common import UtilsInterface
from projectdavid_orm.projectdavid_orm.base import Base

from src.api.entities_api.db.database import engine, wait_for_databases
from src.api.entities_api.observability.tracing import setup_tracing
from src.api.entities_api.routers import api_router

logging_utility = UtilsInterface.LoggingUtility()

wait_for_databases()


def create_app(init_db: bool = True) -> FastAPI:
    logging_utility.info("Creating FastAPI app")

    app = FastAPI(
        title="Entities",
        description="API for AI inference",
        version="1.0.0",
        docs_url="/mydocs",
        redoc_url="/altredoc",
        openapi_url="/openapi.json",
    )

    # 🧠 OTel MUST be initialised before router binding
    setup_tracing(app)

    # CORS — allows ReDoc/Elements/Swagger to fetch openapi.json cross-origin
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # tighten to specific origins in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/v1")

    @app.get("/")
    def read_root():
        logging_utility.info("Root endpoint accessed")
        return {"message": "Welcome to the API!"}

    return app


app = create_app()
