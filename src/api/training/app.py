import asyncio
import os
import sys
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from projectdavid_common import UtilsInterface
from projectdavid_orm.projectdavid_orm.base import Base

from src.api.training.db.database import SessionLocal, engine, wait_for_db
from src.api.training.routers import training_router
from src.api.training.services.cluster_service import (acquire_api_lease,
                                                       reap_stale_nodes,
                                                       renew_api_lease)
from src.api.training.services.training_service import get_redis_client

logging_utility = UtilsInterface.LoggingUtility()

# Generated once per process — this is the identity of THIS instance
INSTANCE_ID = str(uuid.uuid4())

# Block until MySQL is ready
wait_for_db()

# ──────────────────────────────────────────────────────────────────────────────
# CLUSTER MANAGEMENT DAEMON
# ──────────────────────────────────────────────────────────────────────────────


async def cluster_maintenance_loop(r):
    """
    Single unified loop that:
      1. Renews the Singleton API Lease — hard exit if lost.
      2. Runs the Node Reaper to clean up stale VRAM/Jobs.
    """
    logging_utility.info("🔄 Cluster maintenance loop started.")
    while True:
        await asyncio.sleep(20)

        # 1. Lease renewal — if we lose this, we must die immediately
        if not renew_api_lease(r, INSTANCE_ID):
            logging_utility.critical(
                "🛑 API LEASE LOST. Shutting down to prevent metadata corruption."
            )
            os._exit(1)

        # 2. Reaper — clean up ghost nodes and abandoned jobs
        db = SessionLocal()
        try:
            reap_stale_nodes(db)
        except Exception as e:
            logging_utility.error(f"Reaper Task Error: {e}")
        finally:
            db.close()


# ──────────────────────────────────────────────────────────────────────────────
# LIFESPAN
# ──────────────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──
    r = get_redis_client()

    # Attempt to claim the Singleton lease — blocks startup if another master exists
    if not acquire_api_lease(r, INSTANCE_ID):
        logging_utility.error(
            "❌ STARTUP BLOCKED: Another Training API instance is already Master. "
            "Ensure no other instance is running before starting."
        )
        sys.exit(1)

    logging_utility.info(f"👑 Instance [{INSTANCE_ID}] claimed Master lease.")

    # Note: training-api does NOT connect to Ray directly.
    # Resource availability is queried via the Ray dashboard HTTP API
    # (http://training_worker:8265/api/v0/nodes) in model_registry_service.py.
    # This avoids GCS port conflicts with Redis and requires no Ray init here.

    maintenance_task = asyncio.create_task(cluster_maintenance_loop(r))

    yield

    # ── Shutdown ──
    maintenance_task.cancel()
    # Explicitly release the lease so a new instance can start immediately
    # without waiting for the 30s TTL to expire
    r.delete("cluster:active_training_api")
    logging_utility.info(f"🔓 Instance [{INSTANCE_ID}] released Master lease. Cluster is free.")


# ──────────────────────────────────────────────────────────────────────────────
# APP FACTORY
# ──────────────────────────────────────────────────────────────────────────────


def create_app(init_db: bool = True) -> FastAPI:
    logging_utility.info("Creating Training API app")

    app = FastAPI(
        title="ProjectDavid — Training Service",
        description="Private OpenAI-in-a-box Fine-Tuning Pipeline & GPU Mesh",
        version="2.0.0",
        docs_url="/docs",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(training_router, prefix="/v1")

    @app.get("/")
    def read_root():
        return {
            "service": "training",
            "status": "online",
            "instance_id": INSTANCE_ID,
            "cluster_management": "active",
        }

    @app.get("/health")
    def health_check():
        return {"status": "ok", "instance_id": INSTANCE_ID}

    if init_db:
        logging_utility.info("Initializing Training database schema...")
        Base.metadata.create_all(bind=engine)

    return app


app = create_app()
