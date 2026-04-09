import asyncio
import os
import sys
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from projectdavid_common import UtilsInterface

from src.api.training.db.database import wait_for_db
from src.api.training.routers import training_router
from src.api.training.services.lease_service import acquire_api_lease, renew_api_lease
from src.api.training.services.training_service import get_redis_client

logging_utility = UtilsInterface.LoggingUtility()

# ── Sustainability notice ─────────────────────────────────────────────────────
print(
    """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Project David — Sovereign AI Runtime

  Free for personal and research use.
  Organisations running Project David in production are
  invited to discuss a commercial licence.

  licensing@projectdavid.co.uk | projectdavid.co.uk
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""",
    flush=True,
)

logging_utility.info(
    "Project David is free for personal and research use. "
    "Organisations running Project David in production are invited to discuss "
    "a commercial licence:  licensing@projectdavid.co.uk | projectdavid.co.uk"
)

# Generated once per process — identity of THIS instance
INSTANCE_ID = str(uuid.uuid4())

# Block until MySQL is ready
wait_for_db()

# ──────────────────────────────────────────────────────────────────────────────
# CLUSTER MANAGEMENT DAEMON
# ──────────────────────────────────────────────────────────────────────────────


async def cluster_maintenance_loop(r):
    """
    Renews the singleton API lease every 20 seconds.

    Hard exits if the lease is lost — prevents split-brain metadata writes
    when two training-api instances start simultaneously.

    Node reaping and GPU allocation tracking are no longer performed here.
    GPU resource management is handled by Ray (inference-worker) and the
    InferenceReconciler. ComputeNode / GPUAllocation tables are legacy and
    will be dropped in a future migration.
    """
    logging_utility.info("🔄 Cluster maintenance loop started.")
    while True:
        await asyncio.sleep(20)

        if not renew_api_lease(r, INSTANCE_ID):
            logging_utility.critical(
                "🛑 API LEASE LOST. Shutting down to prevent metadata corruption."
            )
            os._exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# LIFESPAN
# ──────────────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──
    r = get_redis_client()

    # Attempt to claim the singleton lease — blocks startup if another master exists
    if not acquire_api_lease(r, INSTANCE_ID):
        logging_utility.error(
            "❌ STARTUP BLOCKED: Another Training API instance is already Master. "
            "Ensure no other instance is running before starting."
        )
        sys.exit(1)

    logging_utility.info(f"👑 Instance [{INSTANCE_ID}] claimed Master lease.")

    # Note: training-api does NOT connect to Ray directly.
    # Resource availability is queried via the Ray dashboard HTTP API
    # (http://inference_worker:8265/api/v0/nodes) in deployment_service.py.
    # This avoids GCC port conflicts and requires no Ray init here.

    maintenance_task = asyncio.create_task(cluster_maintenance_loop(r))

    yield

    # ── Shutdown ──
    maintenance_task.cancel()
    # Explicitly release the lease so a new instance can start immediately
    # without waiting for the 30s TTL to expire
    r.delete("cluster:active_training_api")
    logging_utility.info(
        f"🔓 Instance [{INSTANCE_ID}] released Master lease. Cluster is free."
    )


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

    # All routers are registered via the training_router aggregator.
    # See src/api/training/routers/__init__.py for the full routing table.
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

    return app


app = create_app()
