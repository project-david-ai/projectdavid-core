# ProjectDavid — Phase 2: Distributed Cluster & Inference Mesh Handover
**Date:** 2026-03-21  
**Status:** **v2.0 Managed Mesh Active** — Resource Awareness & Cluster Integrity verified.  
**Repo:** `projectdavid-core`  
**Stack:** FastAPI · MySQL · Redis · Samba · vLLM · Unsloth (LoRA)

---

## 1. Latest Achievement: The "Singleton" Guard
To ensure the integrity of the cluster's metadata, we have implemented an **Active API Lease** via Redis. 
*   **The Logic:** Only one Training API instance can be "Master" at a time. It holds a 30s lease in Redis (`cluster:active_training_api`).
*   **The Benefit:** Prevents race conditions during model activation and ensures the **Node Reaper** doesn't have multiple masters trying to clean the same VRAM ledger.

### Final `src/api/training/app.py` (Merged Version)
```python
import asyncio
import os
import sys
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from projectdavid_common import UtilsInterface
from projectdavid_orm.projectdavid_orm.base import Base

from src.api.training.db.database import engine, wait_for_db, SessionLocal
from src.api.training.routers import training_router
from src.api.training.services.cluster_service import (
    reap_stale_nodes, acquire_api_lease, renew_api_lease
)
from src.api.training.services.training_service import get_redis_client

logging_utility = UtilsInterface.LoggingUtility()
INSTANCE_ID = str(uuid.uuid4())

# ──────────────────────────────────────────────────────────────────────────────
# CLUSTER DAEMONS
# ──────────────────────────────────────────────────────────────────────────────

async def cluster_maintenance_loop(r):
    """
    1. Renews the Singleton API Lease.
    2. Executes the Node Reaper (Cleanup stale VRAM/Jobs).
    """
    while True:
        # 1. Lease Renewal (Critical Singleton logic)
        if not renew_api_lease(r, INSTANCE_ID):
            logging_utility.critical("🛑 API LEASE LOST. Shutting down to prevent metadata corruption.")
            os._exit(1)

        # 2. Reaper (Maintenance)
        db = SessionLocal()
        try:
            reap_stale_nodes(db)
        except Exception as e:
            logging_utility.error(f"Reaper Error: {e}")
        finally:
            db.close()
        
        await asyncio.sleep(20)

@asynccontextmanager
async def lifespan(app: FastAPI):
    r = get_redis_client()
    # Attempt to grab the Singleton Mic
    if not acquire_api_lease(r, INSTANCE_ID):
        logging_utility.error("❌ STARTUP BLOCKED: Another Training API is already Master.")
        sys.exit(1)

    logging_utility.info(f"👑 Instance {INSTANCE_ID} is now the Master Training API.")
    maintenance_task = asyncio.create_task(cluster_maintenance_loop(r))
    yield
    maintenance_task.cancel()
    r.delete("cluster:active_training_api")

# ──────────────────────────────────────────────────────────────────────────────
# APP FACTORY
# ──────────────────────────────────────────────────────────────────────────────

def create_app(init_db: bool = True) -> FastAPI:
    wait_for_db()
    app = FastAPI(title="ProjectDavid — Managed Training API", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    app.include_router(training_router, prefix="/v1")
    
    if init_db:
        Base.metadata.create_all(bind=engine)
    return app

app = create_app()


2. Cluster Architecture State (The 4 New Pillars)
We have expanded the models.py to include the Resource Ledger. The system is now hardware-aware.
compute_nodes: Stores real-time GPU telemetry (rtx4060_laptop_main verified).
gpu_allocations: The "VRAM Ledger" that tracks reservations by Jobs and Deployments.
base_models: A relational catalog of standard models (Llama-3, Qwen-2.5) sitting in the cache.
inference_deployments: Tracks which Node is running which model on which Port.
3. Verified Logic Paths
Worker Heartbeat: Verified. Laptop registers itself and updates usage every 15s.
Smart Scheduler: Verified. API queries compute_nodes and targets the healthiest GPU via Redis target_node payload.
VRAM Claim: Verified. Worker locks 4GB in gpu_allocations during training and releases it in the finally block.
Fine-Tuning: Verified. Real Unsloth run produced 74MB .safetensors adapters on the 4060.
Activation: Verified. Mutex logic ensures only one model is is_active per user.




It has been an absolute privilege to help you lay the foundation for ProjectDavid. You have successfully moved from a single-machine script to a Distributed GPU Mesh architecture.
Below is the meticulous handover document for your next session. It covers the final state of the Singleton Guard and the roadmap for the Inference Cluster.
code
Md
# ProjectDavid — Phase 2: Distributed Cluster & Inference Mesh Handover
**Date:** 2026-03-21  
**Status:** **v2.0 Managed Mesh Active** — Resource Awareness & Cluster Integrity verified.  
**Repo:** `projectdavid-core`  
**Stack:** FastAPI · MySQL · Redis · Samba · vLLM · Unsloth (LoRA)

---

## 1. Latest Achievement: The "Singleton" Guard
To ensure the integrity of the cluster's metadata, we have implemented an **Active API Lease** via Redis. 
*   **The Logic:** Only one Training API instance can be "Master" at a time. It holds a 30s lease in Redis (`cluster:active_training_api`).
*   **The Benefit:** Prevents race conditions during model activation and ensures the **Node Reaper** doesn't have multiple masters trying to clean the same VRAM ledger.

### Final `src/api/training/app.py` (Merged Version)
```python
import asyncio
import os
import sys
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from projectdavid_common import UtilsInterface
from projectdavid_orm.projectdavid_orm.base import Base

from src.api.training.db.database import engine, wait_for_db, SessionLocal
from src.api.training.routers import training_router
from src.api.training.services.cluster_service import (
    reap_stale_nodes, acquire_api_lease, renew_api_lease
)
from src.api.training.services.training_service import get_redis_client

logging_utility = UtilsInterface.LoggingUtility()
INSTANCE_ID = str(uuid.uuid4())

# ──────────────────────────────────────────────────────────────────────────────
# CLUSTER DAEMONS
# ──────────────────────────────────────────────────────────────────────────────

async def cluster_maintenance_loop(r):
    """
    1. Renews the Singleton API Lease.
    2. Executes the Node Reaper (Cleanup stale VRAM/Jobs).
    """
    while True:
        # 1. Lease Renewal (Critical Singleton logic)
        if not renew_api_lease(r, INSTANCE_ID):
            logging_utility.critical("🛑 API LEASE LOST. Shutting down to prevent metadata corruption.")
            os._exit(1)

        # 2. Reaper (Maintenance)
        db = SessionLocal()
        try:
            reap_stale_nodes(db)
        except Exception as e:
            logging_utility.error(f"Reaper Error: {e}")
        finally:
            db.close()
        
        await asyncio.sleep(20)

@asynccontextmanager
async def lifespan(app: FastAPI):
    r = get_redis_client()
    # Attempt to grab the Singleton Mic
    if not acquire_api_lease(r, INSTANCE_ID):
        logging_utility.error("❌ STARTUP BLOCKED: Another Training API is already Master.")
        sys.exit(1)

    logging_utility.info(f"👑 Instance {INSTANCE_ID} is now the Master Training API.")
    maintenance_task = asyncio.create_task(cluster_maintenance_loop(r))
    yield
    maintenance_task.cancel()
    r.delete("cluster:active_training_api")

# ──────────────────────────────────────────────────────────────────────────────
# APP FACTORY
# ──────────────────────────────────────────────────────────────────────────────

def create_app(init_db: bool = True) -> FastAPI:
    wait_for_db()
    app = FastAPI(title="ProjectDavid — Managed Training API", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    app.include_router(training_router, prefix="/v1")
    
    if init_db:
        Base.metadata.create_all(bind=engine)
    return app

app = create_app()
2. Cluster Architecture State (The 4 New Pillars)
We have expanded the models.py to include the Resource Ledger. The system is now hardware-aware.
compute_nodes: Stores real-time GPU telemetry (rtx4060_laptop_main verified).
gpu_allocations: The "VRAM Ledger" that tracks reservations by Jobs and Deployments.
base_models: A relational catalog of standard models (Llama-3, Qwen-2.5) sitting in the cache.
inference_deployments: Tracks which Node is running which model on which Port.
3. Verified Logic Paths
Worker Heartbeat: Verified. Laptop registers itself and updates usage every 15s.
Smart Scheduler: Verified. API queries compute_nodes and targets the healthiest GPU via Redis target_node payload.
VRAM Claim: Verified. Worker locks 4GB in gpu_allocations during training and releases it in the finally block.
Fine-Tuning: Verified. Real Unsloth run produced 74MB .safetensors adapters on the 4060.
Activation: Verified. Mutex logic ensures only one model is is_active per user.
4. NEXT STEPS (Milestone: Inference Cluster)
Step 1: Seed the Catalog
Run scripts/seed_model_catalog.py to populate the base_models table. This allows the system to distinguish between "Backbones" and "Adapters."
Step 2: The Multi-LoRA Handoff
Refactor model_registry_service.activate_model() to target the Inference Cluster:
API checks compute_nodes for an active vLLM node.
API creates an InferenceDeployment record linking the FineTunedModel to a physical Node and Port.
The Docker Manager (CLI) uses this record to boot the vLLM container with the correct --lora-modules mapping.
Step 3: Global Load Balancing
Update the SDK to query the InferenceDeployment table.
Instead of hardcoding localhost:8001, the SDK should ask: "Where is my active model?"
The API returns: http://192.168.1.50:8001 (or whatever the Cluster Deployment record says)


5. Key Known Issues / Watch Points
WSL2 Ghost Volumes: If vllm sees a 25-byte file instead of 74MB, use the Absolute Path in .env for SHARED_PATH and run wsl --shutdown.
VRAM Utilization: Keep gpu_memory_utilization at 0.7 for the RTX 4060 to prevent Windows OS crashes.
Cleanup Daemon: Ensure purge_expired_files.py ignores files with purpose == "training" and any path containing models/.
6. Progress Tracker
Milestone	Status	Notes
Single-Node Training	✅	v1.0 Complete
Resource Ledger	✅	Tables & Logic Verified
Smart Scheduler	✅	Node Assignment Active
Singleton API Guard	✅	Redis Lease Integrated
Inference Deployments	❌	Next Priority
Load Balancer	❌	v3.0 Vision


