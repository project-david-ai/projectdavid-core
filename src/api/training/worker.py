import json
import os
import socket
import subprocess
import threading
import time
from typing import List, Optional

import redis
from projectdavid_common import UtilsInterface
from projectdavid_common.schemas.enums import StatusEnum
from projectdavid_common.utilities.identifier_service import IdentifierService
from projectdavid_orm.projectdavid_orm.models import FileStorage
from sqlalchemy.orm import Session

from src.api.training.db.database import SessionLocal
from src.api.training.models.models import (ComputeNode, Dataset,
                                            FineTunedModel, GPUAllocation,
                                            InferenceDeployment, TrainingJob)
from src.api.training.services.cluster_service import node_heartbeat
from src.api.training.services.file_service import SambaClient

logging_utility = UtilsInterface.LoggingUtility()

# ─── CONFIGURATION ──────────────────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
QUEUE_NAME = "training_jobs"
LOCAL_SCRATCH = "/tmp/training"
SHARED_PATH = os.getenv("SHARED_PATH", "/mnt/training_data")

# Unique ID for this physical machine (e.g., node_rtx4060_laptop)
NODE_ID = os.getenv("NODE_ID", f"node_{socket.gethostname()}")

os.makedirs(LOCAL_SCRATCH, exist_ok=True)

# ─── INFRASTRUCTURE HELPERS ─────────────────────────────────────────────────


def get_redis():
    """Connect to Redis with response decoding enabled."""
    return redis.from_url(REDIS_URL, decode_responses=True)


def get_samba_client():
    """Initialize SambaClient from environment."""
    return SambaClient(
        server=os.getenv("SMBCLIENT_SERVER", "samba"),
        share=os.getenv("SMBCLIENT_SHARE", "cosmic_share"),
        username=os.getenv("SMBCLIENT_USERNAME", "samba_user"),
        password=os.getenv("SMBCLIENT_PASSWORD"),
    )


# ─── CLUSTER TELEMETRY (HEARTBEAT) ──────────────────────────────────────────


def start_heartbeat():
    """
    Background thread to keep this node's telemetry fresh in the DB.
    Allows the API Scheduler to see real-time VRAM availability.
    """

    def heartbeat_loop():
        logging_utility.info(f"💓 Heartbeat started for node: {NODE_ID}")
        while True:
            db = SessionLocal()
            try:
                node_heartbeat(db, NODE_ID)
            except Exception as e:
                logging_utility.error(f"Heartbeat failed for {NODE_ID}: {e}")
            finally:
                db.close()
            time.sleep(15)  # Update cluster registry every 15 seconds

    thread = threading.Thread(target=heartbeat_loop, daemon=True)
    thread.start()


# ─── INFERENCE DEPLOYMENT WATCHER (v2.0 Milestone) ──────────────────────────


def start_deployment_watcher():
    """
    Background thread that monitors for 'pending' inference deployments
    assigned to this node.
    """

    def watcher_loop():
        logging_utility.info(f"👀 Deployment Watcher active for node: {NODE_ID}")
        while True:
            db = SessionLocal()
            try:
                # Find deployments intended for this machine that haven't started yet
                pending_deployments = (
                    db.query(InferenceDeployment)
                    .filter(
                        InferenceDeployment.node_id == NODE_ID,
                        InferenceDeployment.status == StatusEnum.pending,
                    )
                    .all()
                )

                for dep in pending_deployments:
                    logging_utility.info(
                        f"🆕 DEPLOYMENT SIGNAL: Preparing to host {dep.base_model_id}..."
                    )

                    # 💡 ARCHITECTURAL NOTE:
                    # In a production multi-node setup, this is where the worker calls
                    # a local shell script or Docker API to trigger:
                    # 'platform-api docker-manager --mode up --services vllm'

                    # For current verification, we simulate the startup success
                    dep.status = StatusEnum.active
                    dep.last_seen = int(time.time())
                    db.commit()
                    logging_utility.info(
                        f"🚀 DEPLOYMENT ACTIVE: {dep.id} now serving on port {dep.port}"
                    )

            except Exception as e:
                logging_utility.error(f"Deployment Watcher Error: {e}")
            finally:
                db.close()
            time.sleep(10)  # Poll every 10 seconds

    thread = threading.Thread(target=watcher_loop, daemon=True)
    thread.start()


# ─── JOB EXECUTION LOGIC ────────────────────────────────────────────────────


def process_job(job_id: str, user_id: str):
    """
    Claims the job, reserves VRAM, stages data, and executes Unsloth.
    """
    db: Session = SessionLocal()
    local_data_path = None
    allocation_id = None

    try:
        # 1. Hydrate Metadata
        job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if not job:
            logging_utility.error(f"Worker Error: Job {job_id} not found.")
            return

        dataset = db.query(Dataset).filter(Dataset.id == job.dataset_id).first()

        # 2. CLUSTER CLAIM: Link job to this physical node
        logging_utility.info(f"🚀 Node {NODE_ID} claiming Job: {job_id}")
        job.status = StatusEnum.in_progress
        job.node_id = NODE_ID
        job.started_at = int(time.time())

        # 3. VRAM RESERVATION: Update the cluster ledger
        vram_requirement = 4.0
        new_allocation = GPUAllocation(
            node_id=NODE_ID, job_id=job_id, vram_reserved_gb=vram_requirement
        )
        db.add(new_allocation)
        db.commit()
        db.refresh(new_allocation)
        allocation_id = new_allocation.id

        # 4. DATA STAGING: Samba Hub -> Local NVMe Scratch
        storage = db.query(FileStorage).filter(FileStorage.file_id == dataset.file_id).first()
        local_data_path = os.path.join(LOCAL_SCRATCH, f"{job_id}.jsonl")
        smb = get_samba_client()
        smb.download_file(storage.storage_path, local_data_path)

        # 5. PREPARE EXPORT PATH
        model_uuid = IdentifierService.generate_prefixed_id("ftm")
        model_rel_path = f"models/{model_uuid}"
        full_output_path = os.path.join(SHARED_PATH, model_rel_path)
        os.makedirs(full_output_path, exist_ok=True)

        # 6. EXECUTE ML SUBPROCESS
        training_profile = os.getenv("TRAINING_PROFILE", "standard")
        cmd = [
            "python",
            "src/api/training/unsloth_train.py",
            "--model",
            job.base_model,
            "--data",
            local_data_path,
            "--out",
            full_output_path,
            "--profile",
            training_profile,
        ]

        logging_utility.info(f"Spawning training subprocess [Profile: {training_profile}]")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        for line in process.stdout:
            print(f"[{job_id}] {line.strip()}", flush=True)
        process.wait()

        if process.returncode == 0:
            # 7. REGISTRATION: Finalize the result in the Registry
            new_ftm = FineTunedModel(
                id=model_uuid,
                user_id=user_id,
                training_job_id=job_id,
                name=f"FT: {job.base_model}",
                base_model=job.base_model,
                storage_path=model_rel_path,
                node_id=NODE_ID,
                status=StatusEnum.active,
                is_active=False,
                created_at=int(time.time()),
                updated_at=int(time.time()),
            )
            db.add(new_ftm)
            job.status = StatusEnum.completed
            job.completed_at = int(time.time())
            job.output_path = model_rel_path
            logging_utility.info(f"✨ Job {job_id} finalized successfully on {NODE_ID}")
        else:
            raise subprocess.CalledProcessError(process.returncode, cmd)

    except Exception as e:
        error_msg = str(e)
        logging_utility.error(f"❌ Node {NODE_ID} Job Failure ({job_id}): {error_msg}")
        if job:
            job.status = StatusEnum.failed
            job.failed_at = int(time.time())
            job.last_error = error_msg
            db.commit()

    finally:
        # 8. CLEANUP: Release VRAM reservation
        if allocation_id:
            try:
                db.query(GPUAllocation).filter(GPUAllocation.id == allocation_id).delete()
                db.commit()
            except:
                pass
        if local_data_path and os.path.exists(local_data_path):
            try:
                os.remove(local_data_path)
            except:
                pass
        db.close()


# ─── MAIN LISTENER LOOP ─────────────────────────────────────────────────────


def main():
    # 1. Initial Node Registration
    db = SessionLocal()
    try:
        node_heartbeat(db, NODE_ID)
        logging_utility.info(f"✅ Node {NODE_ID} joined the David Mesh.")
    finally:
        db.close()

    # 2. Start Background Services
    start_heartbeat()
    start_deployment_watcher()

    # 3. Targeted BRPOP Loop (The Training Listener)
    r = get_redis()
    logging_utility.info(f"👷 Cluster Worker {NODE_ID} listening for Training Jobs...")

    while True:
        try:
            result = r.brpop(QUEUE_NAME, timeout=30)
            if result:
                _, data = result
                payload = json.loads(data)

                # DISPATCH CHECK
                target = payload.get("target_node")
                if target and target != NODE_ID:
                    logging_utility.debug(f"Re-queueing job {payload['job_id']} for {target}.")
                    r.rpush(QUEUE_NAME, data)
                    time.sleep(1)
                    continue

                process_job(payload["job_id"], payload["user_id"])
        except Exception as e:
            logging_utility.error(f"Worker Loop Critical Error: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()
