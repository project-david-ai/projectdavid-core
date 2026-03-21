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

import docker  # Required: pip install docker
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
HF_CACHE_PATH = os.getenv("HF_CACHE_PATH", "/root/.cache/huggingface")

# Unique ID for this physical machine
NODE_ID = os.getenv("NODE_ID", f"node_{socket.gethostname()}")

os.makedirs(LOCAL_SCRATCH, exist_ok=True)

# Initialize Docker client to manage inference containers on the host
try:
    docker_client = docker.from_env()
except Exception as e:
    logging_utility.error(f"Failed to initialize Docker SDK: {e}")
    docker_client = None

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
            time.sleep(15)

    thread = threading.Thread(target=heartbeat_loop, daemon=True)
    thread.start()


# ─── INFERENCE CLUSTER LOGIC (v2.0 Milestone) ───────────────────────────────


def manage_vllm_container(deployment: InferenceDeployment, action: str = "start"):
    """
    Physically manages the vLLM container lifecycle on the host machine.
    Uses the Docker SDK to spawn containers with NVIDIA runtime.
    """
    if not docker_client:
        logging_utility.error("Docker client not available. Cannot manage containers.")
        return False

    # Name is based on the deployment ID to allow multiple instances on one node
    container_name = f"pd_vllm_{deployment.id}"

    if action == "stop":
        try:
            container = docker_client.containers.get(container_name)
            container.remove(force=True)
            logging_utility.info(f"🛑 Stopped vLLM container: {container_name}")
            return True
        except docker.errors.NotFound:
            return True
        except Exception as e:
            logging_utility.error(f"Error stopping container {container_name}: {e}")
            return False

    # 1. Prepare Environment
    env = {
        "HF_TOKEN": os.getenv("HF_TOKEN"),
        "NVIDIA_VISIBLE_DEVICES": "all",
        # 🎯 BYPASS: Tells the NVIDIA driver to ignore the 'CUDA >= 12.9' requirement
        "NVIDIA_DISABLE_REQUIRE": "true",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    }

    # 2. Construct Command
    # We use the specific deployment metadata from the Mesh Ledger
    cmd = f"--model {deployment.base_model_id} --dtype float16 --max-model-len 2048 --gpu-memory-utilization 0.5"
    if deployment.fine_tuned_model_id:
        adapter_path = f"/mnt/training_data/{deployment.fine_tuned_model.storage_path}"
        cmd += f" --enable-lora --lora-modules {deployment.fine_tuned_model_id}={adapter_path}"

    try:
        # Cleanup any existing container with this name
        manage_vllm_container(deployment, action="stop")

        # 🎯 HARDENED GPU REQUEST: Explicitly request compute/utility for WSL2
        gpu_config = docker.types.DeviceRequest(
            count=-1, capabilities=[['gpu', 'compute', 'utility']]
        )

        logging_utility.info(
            f"🚢 Spawning vLLM: {container_name} for model {deployment.base_model_id}"
        )
        docker_client.containers.run(
            "vllm/vllm-openai:latest",
            name=container_name,
            command=cmd,
            environment=env,
            detach=True,
            device_requests=[gpu_config],
            runtime="nvidia",  # Required for GPU passthrough
            network="projectdavid-core_my_custom_network",
            ports={f"8000/tcp": deployment.port},
            volumes={
                HF_CACHE_PATH: {"bind": "/root/.cache/huggingface", "mode": "rw"},
                SHARED_PATH: {"bind": "/mnt/training_data", "mode": "rw"},
            },
        )
        return True
    except Exception as e:
        logging_utility.error(f"Docker spawn failed for {container_name}: {e}")
        return False


def start_deployment_supervisor():
    """
    Background thread that ensures physical vLLM state matches the DB ledger.
    Actively heals the node if a container crashes.
    """

    def supervisor_loop():
        logging_utility.info(f"👀 Inference Supervisor active for node: {NODE_ID}")
        while True:
            db = SessionLocal()
            try:
                # 1. Fetch active assignments for this node from the Global Ledger
                active_deployments = (
                    db.query(InferenceDeployment)
                    .filter(
                        InferenceDeployment.node_id == NODE_ID,
                        InferenceDeployment.status.in_([StatusEnum.pending, StatusEnum.active]),
                    )
                    .all()
                )

                for dep in active_deployments:
                    # 2. Check physical container state
                    container_name = f"pd_vllm_{dep.id}"
                    is_running = False
                    try:
                        c = docker_client.containers.get(container_name)
                        is_running = c.status == "running"
                    except:
                        pass

                    # 3. Synchronize: Start if pending or crashed
                    if dep.status == StatusEnum.pending or not is_running:
                        logging_utility.warning(f"🚨 Deployment drift! Syncing {container_name}")
                        if manage_vllm_container(dep, action="start"):
                            dep.status = StatusEnum.active
                            # Ensure VRAM is locked in the ledger
                            existing_alloc = (
                                db.query(GPUAllocation)
                                .filter(
                                    GPUAllocation.node_id == NODE_ID,
                                    (
                                        (GPUAllocation.model_id == dep.fine_tuned_model_id)
                                        if dep.fine_tuned_model_id
                                        else (GPUAllocation.job_id == None)
                                    ),
                                )
                                .first()
                            )

                            if not existing_alloc:
                                db.add(
                                    GPUAllocation(
                                        node_id=NODE_ID,
                                        model_id=dep.fine_tuned_model_id,
                                        vram_reserved_gb=4.0,
                                    )
                                )
                            db.commit()

            except Exception as e:
                logging_utility.error(f"Supervisor Loop Error: {e}")
            finally:
                db.close()
            time.sleep(20)

    thread = threading.Thread(target=supervisor_loop, daemon=True)
    thread.start()


# ─── TRAINING JOB LOGIC ─────────────────────────────────────────────────────


def process_job(job_id: str, user_id: str):
    """
    Standard LoRA training path. Reserved VRAM is released on completion.
    """
    db: Session = SessionLocal()
    local_data_path = None
    allocation_id = None

    try:
        job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        dataset = db.query(Dataset).filter(Dataset.id == job.dataset_id).first()

        logging_utility.info(f"🚀 Node {NODE_ID} claiming Training Job: {job_id}")
        job.status = StatusEnum.in_progress
        job.node_id = NODE_ID
        job.started_at = int(time.time())

        # VRAM Ledger Lock (5GB for Training)
        new_allocation = GPUAllocation(node_id=NODE_ID, job_id=job_id, vram_reserved_gb=5.0)
        db.add(new_allocation)
        db.commit()
        db.refresh(new_allocation)
        allocation_id = new_allocation.id

        # Data Staging
        storage = db.query(FileStorage).filter(FileStorage.file_id == dataset.file_id).first()
        local_data_path = os.path.join(LOCAL_SCRATCH, f"{job_id}.jsonl")
        smb = get_samba_client()
        smb.download_file(storage.storage_path, local_data_path)

        # Output Artifact Dir
        model_uuid = IdentifierService.generate_prefixed_id("ftm")
        model_rel_path = f"models/{model_uuid}"
        full_output_path = os.path.join(SHARED_PATH, model_rel_path)
        os.makedirs(full_output_path, exist_ok=True)

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
            os.getenv("TRAINING_PROFILE", "laptop"),
        ]

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(f"[{job_id}] {line.strip()}", flush=True)
        process.wait()

        if process.returncode == 0:
            new_ftm = FineTunedModel(
                id=model_uuid,
                user_id=user_id,
                training_job_id=job_id,
                name=f"FT: {job.base_model}",
                base_model=job.base_model,
                storage_path=model_rel_path,
                node_id=NODE_ID,
                status=StatusEnum.active,
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
        logging_utility.error(f"❌ Training Failure ({job_id}): {e}")
        if job:
            job.status = StatusEnum.failed
            job.last_error = str(e)
            db.commit()
    finally:
        # Atomic release of VRAM Ledger
        if allocation_id:
            db.query(GPUAllocation).filter(GPUAllocation.id == allocation_id).delete()
            db.commit()
        if local_data_path and os.path.exists(local_data_path):
            try:
                os.remove(local_data_path)
            except:
                pass
        db.close()


# ─── MAIN ───────────────────────────────────────────────────────────────────


def main():
    db = SessionLocal()
    try:
        node_heartbeat(db, NODE_ID)
        logging_utility.info(f"✅ Node {NODE_ID} joined the David Mesh.")
    finally:
        db.close()

    # Launch Cluster Daemons
    start_heartbeat()
    start_deployment_supervisor()

    # Handoff Loop
    r = get_redis()
    logging_utility.info(f"👷 Cluster Worker {NODE_ID} listening for jobs...")

    while True:
        try:
            result = r.brpop(QUEUE_NAME, timeout=30)
            if result:
                _, data = result
                payload = json.loads(data)

                # Targeted Dispatch Check
                if payload.get("target_node") and payload.get("target_node") != NODE_ID:
                    r.rpush(QUEUE_NAME, data)
                    time.sleep(1)
                    continue

                process_job(payload["job_id"], payload["user_id"])
        except Exception as e:
            logging_utility.error(f"Critical Loop Error: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()
