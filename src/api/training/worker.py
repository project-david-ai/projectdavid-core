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

# The Docker network all services share — used to look up the container's real IP
DOCKER_NETWORK_NAME = os.getenv("DOCKER_NETWORK_NAME", "projectdavid-core_my_custom_network")

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
    return redis.from_url(REDIS_URL, decode_responses=True)


def get_samba_client():
    return SambaClient(
        server=os.getenv("SMBCLIENT_SERVER", "samba"),
        share=os.getenv("SMBCLIENT_SHARE", "cosmic_share"),
        username=os.getenv("SMBCLIENT_USERNAME", "samba_user"),
        password=os.getenv("SMBCLIENT_PASSWORD"),
    )


# ─── CLUSTER TELEMETRY (HEARTBEAT) ──────────────────────────────────────────


def start_heartbeat():
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


def _resolve_container_ip(container, network_name: str) -> Optional[str]:
    """
    Inspects a running container and returns its real IP address on the
    specified Docker network.

    Falls back to the container name only as a last resort, so callers
    always get something routable rather than None.

    Root cause this fixes:
      Docker's internal hostname for a container is its 12-char short ID
      (e.g. '3ee617956d53'). When that value is stored as internal_hostname
      and used to build http://<hostname>:<port>, it does not resolve
      reliably inside the overlay network. The actual IPAddress field is
      always a proper dotted-quad and is unambiguously routable.
    """
    try:
        container.reload()  # Ensure we have post-start network metadata
        networks = container.attrs.get("NetworkSettings", {}).get("Networks", {})

        # 1. Prefer the known shared project network
        if network_name in networks:
            ip = networks[network_name].get("IPAddress")
            if ip:
                logging_utility.info(f"🔍 Container {container.name} IP on '{network_name}': {ip}")
                return ip

        # 2. Fall back to any network that has an IP
        for net_name, net_cfg in networks.items():
            ip = net_cfg.get("IPAddress")
            if ip:
                logging_utility.warning(
                    f"⚠️  Network '{network_name}' not found; " f"using IP from '{net_name}': {ip}"
                )
                return ip

    except Exception as e:
        logging_utility.error(f"IP resolution failed for {container.name}: {e}")

    # 3. Last resort — container name (resolvable via Docker DNS within the network)
    logging_utility.warning(
        f"⚠️  Could not resolve IP for {container.name}; " f"falling back to container name."
    )
    return container.name


def manage_vllm_container(deployment: InferenceDeployment, action: str = "start") -> Optional[str]:
    """
    Physically manages the vLLM container lifecycle on the host machine.

    Returns:
        On 'start': the container's routable IP address on the Docker network
                    (NOT the container name / short-ID hostname).
        On 'stop':  the container name that was removed.
    """
    if not docker_client:
        return None

    container_name = f"pd_vllm_{deployment.id}"

    if action == "stop":
        try:
            container = docker_client.containers.get(container_name)
            container.remove(force=True)
            logging_utility.info(f"🛑 Stopped vLLM container: {container_name}")
            return container_name
        except Exception as e:
            logging_utility.warning(f"⚠️  Could not stop container {container_name}: {e}")
            return container_name

    # ── 1. Prepare Environment & Command ─────────────────────────────────
    env = {
        "HF_TOKEN": os.getenv("HF_TOKEN"),
        "NVIDIA_DISABLE_REQUIRE": "true",  # Bypasses CUDA driver version gate
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "NVIDIA_VISIBLE_DEVICES": "all",
    }

    cmd = f"--model {deployment.base_model_id} --dtype float16 --max-model-len 2048 --gpu-memory-utilization 0.5"
    if deployment.fine_tuned_model_id:
        adapter_path = f"/mnt/training_data/{deployment.fine_tuned_model.storage_path}"
        cmd += f" --enable-lora --lora-modules {deployment.fine_tuned_model_id}={adapter_path}"

    try:
        # ── 2. AGGRESSIVE CLEANUP ─────────────────────────────────────────
        # Scan ALL containers (not just our own name) for anything already
        # bound to our target port and evict any squatters before spawning.
        all_containers = docker_client.containers.list(all=True)
        for c in all_containers:
            port_bindings = c.attrs.get('HostConfig', {}).get('PortBindings', {})
            if f"8000/tcp" in port_bindings:
                bound_port = port_bindings[f"8000/tcp"][0].get('HostPort')
                if bound_port == str(deployment.port):
                    logging_utility.warning(
                        f"🧹 Port {deployment.port} is hogged by {c.name}. Evicting..."
                    )
                    c.remove(force=True)

        # ── 3. Setup GPU and Spawn ────────────────────────────────────────
        # Hardened GPU Request for WSL2/Windows
        gpu_config = docker.types.DeviceRequest(
            count=-1, capabilities=[['gpu', 'compute', 'utility']]
        )

        logging_utility.info(f"🚢 Spawning vLLM: {container_name}")
        container = docker_client.containers.run(
            "vllm/vllm-openai:latest",
            name=container_name,
            command=cmd,
            environment=env,
            detach=True,
            device_requests=[gpu_config],
            runtime="nvidia",
            network=DOCKER_NETWORK_NAME,
            ports={f"8000/tcp": deployment.port},
            volumes={
                HF_CACHE_PATH: {"bind": "/root/.cache/huggingface", "mode": "rw"},
                SHARED_PATH: {"bind": "/mnt/training_data", "mode": "rw"},
            },
        )

        # ── 4. Resolve Real IP — the critical fix ─────────────────────────
        # We must store the container's actual dotted-quad IP, NOT its name
        # or short-ID hostname. The InferenceResolver builds the vLLM endpoint
        # URL directly from this value: http://<internal_hostname>:<port>
        container_ip = _resolve_container_ip(container, DOCKER_NETWORK_NAME)
        logging_utility.info(
            f"✅ vLLM container {container_name} is up — routable IP: {container_ip}"
        )
        return container_ip

    except Exception as e:
        logging_utility.error(f"Docker spawn failed for {container_name}: {e}")
        return None


def start_deployment_supervisor():
    """
    Background thread that ensures physical vLLM state matches the DB ledger.
    """

    def supervisor_loop():
        logging_utility.info(f"👀 Inference Supervisor active for node: {NODE_ID}")
        while True:
            db = SessionLocal()
            try:
                active_deployments = (
                    db.query(InferenceDeployment)
                    .filter(
                        InferenceDeployment.node_id == NODE_ID,
                        InferenceDeployment.status.in_([StatusEnum.pending, StatusEnum.active]),
                    )
                    .all()
                )

                for dep in active_deployments:
                    container_name = f"pd_vllm_{dep.id}"
                    is_running = False
                    try:
                        c = docker_client.containers.get(container_name)
                        is_running = c.status == "running"
                    except Exception:
                        pass

                    if dep.status == StatusEnum.pending or not is_running:
                        logging_utility.warning(f"🚨 Deployment drift! Syncing {container_name}")

                        # manage_vllm_container now returns the container's real IP
                        container_ip = manage_vllm_container(dep, action="start")
                        if container_ip:
                            dep.status = StatusEnum.active
                            # Store the real dotted-quad IP so InferenceResolver
                            # can build a valid http://<ip>:<port> endpoint URL.
                            dep.internal_hostname = container_ip

                            # Reserve VRAM in the cluster ledger
                            existing_alloc = (
                                db.query(GPUAllocation)
                                .filter(
                                    GPUAllocation.node_id == NODE_ID,
                                    (
                                        (GPUAllocation.model_id == dep.fine_tuned_model_id)
                                        if dep.fine_tuned_model_id
                                        else (GPUAllocation.job_id.is_(None))
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

        # VRAM Ledger Lock
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

        # Artifact Path
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
            logging_utility.info(f"✨ Job {job_id} finalized successfully.")
        else:
            raise subprocess.CalledProcessError(process.returncode, cmd)

    except Exception as e:
        logging_utility.error(f"❌ Training Failure ({job_id}): {e}")
        if job:
            job.status = StatusEnum.failed
            job.last_error = str(e)
            db.commit()
    finally:
        if allocation_id:
            db.query(GPUAllocation).filter(GPUAllocation.id == allocation_id).delete()
            db.commit()
        if local_data_path and os.path.exists(local_data_path):
            os.remove(local_data_path)
        db.close()


# ─── MAIN ───────────────────────────────────────────────────────────────────


def main():
    db = SessionLocal()
    try:
        node_heartbeat(db, NODE_ID)
        logging_utility.info(f"✅ Node {NODE_ID} mesh heartbeat initialized.")
    finally:
        db.close()

    start_heartbeat()
    start_deployment_supervisor()

    r = get_redis()
    logging_utility.info(f"👷 Cluster Worker {NODE_ID} listening for jobs...")

    while True:
        try:
            result = r.brpop(QUEUE_NAME, timeout=30)
            if result:
                _, data = result
                payload = json.loads(data)
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
