import json
import os
import socket
import subprocess
import threading
import time
from typing import List, Optional

import docker  # Required: pip install docker
import ray
import redis
from projectdavid_common import UtilsInterface
from projectdavid_common.schemas.enums import StatusEnum
from projectdavid_common.utilities.identifier_service import IdentifierService
from projectdavid_orm.projectdavid_orm.models import FileStorage
from sqlalchemy.orm import Session

from src.api.training.db.database import SessionLocal
from src.api.training.models.models import (
    Dataset,
    FineTunedModel,
    InferenceDeployment,
    TrainingJob,
)
from src.api.training.services.file_service import SambaClient

logging_utility = UtilsInterface.LoggingUtility()

# ─── CONFIGURATION ──────────────────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
QUEUE_NAME = "training_jobs"
LOCAL_SCRATCH = "/tmp/training"
SHARED_PATH = os.getenv("SHARED_PATH", "/mnt/training_data")
HF_CACHE_PATH = os.getenv("HF_CACHE_PATH", "/root/.cache/huggingface")

DOCKER_NETWORK_NAME = os.getenv("DOCKER_NETWORK_NAME", "projectdavid-core_my_custom_network")

# Phase 4: NODE_ID is derived from the Ray node ID at startup rather than
# a manually configured env var. Kept as a fallback for logging only.
NODE_ID = os.getenv("NODE_ID", f"node_{socket.gethostname()}")

os.makedirs(LOCAL_SCRATCH, exist_ok=True)

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


# ─── PHASE 4: RAY NODE IDENTITY ──────────────────────────────────────────────


def get_ray_node_id() -> str:
    """
    Phase 4: Returns the Ray node ID of the current head node.

    This replaces the NODE_ID env var / compute_nodes heartbeat pattern.
    Ray always knows which nodes are alive — we just read that state.
    The node ID is a stable hex string (e.g. aea109206314b87a...) that
    uniquely identifies this physical machine in the Ray cluster.
    """
    try:
        nodes = ray.nodes()
        for node in nodes:
            if node.get("Alive") and node.get("NodeManagerAddress"):
                return node["NodeID"]
    except Exception as e:
        logging_utility.warning(f"Could not resolve Ray node ID: {e}")
    # Fallback to env-based ID
    return NODE_ID


# ─── INFERENCE CLUSTER LOGIC ────────────────────────────────────────────────


def _resolve_container_ip(container, network_name: str) -> Optional[str]:
    try:
        container.reload()
        networks = container.attrs.get("NetworkSettings", {}).get("Networks", {})

        if network_name in networks:
            ip = networks[network_name].get("IPAddress")
            if ip:
                logging_utility.info(f"🔍 Container {container.name} IP on '{network_name}': {ip}")
                return ip

        for net_name, net_cfg in networks.items():
            ip = net_cfg.get("IPAddress")
            if ip:
                logging_utility.warning(
                    f"⚠️  Network '{network_name}' not found; " f"using IP from '{net_name}': {ip}"
                )
                return ip

    except Exception as e:
        logging_utility.error(f"IP resolution failed for {container.name}: {e}")

    logging_utility.warning(
        f"⚠️  Could not resolve IP for {container.name}; " f"falling back to container name."
    )
    return container.name


def manage_vllm_container(deployment: InferenceDeployment, action: str = "start") -> Optional[str]:
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

    env = {
        "HF_TOKEN": os.getenv("HF_TOKEN"),
        "NVIDIA_DISABLE_REQUIRE": "true",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "NVIDIA_VISIBLE_DEVICES": "all",
    }

    cmd = (
        f"--model {deployment.base_model_id} "
        f"--dtype float16 --max-model-len 2048 --gpu-memory-utilization 0.5"
    )
    if deployment.fine_tuned_model_id:
        adapter_path = f"/mnt/training_data/{deployment.fine_tuned_model.storage_path}"
        cmd += f" --enable-lora --lora-modules {deployment.fine_tuned_model_id}={adapter_path}"

    try:
        all_containers = docker_client.containers.list(all=True)
        for c in all_containers:
            port_bindings = c.attrs.get("HostConfig", {}).get("PortBindings", {})
            if "8000/tcp" in port_bindings:
                bound_port = port_bindings["8000/tcp"][0].get("HostPort")
                if bound_port == str(deployment.port):
                    logging_utility.warning(
                        f"🧹 Port {deployment.port} is hogged by {c.name}. Evicting..."
                    )
                    c.remove(force=True)

        gpu_config = docker.types.DeviceRequest(
            count=-1, capabilities=[["gpu", "compute", "utility"]]
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
            ports={"8000/tcp": deployment.port},
            volumes={
                HF_CACHE_PATH: {"bind": "/root/.cache/huggingface", "mode": "rw"},
                SHARED_PATH: {"bind": "/mnt/training_data", "mode": "rw"},
            },
        )

        container_ip = _resolve_container_ip(container, DOCKER_NETWORK_NAME)
        logging_utility.info(
            f"✅ vLLM container {container_name} is up — routable IP: {container_ip}"
        )
        return container_ip

    except Exception as e:
        logging_utility.error(f"Docker spawn failed for {container_name}: {e}")
        return None


# ─── PHASE 3B: DEPLOYMENT SUPERVISOR RAY ACTOR ──────────────────────────────


@ray.remote
class DeploymentSupervisor:
    """
    Phase 3B: Persistent Ray actor replacing the supervisor thread.

    Phase 4 update: node_id is now the Ray node ID (hex string) written to
    inference_deployments for traceability, instead of the legacy NODE_ID
    env var / compute_nodes heartbeat value.

    SERIALIZATION: all dependencies imported locally inside run() to avoid
    pickling module-level objects with thread locks or weakrefs.
    """

    def __init__(self, node_id: str, docker_network: str, poll_interval: int = 20):
        self.node_id = node_id
        self.docker_network = docker_network
        self.poll_interval = poll_interval
        self._running = True

    def run(self):
        import os as _os
        import time as _time
        import docker as _docker
        from projectdavid_common import UtilsInterface as _UI
        from projectdavid_common.schemas.enums import StatusEnum as _Status
        from src.api.training.db.database import SessionLocal as _SessionLocal
        from src.api.training.models.models import InferenceDeployment as _Dep

        _log = _UI.LoggingUtility()
        _network = self.docker_network

        try:
            _dc = _docker.from_env()
        except Exception as e:
            _dc = None
            _log.error(f"Supervisor: Docker SDK init failed: {e}")

        def _resolve_ip(container):
            try:
                container.reload()
                nets = container.attrs.get("NetworkSettings", {}).get("Networks", {})
                if _network in nets:
                    ip = nets[_network].get("IPAddress")
                    if ip:
                        return ip
                for _, cfg in nets.items():
                    ip = cfg.get("IPAddress")
                    if ip:
                        return ip
            except Exception:
                pass
            return container.name

        def _spawn(dep):
            if not _dc:
                return None
            container_name = f"pd_vllm_{dep.id}"
            env = {
                "HF_TOKEN": _os.getenv("HF_TOKEN"),
                "NVIDIA_DISABLE_REQUIRE": "true",
                "PYTORCH_ALLOC_CONF": "expandable_segments:True",
                "NVIDIA_VISIBLE_DEVICES": "all",
            }
            cmd = (
                f"--model {dep.base_model_id} "
                f"--dtype float16 --max-model-len 2048 --gpu-memory-utilization 0.5"
            )
            if dep.fine_tuned_model_id:
                adapter_path = f"/mnt/training_data/{dep.fine_tuned_model.storage_path}"
                cmd += f" --enable-lora " f"--lora-modules {dep.fine_tuned_model_id}={adapter_path}"
            try:
                shared = _os.getenv("SHARED_PATH", "/mnt/training_data")
                hf = _os.getenv("HF_CACHE_PATH", "/root/.cache/huggingface")
                all_c = _dc.containers.list(all=True)
                for c in all_c:
                    pb = c.attrs.get("HostConfig", {}).get("PortBindings", {})
                    if "8000/tcp" in pb:
                        bp = pb["8000/tcp"][0].get("HostPort")
                        if bp == str(dep.port):
                            _log.warning(f"🧹 Port {dep.port} hogged by {c.name}. Evicting...")
                            c.remove(force=True)
                gpu = _docker.types.DeviceRequest(
                    count=-1, capabilities=[["gpu", "compute", "utility"]]
                )
                _log.info(f"🚢 Spawning vLLM: {container_name}")
                container = _dc.containers.run(
                    "vllm/vllm-openai:latest",
                    name=container_name,
                    command=cmd,
                    environment=env,
                    detach=True,
                    device_requests=[gpu],
                    runtime="nvidia",
                    network=_network,
                    ports={"8000/tcp": dep.port},
                    volumes={
                        hf: {"bind": "/root/.cache/huggingface", "mode": "rw"},
                        shared: {"bind": "/mnt/training_data", "mode": "rw"},
                    },
                )
                ip = _resolve_ip(container)
                _log.info(f"✅ {container_name} up — IP: {ip}")
                return ip
            except Exception as e:
                _log.error(f"Docker spawn failed for {container_name}: {e}")
                return None

        _log.info(f"👀 DeploymentSupervisor actor active for node: {self.node_id}")

        while self._running:
            db = _SessionLocal()
            try:
                deployments = (
                    db.query(_Dep)
                    .filter(
                        _Dep.node_id == self.node_id,
                        _Dep.status.in_([_Status.pending, _Status.active]),
                    )
                    .all()
                )
                for dep in deployments:
                    container_name = f"pd_vllm_{dep.id}"
                    is_running = False
                    try:
                        if _dc:
                            c = _dc.containers.get(container_name)
                            is_running = c.status == "running"
                    except Exception:
                        pass

                    if dep.status == _Status.pending or not is_running:
                        _log.warning(f"🚨 Deployment drift! Syncing {container_name}")
                        ip = _spawn(dep)
                        if ip:
                            dep.status = _Status.active
                            dep.internal_hostname = ip
                            db.commit()

            except Exception as e:
                _log.error(f"Supervisor Loop Error: {e}")
            finally:
                db.close()

            _time.sleep(self.poll_interval)

    def stop(self):
        self._running = False


# ─── PHASE 3A: TRAINING JOB AS RAY REMOTE TASK ──────────────────────────────


@ray.remote(num_gpus=1)
def process_job_remote(job_id: str, user_id: str):
    """
    Phase 3A: Ray remote task wrapping process_job with num_gpus=1.
    Ray reserves 1 GPU, tracks the task in the dashboard, and releases
    the reservation automatically on completion or failure.
    """
    process_job(job_id, user_id)


def process_job(job_id: str, user_id: str):
    """Core training job logic."""
    db: Session = SessionLocal()
    local_data_path = None

    try:
        job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        dataset = db.query(Dataset).filter(Dataset.id == job.dataset_id).first()

        logging_utility.info(f"🚀 Node {NODE_ID} claiming Training Job: {job_id}")
        job.status = StatusEnum.in_progress
        job.node_id = NODE_ID
        job.started_at = int(time.time())
        db.commit()

        storage = db.query(FileStorage).filter(FileStorage.file_id == dataset.file_id).first()
        local_data_path = os.path.join(LOCAL_SCRATCH, f"{job_id}.jsonl")
        smb = get_samba_client()
        smb.download_file(storage.storage_path, local_data_path)

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
            db.commit()
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
        if local_data_path and os.path.exists(local_data_path):
            os.remove(local_data_path)
        db.close()


# ─── MAIN ───────────────────────────────────────────────────────────────────


def main():
    # ── Phase 1: Ray cluster init ────────────────────────────────────────────
    ray_address = os.getenv("RAY_ADDRESS") or None
    ray.init(
        address=ray_address,
        ignore_reinit_error=True,
        include_dashboard=True,
        dashboard_host="0.0.0.0",
        dashboard_port=int(os.getenv("RAY_DASHBOARD_PORT", "8265")),
        logging_level="WARNING",
    )
    logging_utility.info(
        f"🌐 Ray cluster online — "
        f"dashboard: http://localhost:{os.getenv('RAY_DASHBOARD_PORT', '8265')}"
    )
    logging_utility.info(f"🔵 Ray resources: {ray.cluster_resources()}")

    # ── Phase 4: Resolve node identity from Ray ───────────────────────────────
    # Replaces the start_heartbeat() thread and node_heartbeat() DB call.
    # Ray already knows which nodes are alive — we read that state directly.
    # The Ray node ID (hex string) is written to inference_deployments for
    # traceability, consistent with what appears in the Ray dashboard.
    ray_node_id = get_ray_node_id()
    logging_utility.info(f"✅ Node identity resolved from Ray cluster: {ray_node_id}")

    # ── Phase 3B: DeploymentSupervisor Ray actor ──────────────────────────────
    # Pass the Ray node ID so the supervisor queries deployments by the
    # same identifier that gets written to inference_deployments.node_id.
    supervisor = DeploymentSupervisor.options(
        name="DeploymentSupervisor",
        get_if_exists=True,
    ).remote(
        node_id=ray_node_id,
        docker_network=DOCKER_NETWORK_NAME,
    )
    supervisor.run.remote()
    logging_utility.info("👀 DeploymentSupervisor Ray actor started.")

    # ── Phase 3A: Redis intake → Ray task submission ──────────────────────────
    r = get_redis()
    logging_utility.info(f"👷 Cluster Worker {ray_node_id} listening for jobs...")

    while True:
        try:
            result = r.brpop(QUEUE_NAME, timeout=30)
            if result:
                _, data = result
                payload = json.loads(data)
                if payload.get("target_node") and payload.get("target_node") != ray_node_id:
                    r.rpush(QUEUE_NAME, data)
                    time.sleep(1)
                    continue

                job_id = payload["job_id"]
                user_id = payload["user_id"]
                logging_utility.info(f"📬 Submitting job {job_id} to Ray cluster as remote task")
                process_job_remote.remote(job_id, user_id)

        except Exception as e:
            logging_utility.error(f"Critical Loop Error: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()
