import json
import os
import socket
import subprocess  # nosec B404
import threading
import time
from typing import List, Optional

import ray
import redis
from projectdavid_common import UtilsInterface
from projectdavid_common.schemas.enums import StatusEnum
from projectdavid_common.utilities.identifier_service import IdentifierService
from projectdavid_orm.projectdavid_orm.models import FileStorage
from sqlalchemy.orm import Session

import docker
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
# — container-only path, not a shared host tmp
LOCAL_SCRATCH = "/tmp/training"  # nosec B108
SHARED_PATH = os.getenv("SHARED_PATH", "/mnt/training_data")
HF_CACHE_PATH = os.getenv("HF_CACHE_PATH", "/root/.cache/huggingface")

DOCKER_NETWORK_NAME = os.getenv(
    "DOCKER_NETWORK_NAME", "projectdavid-core_my_custom_network"
)

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
    """Returns the Ray node ID of the head node from the live cluster state."""
    try:
        nodes = ray.nodes()
        for node in nodes:
            if node.get("Alive") and node.get("NodeManagerAddress"):
                return node["NodeID"]
    except Exception as e:
        logging_utility.warning(f"Could not resolve Ray node ID: {e}")
    return NODE_ID


# ─── INFERENCE CLUSTER LOGIC ────────────────────────────────────────────────


def _resolve_container_ip(container, network_name: str) -> Optional[str]:
    try:
        container.reload()
        networks = container.attrs.get("NetworkSettings", {}).get("Networks", {})

        if network_name in networks:
            ip = networks[network_name].get("IPAddress")
            if ip:
                logging_utility.info(
                    f"🔍 Container {container.name} IP on '{network_name}': {ip}"
                )
                return ip

        for net_name, net_cfg in networks.items():
            ip = net_cfg.get("IPAddress")
            if ip:
                logging_utility.warning(
                    f"⚠️  Network '{network_name}' not found; "
                    f"using IP from '{net_name}': {ip}"
                )
                return ip

    except Exception as e:
        logging_utility.error(f"IP resolution failed for {container.name}: {e}")

    logging_utility.warning(
        f"⚠️  Could not resolve IP for {container.name}; "
        f"falling back to container name."
    )
    return container.name


def manage_vllm_container(
    deployment: InferenceDeployment, action: str = "start"
) -> Optional[str]:
    """
    Physically manages the vLLM container lifecycle on the host machine.

    Sharding: reads deployment.tensor_parallel_size and passes it to vLLM
    via --tensor-parallel-size. Default is 1 (single GPU, no sharding).
    When tensor_parallel_size > 1, vLLM splits the model across N GPUs
    on the same host using NCCL. Each GPU receives one shard of the
    model's layers (tensor parallelism).

    RAY_ADDRESS is forwarded to the container so that on worker nodes,
    vLLM joins the cluster Ray for cross-node tensor parallelism.
    On the head node RAY_ADDRESS is empty — vLLM uses its own isolated
    Ray instance (correct for single-node deployments).

    Returns:
        On 'start': the container's routable IP address.
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
            logging_utility.warning(
                f"⚠️  Could not stop container {container_name}: {e}"
            )
            return container_name

    # ── Resolve tensor parallel size from the deployment record ───────────
    tp_size = getattr(deployment, "tensor_parallel_size", 1) or 1

    env = {
        "HF_TOKEN": os.getenv("HF_TOKEN"),
        "NVIDIA_DISABLE_REQUIRE": "true",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "NVIDIA_VISIBLE_DEVICES": "all",
        # Forward RAY_ADDRESS so vLLM joins the cluster on worker nodes.
        # Empty on head node — vLLM runs isolated Ray (single-node sharding only).
        # Set on worker nodes — vLLM joins cluster Ray for cross-node sharding.
        "RAY_ADDRESS": os.getenv("RAY_ADDRESS", ""),
    }

    # ── Build vLLM command ────────────────────────────────────────────────
    cmd = (
        f"--model {deployment.base_model_id} "
        f"--dtype float16 "
        f"--max-model-len 2048 "
        f"--gpu-memory-utilization 0.5 "
        f"--tensor-parallel-size {tp_size}"
    )

    if deployment.fine_tuned_model_id:
        adapter_path = f"/mnt/training_data/{deployment.fine_tuned_model.storage_path}"
        cmd += f" --enable-lora --lora-modules {deployment.fine_tuned_model_id}={adapter_path}"

    logging_utility.info(
        f"🔀 Tensor parallel size: {tp_size} GPU(s) for {deployment.base_model_id}"
    )

    try:
        # ── AGGRESSIVE CLEANUP ────────────────────────────────────────────
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
            f"✅ vLLM container {container_name} is up — "
            f"routable IP: {container_ip} — tp={tp_size}"
        )
        return container_ip

    except Exception as e:
        logging_utility.error(f"Docker spawn failed for {container_name}: {e}")
        return None


# ─── PHASE 3B: DEPLOYMENT SUPERVISOR RAY ACTOR ──────────────────────────────


@ray.remote
class DeploymentSupervisor:
    """
    Persistent Ray actor that reconciles vLLM container state with the
    inference_deployments ledger every poll_interval seconds.

    Sharding: reads tensor_parallel_size from each deployment record and
    passes it through to _spawn(), which builds the correct vLLM command.

    RAY_ADDRESS is forwarded into each spawned vLLM container:
      - Head node  (RAY_ADDRESS=""):  vLLM starts isolated Ray — single-node only.
      - Worker node (RAY_ADDRESS set): vLLM joins cluster Ray — cross-node sharding.

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

        from projectdavid_common import UtilsInterface as _UI
        from projectdavid_common.schemas.enums import StatusEnum as _Status

        import docker as _docker
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

            tp_size = getattr(dep, "tensor_parallel_size", 1) or 1

            env = {
                "HF_TOKEN": _os.getenv("HF_TOKEN"),
                "NVIDIA_DISABLE_REQUIRE": "true",
                "PYTORCH_ALLOC_CONF": "expandable_segments:True",
                "NVIDIA_VISIBLE_DEVICES": "all",
                # Forward RAY_ADDRESS so vLLM joins the cluster on worker nodes.
                # Empty on head node — vLLM runs isolated Ray (single-node sharding only).
                # Set on worker nodes — vLLM joins cluster Ray for cross-node sharding.
                "RAY_ADDRESS": _os.getenv("RAY_ADDRESS", ""),
            }

            cmd = (
                f"--model {dep.base_model_id} "
                f"--dtype float16 "
                f"--max-model-len 2048 "
                f"--gpu-memory-utilization 0.5 "
                f"--tensor-parallel-size {tp_size}"
            )
            if dep.fine_tuned_model_id:
                adapter_path = f"/mnt/training_data/{dep.fine_tuned_model.storage_path}"
                cmd += (
                    f" --enable-lora "
                    f"--lora-modules {dep.fine_tuned_model_id}={adapter_path}"
                )
            try:
                shared = _os.getenv("SHARED_PATH", "/mnt/training_data")
                hf = _os.getenv("HF_CACHE_PATH", "/root/.cache/huggingface")
                all_c = _dc.containers.list(all=True)
                for c in all_c:
                    pb = c.attrs.get("HostConfig", {}).get("PortBindings", {})
                    if "8000/tcp" in pb:
                        bp = pb["8000/tcp"][0].get("HostPort")
                        if bp == str(dep.port):
                            _log.warning(
                                f"🧹 Port {dep.port} hogged by {c.name}. Evicting..."
                            )
                            c.remove(force=True)
                gpu = _docker.types.DeviceRequest(
                    count=-1, capabilities=[["gpu", "compute", "utility"]]
                )
                _log.info(f"🚢 Spawning vLLM: {container_name} (tp={tp_size})")
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
                _log.info(f"✅ {container_name} up — IP: {ip} — tp={tp_size}")
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
    """Ray remote task — reserves 1 GPU, tracks in dashboard."""
    process_job(job_id, user_id)


def process_job(job_id: str, user_id: str):
    """Core training job logic."""
    # All imports local to avoid Ray serialization failures —
    # SQLAlchemy engine contains thread locks that cannot be pickled.
    import os as _os
    import subprocess as _subprocess  # nosec B404
    import time as _time

    from projectdavid_common.schemas.enums import StatusEnum as _StatusEnum
    from projectdavid_common.utilities.identifier_service import (
        IdentifierService as _IdentifierService,
    )
    from projectdavid_orm.projectdavid_orm.models import FileStorage as _FileStorage
    from sqlalchemy.orm import Session

    from src.api.training.db.database import SessionLocal as _SessionLocal
    from src.api.training.models.models import Dataset as _Dataset
    from src.api.training.models.models import FineTunedModel as _FineTunedModel
    from src.api.training.models.models import TrainingJob as _TrainingJob
    from src.api.training.services.file_service import SambaClient as _SambaClient

    _NODE_ID = _os.getenv("NODE_ID", f"node_{__import__('socket').gethostname()}")
    _LOCAL_SCRATCH = "/tmp/training"  # nosec B108
    _SHARED_PATH = _os.getenv("SHARED_PATH", "/mnt/training_data")

    def _get_samba_client():
        return _SambaClient(
            server=_os.getenv("SMBCLIENT_SERVER", "samba"),
            share=_os.getenv("SMBCLIENT_SHARE", "cosmic_share"),
            username=_os.getenv("SMBCLIENT_USERNAME", "samba_user"),
            password=_os.getenv("SMBCLIENT_PASSWORD"),
        )

    db: Session = _SessionLocal()
    local_data_path = None
    job = None

    try:
        job = db.query(_TrainingJob).filter(_TrainingJob.id == job_id).first()
        dataset = db.query(_Dataset).filter(_Dataset.id == job.dataset_id).first()

        print(f"🚀 Node {_NODE_ID} claiming Training Job: {job_id}", flush=True)
        job.status = _StatusEnum.in_progress
        job.started_at = int(_time.time())
        db.commit()

        storage = (
            db.query(_FileStorage)
            .filter(_FileStorage.file_id == dataset.file_id)
            .first()
        )
        local_data_path = _os.path.join(_LOCAL_SCRATCH, f"{job_id}.jsonl")
        smb = _get_samba_client()
        smb.download_file(storage.storage_path, local_data_path)

        model_uuid = _IdentifierService.generate_prefixed_id("ftm")
        model_rel_path = f"models/{model_uuid}"
        full_output_path = _os.path.join(_SHARED_PATH, model_rel_path)
        _os.makedirs(full_output_path, exist_ok=True)

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
            _os.getenv("TRAINING_PROFILE", "laptop"),
        ]

        process = _subprocess.Popen(  # nosec B603
            cmd, stdout=_subprocess.PIPE, stderr=_subprocess.STDOUT, text=True
        )

        for line in process.stdout:
            print(f"[{job_id}] {line.strip()}", flush=True)
        process.wait()

        if process.returncode == 0:

            new_ftm = _FineTunedModel(
                id=model_uuid,
                user_id=user_id,
                training_job_id=job_id,
                name=f"FT: {job.base_model}",
                base_model=job.base_model,
                storage_path=model_rel_path,
                # node_id removed — FK references compute_nodes which is a legacy table
                status=_StatusEnum.active,
                created_at=int(_time.time()),
                updated_at=int(_time.time()),
            )

            db.add(new_ftm)
            job.status = _StatusEnum.completed
            job.completed_at = int(_time.time())
            job.output_path = model_rel_path
            db.commit()
            print(f"✨ Job {job_id} finalized successfully.", flush=True)
        else:
            raise _subprocess.CalledProcessError(process.returncode, cmd)

    except Exception as e:
        print(f"❌ Training Failure ({job_id}): {e}", flush=True)
        if job:
            job.status = _StatusEnum.failed
            job.last_error = str(e)
            db.commit()
    finally:
        if local_data_path and _os.path.exists(local_data_path):
            _os.remove(local_data_path)
        db.close()


# ─── MAIN ───────────────────────────────────────────────────────────────────


def main():
    # ── Phase 1: Ray cluster init ────────────────────────────────────────────
    # RAY_ADDRESS blank = this node is the head. It starts the cluster and
    # exposes the dashboard. Worker nodes set RAY_ADDRESS=ray://<head>:10001
    # and join the existing cluster without starting their own Ray instance.
    ray_address = os.getenv("RAY_ADDRESS") or None
    is_head = ray_address is None

    if is_head:
        ray.init(
            address=None,
            ignore_reinit_error=True,
            include_dashboard=True,
            # — Ray dashboard intentionally binds all interfaces inside container
            dashboard_host="0.0.0.0",  # nosec B104
            dashboard_port=int(os.getenv("RAY_DASHBOARD_PORT", "8265")),
            logging_level="WARNING",
        )
        logging_utility.info(
            f"🌐 Ray HEAD started — "
            f"dashboard: http://localhost:{os.getenv('RAY_DASHBOARD_PORT', '8265')}"
        )
    else:
        # Worker node — join existing cluster. No dashboard, no port conflicts.
        ray.init(
            address=ray_address,
            ignore_reinit_error=True,
            logging_level="WARNING",
        )
        logging_utility.info(f"🔗 Ray WORKER joined cluster at: {ray_address}")

    logging_utility.info(f"🔵 Ray resources: {ray.cluster_resources()}")

    # ── Phase 4: Resolve node identity from Ray ───────────────────────────────
    ray_node_id = get_ray_node_id()
    logging_utility.info(f"✅ Node identity resolved from Ray cluster: {ray_node_id}")

    # ── Phase 3B: DeploymentSupervisor Ray actor ──────────────────────────────
    # On the head node this actor is created fresh.
    # On worker nodes get_if_exists=True returns the already-running head actor
    # rather than spawning a duplicate — only one supervisor runs per cluster.
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
                if (
                    payload.get("target_node")
                    and payload.get("target_node") != ray_node_id
                ):
                    r.rpush(QUEUE_NAME, data)
                    time.sleep(1)
                    continue

                job_id = payload["job_id"]
                user_id = payload["user_id"]
                logging_utility.info(
                    f"📬 Submitting job {job_id} to Ray cluster as remote task"
                )
                process_job_remote.remote(job_id, user_id)

        except Exception as e:
            logging_utility.error(f"Critical Loop Error: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()
