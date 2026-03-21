# src/api/training/services/cluster_service.py
import os
import subprocess
import time
from typing import Dict, List, Optional

import redis
from projectdavid_common import UtilsInterface
from projectdavid_common.schemas.enums import StatusEnum
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from src.api.training.models.models import ComputeNode

logging_utility = UtilsInterface.LoggingUtility()


def get_gpu_telemetry():
    """
    Executes nvidia-smi to get live VRAM usage.
    """
    try:
        # Query total memory, used memory, and the GPU name
        cmd = (
            "nvidia-smi --query-gpu=memory.total,memory.used,gpu_name --format=csv,nounits,noheader"
        )
        output = subprocess.check_output(cmd, shell=True).decode().strip()
        total, used, name = output.split(', ')
        return {
            "total_vram": float(total) / 1024,  # Convert MiB to GB
            "used_vram": float(used) / 1024,
            "gpu_model": name,
        }
    except Exception:
        # Fallback for CPU-only or non-NVIDIA environments
        return {"total_vram": 0.0, "used_vram": 0.0, "gpu_model": "CPU/Unknown"}


def node_heartbeat(db: Session, node_id: str):
    telemetry = get_gpu_telemetry()
    node = db.query(ComputeNode).filter(ComputeNode.id == node_id).first()

    if not node:
        import socket

        node = ComputeNode(
            id=node_id,
            hostname=socket.gethostname(),
            # Store the vLLM service hostname — this is how other services
            # reach the inference endpoint within my_custom_network
            ip_address=os.getenv("VLLM_HOSTNAME", "vllm_server"),
            gpu_model=telemetry["gpu_model"],
            total_vram_gb=telemetry["total_vram"],
        )
        db.add(node)

    node.current_vram_usage_gb = telemetry["used_vram"]
    node.last_heartbeat = int(time.time())
    node.status = StatusEnum.active
    db.commit()


def select_best_node(db: Session, required_vram_gb: float = 4.0) -> Optional[str]:
    """
    SCHEDULER LOGIC:
    Uses 'Logical Free Space' (Total - Sum of Ledger Allocations)
    This is safer than nvidia-smi telemetry for rapid-fire requests.
    """
    from src.api.training.models.models import ComputeNode, GPUAllocation

    # 1. Get all active nodes
    nodes = db.query(ComputeNode).filter(ComputeNode.status == StatusEnum.active).all()

    best_node = None
    max_logical_free = -1.0

    for node in nodes:
        # 2. Calculate what we've ALREADY promised on this node
        reserved = (
            db.query(func.sum(GPUAllocation.vram_reserved_gb))
            .filter(GPUAllocation.node_id == node.id)
            .scalar()
            or 0.0
        )

        logical_free = node.total_vram_gb - reserved

        if logical_free >= required_vram_gb:
            if logical_free > max_logical_free:
                max_logical_free = logical_free
                best_node = node

    return best_node.id if best_node else None


def reap_stale_nodes(db: Session):
    """
    Scans for nodes that have stopped heartbeating and cleans up their resources.
    """
    # 1. Define thresholds
    # We allow 2 minutes of silence before marking a node as offline
    timeout_threshold = 120
    cutoff = int(time.time()) - timeout_threshold

    # 2. Find nodes that haven't checked in
    stale_nodes = (
        db.query(ComputeNode)
        .filter(ComputeNode.status == StatusEnum.active, ComputeNode.last_heartbeat < cutoff)
        .all()
    )

    if not stale_nodes:
        return

    stale_node_ids = [node.id for node in stale_nodes]
    logging_utility.warning(f"💀 Reaper: Found {len(stale_node_ids)} stale nodes: {stale_node_ids}")

    try:
        # 3. Fail jobs that were running on these nodes
        from src.api.training.models.models import TrainingJob

        abandoned_jobs = (
            db.query(TrainingJob)
            .filter(
                TrainingJob.node_id.in_(stale_node_ids),
                TrainingJob.status == StatusEnum.in_progress,
            )
            .all()
        )

        for job in abandoned_jobs:
            job.status = StatusEnum.failed
            job.last_error = "Node failure: Heartbeat lost during training."
            logging_utility.info(f"🚩 Reaper: Marked job {job.id} as failed.")

        # 4. Clear the VRAM Ledger for these nodes
        from src.api.training.models.models import GPUAllocation

        db.query(GPUAllocation).filter(GPUAllocation.node_id.in_(stale_node_ids)).delete(
            synchronize_session=False
        )
        logging_utility.info(f"🧹 Reaper: Cleared VRAM allocations for stale nodes.")

        # 5. Mark nodes as offline
        for node in stale_nodes:
            node.status = StatusEnum.offline

        db.commit()
        logging_utility.info(f"✅ Reaper: Cluster integrity restored.")

    except Exception as e:
        db.rollback()
        logging_utility.error(f"❌ Reaper Error: {e}")


def acquire_api_lease(r, instance_id: str, timeout: int = 30) -> bool:
    """SET NX: Only succeeds if the key doesn't exist."""
    return r.set("cluster:active_training_api", instance_id, ex=timeout, nx=True)


def renew_api_lease(r, instance_id: str, timeout: int = 30) -> bool:
    """Atomic Check-and-Renew using a Lua script snippet."""
    script = """
    if redis.call("get", KEYS[1]) == ARGV[1] then
        return redis.call("expire", KEYS[1], ARGV[2])
    else
        return 0
    end
    """
    return r.eval(script, 1, "cluster:active_training_api", instance_id, timeout)
