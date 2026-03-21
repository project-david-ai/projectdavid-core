import os
import subprocess
import time
from typing import Dict, List, Optional

from projectdavid_common.schemas.enums import StatusEnum
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.api.training.models.models import ComputeNode


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
    """
    Registers or updates the current node's status in the cluster registry.
    """
    telemetry = get_gpu_telemetry()

    # Check if node already exists
    node = db.query(ComputeNode).filter(ComputeNode.id == node_id).first()

    if not node:
        # First-time registration for this hardware
        import socket

        node = ComputeNode(
            id=node_id,
            hostname=socket.gethostname(),
            gpu_model=telemetry["gpu_model"],
            total_vram_gb=telemetry["total_vram"],
        )
        db.add(node)

    # Update live stats
    node.current_vram_usage_gb = telemetry["used_vram"]
    node.last_heartbeat = int(time.time())
    node.status = StatusEnum.active

    db.commit()


def select_best_node(db: Session, required_vram_gb: float = 4.0) -> Optional[str]:
    """
    Finds the active node with the most free VRAM.
    """
    # Nodes must have checked in within the last 60 seconds to be considered 'online'
    heartbeat_cutoff = int(time.time()) - 60

    best_node = (
        db.query(ComputeNode)
        .filter(
            ComputeNode.status == StatusEnum.active,
            ComputeNode.last_heartbeat > heartbeat_cutoff,
            (ComputeNode.total_vram_gb - ComputeNode.current_vram_usage_gb) >= required_vram_gb,
        )
        .order_by((ComputeNode.total_vram_gb - ComputeNode.current_vram_usage_gb).desc())
        .first()
    )

    return best_node.id if best_node else None
