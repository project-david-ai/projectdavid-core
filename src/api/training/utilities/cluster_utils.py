# src/api/training/utilities/cluster_utils.py
import logging
import os
import socket
import subprocess


def get_gpu_info():
    """Queries nvidia-smi for the local GPU profile."""
    try:
        # Query memory.total, gpu_name
        cmd = "nvidia-smi --query-gpu=memory.total,gpu_name --format=csv,nounits,noheader"
        output = subprocess.check_output(cmd, shell=True).decode().strip()
        total_mib, name = output.split(', ')
        return {"name": name, "total_gb": round(float(total_mib) / 1024, 2)}
    except Exception:
        return {"name": "CPU-Only / Unknown", "total_gb": 0.0}


def get_node_id():
    """Generates a unique ID for this physical machine."""
    hostname = socket.gethostname()
    # Can be overridden by env for cluster naming consistency
    return os.getenv("NODE_ID", f"node_{hostname}")
