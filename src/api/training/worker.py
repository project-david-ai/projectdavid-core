# docker/training/worker.py
#
# Long-running training worker for the projectdavid fine-tuning pipeline.
#
# Responsibilities:
#   - Block on Redis BRPOP for training jobs
#   - Spawn entrypoint.sh as a subprocess per job (GPU memory released between jobs)
#   - Update TrainingCache status on every transition
#   - Update the DB via the API's internal REST endpoint on completion/failure
#     so the TrainingJob record stays in sync
#
# One worker per GPU machine. Processes one job at a time — the GPU is a
# shared resource. Multi-GPU parallelism can be added later via multiple
# worker instances with a distributed lock.
#
# Environment variables:
#   REDIS_URL              — Redis connection string (default: redis://redis:6379/0)
#   ASSISTANTS_BASE_URL    — Internal API base URL for DB status updates
#   WORKER_API_KEY         — API key for internal status update calls (use ADMIN_API_KEY)
#   SHARED_PATH            — Host shared data path (mounted at /mnt/training_data)
#   WORKER_BRPOP_TIMEOUT   — Seconds to block on BRPOP before looping (default: 30)

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import redis

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [worker] %(message)s",
)
log = logging.getLogger("training_worker")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
ASSISTANTS_BASE_URL = os.getenv("ASSISTANTS_BASE_URL", "http://api:9000").rstrip("/")
WORKER_API_KEY = os.getenv("WORKER_API_KEY") or os.getenv("ADMIN_API_KEY", "")
SHARED_PATH = os.getenv("SHARED_PATH", "/mnt/training_data")
BRPOP_TIMEOUT = int(os.getenv("WORKER_BRPOP_TIMEOUT", "30"))

TRAINING_QUEUE_KEY = "training:queue"
TRAINING_STATUS_TTL = int(os.getenv("REDIS_TRAINING_STATUS_TTL_SECONDS", "86400"))

# ---------------------------------------------------------------------------
# Redis helpers
# ---------------------------------------------------------------------------


def get_redis() -> redis.Redis:
    return redis.Redis.from_url(REDIS_URL, decode_responses=True)


def set_status(r: redis.Redis, job_id: str, status: str, **kwargs) -> None:
    """Write job status to Redis cache."""
    key = f"training:job:{job_id}:status"
    payload = {
        "job_id": job_id,
        "status": status,
        "updated_at": time.time(),
        **{k: v for k, v in kwargs.items() if v is not None},
    }
    r.set(key, json.dumps(payload), ex=TRAINING_STATUS_TTL)
    log.info("Status: %s → %s", job_id, status)


# ---------------------------------------------------------------------------
# DB sync — notifies the API to update the TrainingJob record
# ---------------------------------------------------------------------------


def notify_api(job_id: str, payload: dict) -> None:
    """
    PATCH /v1/training-jobs/{job_id}/status (internal endpoint).
    Non-blocking — logs and continues on failure. Redis cache is the
    fast path; the API will reconcile from the cache on next poll.
    """
    if not WORKER_API_KEY:
        log.warning("WORKER_API_KEY not set — skipping API notification for %s", job_id)
        return

    url = f"{ASSISTANTS_BASE_URL}/v1/training-jobs/{job_id}/status"
    try:
        resp = httpx.patch(
            url,
            json=payload,
            headers={"X-API-Key": WORKER_API_KEY},
            timeout=10.0,
        )
        if resp.status_code not in (200, 204):
            log.warning(
                "API notification for %s returned %s: %s",
                job_id,
                resp.status_code,
                resp.text[:200],
            )
    except Exception as e:
        log.warning("API notification failed for %s: %s", job_id, e)


# ---------------------------------------------------------------------------
# Config file writer
# ---------------------------------------------------------------------------


def write_config(job: dict) -> str:
    """
    Write the training config YAML to the shared path so entrypoint.sh
    can pass it to Axolotl/Unsloth.

    Returns the absolute config path inside the container.
    """
    import yaml

    job_id = job["job_id"]
    config = job["config"]

    config_dir = Path(SHARED_PATH) / "configs" / job_id
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.yml"

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    log.info("Config written to: %s", config_path)
    return str(config_path)


# ---------------------------------------------------------------------------
# Job runner
# ---------------------------------------------------------------------------


def run_job(r: redis.Redis, job: dict) -> None:
    """
    Execute a single training job as a subprocess.

    The subprocess runs entrypoint.sh which calls Axolotl or Unsloth.
    GPU memory is held only for the duration of this call and released
    when the subprocess exits.
    """
    job_id = job["job_id"]
    user_id = job["user_id"]
    framework = job.get("framework", "axolotl")
    started_at = time.time()

    log.info("Starting job %s (user=%s framework=%s)", job_id, user_id, framework)

    # ── Mark in_progress ─────────────────────────────────────────────
    set_status(r, job_id, "in_progress", started_at=started_at)
    notify_api(job_id, {"status": "in_progress", "started_at": int(started_at)})

    # ── Write config to shared path ───────────────────────────────────
    try:
        config_path = write_config(job)
    except Exception as e:
        log.error("Failed to write config for job %s: %s", job_id, e)
        failed_at = time.time()
        set_status(r, job_id, "failed", failed_at=failed_at, last_error=str(e))
        notify_api(job_id, {"status": "failed", "failed_at": int(failed_at), "last_error": str(e)})
        return

    # ── Spawn training subprocess ─────────────────────────────────────
    cmd = ["/app/entrypoint.sh", "--framework", framework, "--config", config_path]
    log.info("Running: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=False,  # stream stdout/stderr to container logs
            text=True,
        )
        exit_code = result.returncode
    except Exception as e:
        log.error("Subprocess error for job %s: %s", job_id, e)
        exit_code = -1
        result = type("R", (), {"stderr": str(e)})()

    # ── Handle result ─────────────────────────────────────────────────
    output_path = str(Path(SHARED_PATH) / "checkpoints" / job_id)

    if exit_code == 0:
        completed_at = time.time()
        log.info("Job %s completed successfully.", job_id)
        set_status(
            r,
            job_id,
            "completed",
            completed_at=completed_at,
            output_path=output_path,
        )
        notify_api(
            job_id,
            {
                "status": "completed",
                "completed_at": int(completed_at),
                "output_path": output_path,
            },
        )
    else:
        failed_at = time.time()
        last_error = f"Training process exited with code {exit_code}"
        log.error("Job %s FAILED. %s", job_id, last_error)
        set_status(
            r,
            job_id,
            "failed",
            failed_at=failed_at,
            last_error=last_error,
        )
        notify_api(
            job_id,
            {
                "status": "failed",
                "failed_at": int(failed_at),
                "last_error": last_error,
            },
        )


# ---------------------------------------------------------------------------
# Main worker loop
# ---------------------------------------------------------------------------


def main() -> None:
    log.info("Training worker starting. Connecting to Redis: %s", REDIS_URL)

    r = get_redis()

    try:
        r.ping()
        log.info("Redis connection established.")
    except Exception as e:
        log.error("Cannot connect to Redis: %s", e)
        sys.exit(1)

    log.info(
        "Worker ready. Listening on queue '%s' (BRPOP timeout=%ds).",
        TRAINING_QUEUE_KEY,
        BRPOP_TIMEOUT,
    )

    while True:
        try:
            result = r.brpop(TRAINING_QUEUE_KEY, timeout=BRPOP_TIMEOUT)

            if result is None:
                # Timeout — no jobs, loop again
                log.debug("Queue empty, waiting...")
                continue

            _, raw = result

            try:
                job = json.loads(raw)
            except json.JSONDecodeError as e:
                log.error("Failed to decode job envelope: %s — skipping", e)
                continue

            log.info(
                "Dequeued job %s (user=%s framework=%s)",
                job.get("job_id"),
                job.get("user_id"),
                job.get("framework"),
            )

            run_job(r, job)

        except KeyboardInterrupt:
            log.info("Worker shutting down.")
            break
        except Exception as e:
            log.error("Unexpected error in worker loop: %s", e)
            time.sleep(5)  # brief back-off before retrying


if __name__ == "__main__":
    main()
