"""
worker.py

Sovereign Forge — Training Worker

Responsibilities:
  - Listen to the Redis training job queue
  - Dispatch training jobs as direct subprocesses (GPU claimed for duration)
  - Parse PROGRESS: lines from unsloth_train.py and write live metrics
    to job.metrics so users get feedback during training

Ray is NOT used here. inference_worker owns the Ray cluster (HEAD node)
and manages all inference GPU reservations via Ray Serve.

GPU contention between training and inference is managed by policy on
single-GPU machines — the operator activates inference OR submits training,
not both simultaneously. On multi-GPU machines this is a non-issue.

For multi-node clusters: additional compute nodes join the Ray cluster
via ray://inference_worker:10001 and contribute their GPUs to Ray Serve.
Training jobs on those nodes also use this same pure-subprocess approach.
"""

import json
import os
import socket
import time

import redis
from projectdavid_common import UtilsInterface

logging_utility = UtilsInterface.LoggingUtility()

# ─── CONFIGURATION ──────────────────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
QUEUE_NAME = "training_jobs"
LOCAL_SCRATCH = "/tmp/training"  # nosec B108
SHARED_PATH = os.getenv("SHARED_PATH", "/mnt/training_data")
HF_CACHE_PATH = os.getenv("HF_CACHE_PATH", "/root/.cache/huggingface")
NODE_ID = os.getenv("NODE_ID", f"node_{socket.gethostname()}")

os.makedirs(LOCAL_SCRATCH, exist_ok=True)


# ─── INFRASTRUCTURE HELPERS ─────────────────────────────────────────────────


def get_redis():
    return redis.from_url(REDIS_URL, decode_responses=True)


# ─── TRAINING JOB ───────────────────────────────────────────────────────────


def process_job(job_id: str, user_id: str):
    """
    Core training job logic — runs as a direct subprocess.

    Progress feedback:
        unsloth_train.py emits PROGRESS:{...} lines on every logging step.
        This loop parses those lines and writes them to job.metrics so users
        polling client.training.retrieve(job_id) get live loss and step count
        rather than a black hole between dispatch and completion.

    All imports are local to keep the module lightweight and avoid
    import-time side effects from SQLAlchemy / ORM modules.
    """
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

        logging_utility.info(f"🚀 Node {_NODE_ID} claiming Training Job: {job_id}")
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

        # ─── STDOUT LOOP WITH PROGRESS PARSING ───────────────────────────────
        # unsloth_train.py emits PROGRESS:{...} lines on every logging step.
        # We parse these and write them to job.metrics so polling clients
        # get live feedback. Non-PROGRESS lines are logged normally.
        for line in process.stdout:
            line = line.strip()
            logging_utility.info(f"[{job_id}] {line}")

            if line.startswith("PROGRESS:"):
                try:
                    metrics = json.loads(line[9:])
                    job.metrics = metrics
                    job.updated_at = int(_time.time())
                    db.commit()
                except Exception as parse_err:
                    logging_utility.warning(
                        f"[{job_id}] Failed to parse PROGRESS line: {parse_err}"
                    )
        # ─────────────────────────────────────────────────────────────────────

        process.wait()

        if process.returncode == 0:
            new_ftm = _FineTunedModel(
                id=model_uuid,
                user_id=user_id,
                training_job_id=job_id,
                name=f"FT: {job.base_model}",
                base_model=job.base_model,
                storage_path=model_rel_path,
                status=_StatusEnum.active,
                created_at=int(_time.time()),
                updated_at=int(_time.time()),
            )

            db.add(new_ftm)
            job.status = _StatusEnum.completed
            job.completed_at = int(_time.time())
            job.output_path = model_rel_path
            db.commit()
            logging_utility.info(f"✨ Job {job_id} finalized successfully.")
        else:
            raise _subprocess.CalledProcessError(process.returncode, cmd)

    except Exception as e:
        logging_utility.error(f"❌ Training Failure ({job_id}): {e}")
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
    r = get_redis()
    logging_utility.info(
        f"👷 Training Worker {NODE_ID} listening for jobs on queue: {QUEUE_NAME}"
    )

    while True:
        try:
            result = r.brpop(QUEUE_NAME, timeout=30)
            if result:
                _, data = result
                payload = json.loads(data)

                # Target node routing — if job specifies a different node,
                # re-queue and let that node pick it up.
                if payload.get("target_node") and payload.get("target_node") != NODE_ID:
                    r.rpush(QUEUE_NAME, data)
                    time.sleep(1)
                    continue

                job_id = payload["job_id"]
                user_id = payload["user_id"]
                logging_utility.info(
                    f"📬 Processing training job {job_id} on node {NODE_ID}"
                )
                process_job(job_id, user_id)

        except Exception as e:
            logging_utility.error(f"Critical Loop Error: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()
