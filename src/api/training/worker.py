import json
import os
import subprocess
import time
from typing import List, Optional

import redis
from projectdavid_common import UtilsInterface
from projectdavid_common.schemas.enums import StatusEnum
from projectdavid_common.utilities.identifier_service import IdentifierService
from projectdavid_orm.projectdavid_orm.models import FileStorage
from sqlalchemy.orm import Session

from src.api.training.db.database import SessionLocal
from src.api.training.models.models import Dataset, FineTunedModel, TrainingJob
from src.api.training.services.file_service import SambaClient

logging_utility = UtilsInterface.LoggingUtility()

# --- Configuration ---
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
QUEUE_NAME = "training_jobs"
LOCAL_SCRATCH = "/tmp/training"
SHARED_PATH = os.getenv("SHARED_PATH", "/mnt/training_data")

os.makedirs(LOCAL_SCRATCH, exist_ok=True)


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


def process_job(job_id: str, user_id: str):
    db: Session = SessionLocal()
    local_data_path = None  # FIX: Pre-initialize to prevent UnboundLocalError in finally block

    try:
        # 1. Fetch metadata
        job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        if not job:
            logging_utility.error(f"Worker Error: Job {job_id} not found in database.")
            return

        dataset = db.query(Dataset).filter(Dataset.id == job.dataset_id).first()
        if not dataset:
            raise ValueError(f"Dataset reference {job.dataset_id} not found in database.")

        # 2. Update Status -> In Progress
        logging_utility.info(f"🚀 Worker taking Job: {job_id} (User: {user_id})")
        job.status = StatusEnum.in_progress
        job.started_at = int(time.time())
        db.commit()

        # 3. Stage Data (Samba -> Local NVMe Scratch)
        # Query shared FileStorage table owned by core API
        storage = db.query(FileStorage).filter(FileStorage.file_id == dataset.file_id).first()
        if not storage or not storage.storage_path:
            raise ValueError(f"No storage path found for file_id {dataset.file_id}")

        local_data_path = os.path.join(LOCAL_SCRATCH, f"{job_id}.jsonl")
        smb = get_samba_client()

        logging_utility.info(f"Staging dataset from Samba: {storage.storage_path}")
        smb.download_file(storage.storage_path, local_data_path)

        # 4. Prepare Output Artifact Path
        model_uuid = IdentifierService.generate_prefixed_id("ftm")
        model_rel_path = f"models/{model_uuid}"
        full_output_path = os.path.join(SHARED_PATH, model_rel_path)
        os.makedirs(full_output_path, exist_ok=True)

        # 1. Read the profile from the container's environment variables
        # Default to 'standard' if not set, but on your machine, we'll set it to 'laptop'
        training_profile = os.getenv("TRAINING_PROFILE", "standard")

        # 5. Spawn Training Subprocess
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
            training_profile,  # <--- Dynamically passed
        ]
        logging_utility.info(f"Spawning training subprocess with profile: {training_profile}")
        logging_utility.info(f"Spawning training subprocess: {' '.join(cmd)}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        for line in process.stdout:
            # Print directly to Docker logs for real-time visibility
            print(f"[{job_id}] {line.strip()}", flush=True)

        process.wait()

        if process.returncode == 0:
            # 6. Success: Register FineTunedModel
            logging_utility.info(f"✅ Subprocess finished. Registering model {model_uuid}...")

            new_ftm = FineTunedModel(
                id=model_uuid,
                user_id=user_id,
                training_job_id=job_id,
                name=f"FT: {job.base_model}",
                base_model=job.base_model,
                storage_path=model_rel_path,
                status=StatusEnum.active,
                is_active=False,
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
        error_msg = str(e)
        logging_utility.error(f"❌ Worker Job Failure ({job_id}): {error_msg}")
        # Re-fetch job in case of session issues
        try:
            if job:
                job.status = StatusEnum.failed
                job.failed_at = int(time.time())
                job.last_error = error_msg
                db.commit()
        except Exception as commit_err:
            logging_utility.error(f"Failed to save failure status for {job_id}: {commit_err}")

    finally:
        # 7. Cleanup local scratch file and close session
        if local_data_path and os.path.exists(local_data_path):
            try:
                os.remove(local_data_path)
            except Exception as cleanup_err:
                logging_utility.warning(f"Failed to cleanup {local_data_path}: {cleanup_err}")
        db.close()


def main():
    r = get_redis()
    logging_utility.info(f"👷 Training Worker ready. Listening on queue: '{QUEUE_NAME}'")

    while True:
        try:
            # BRPOP blocks until an item is available, returns (key, value)
            result = r.brpop(QUEUE_NAME, timeout=30)
            if result:
                _, data = result
                payload = json.loads(data)
                process_job(payload["job_id"], payload["user_id"])
        except Exception as e:
            logging_utility.error(f"Worker Loop Critical Error: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()
