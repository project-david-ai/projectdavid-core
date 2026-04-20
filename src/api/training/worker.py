"""
worker.py

Sovereign Forge — Training Worker

Responsibilities:
  - Listen to the Redis training job queue
  - Dispatch training jobs as direct subprocesses (GPU claimed for duration)
  - Parse PROGRESS: lines from unsloth_train.py and write live metrics
    to job.metrics so users get feedback during training
  - Honour cancellation signals (DB status + Redis fast-path) and cleanly
    unwind subprocess + discard partial adapter artifacts

Ray is NOT used here. inference_worker owns the Ray cluster (HEAD node)
and manages all inference GPU reservations via Ray Serve.

Cancellation model:
    API flips DB status to CANCELLING and sets Redis key `cancel:job:{id}`.
    Worker polls the Redis key between stdout reads. On cancel:
        1. SIGTERM the subprocess (process group — kills children too)
        2. Wait up to SIGTERM_GRACE_SECONDS for clean exit
        3. SIGKILL if still alive
        4. Delete partial adapter output directory
        5. Flip DB status CANCELLING → CANCELLED
"""

import json
import os
import queue
import shutil
import signal
import socket
import threading
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

# Cancellation signaling — must match src/api/training/services/training_service.py
CANCEL_KEY_PREFIX = "cancel:job:"
SIGTERM_GRACE_SECONDS = 30  # Wait up to this long for graceful subprocess exit
DB_CANCEL_POLL_INTERVAL = 10  # How often to fall back to DB status check (seconds)

os.makedirs(LOCAL_SCRATCH, exist_ok=True)


# ─── INFRASTRUCTURE HELPERS ─────────────────────────────────────────────────


def get_redis():
    return redis.from_url(REDIS_URL, decode_responses=True)


def _cancel_key(job_id: str) -> str:
    return f"{CANCEL_KEY_PREFIX}{job_id}"


# ─── TRAINING JOB ───────────────────────────────────────────────────────────


def process_job(job_id: str, user_id: str):
    """
    Core training job logic — runs as a direct subprocess.

    Progress feedback:
        unsloth_train.py emits PROGRESS:{...} lines on every logging step.
        The read loop parses these and pushes metrics dicts to a queue.
        A background writer thread drains the queue and commits to the DB.

    Cancellation:
        Between every stdout line read, the worker checks Redis for a cancel
        signal. A periodic DB status check (every DB_CANCEL_POLL_INTERVAL
        seconds) provides a backstop if Redis is unavailable.
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
    _r = get_redis()

    def _get_samba_client():
        return _SambaClient(
            server=_os.getenv("SMBCLIENT_SERVER", "samba"),
            share=_os.getenv("SMBCLIENT_SHARE", "cosmic_share"),
            username=_os.getenv("SMBCLIENT_USERNAME", "samba_user"),
            password=_os.getenv("SMBCLIENT_PASSWORD"),
        )

    def _is_cancel_requested(job_id: str) -> bool:
        try:
            return _r.get(_cancel_key(job_id)) is not None
        except Exception as e:
            logging_utility.warning(
                f"[{job_id}] Redis cancel check failed (will fall back to DB): {e}"
            )
            return False

    def _db_confirms_cancel(job_id: str) -> bool:
        _db = _SessionLocal()
        try:
            _job = _db.query(_TrainingJob).filter(_TrainingJob.id == job_id).first()
            if not _job:
                return False
            return _job.status in (_StatusEnum.cancelling, _StatusEnum.cancelled)
        except Exception as e:
            logging_utility.warning(f"[{job_id}] DB cancel check failed: {e}")
            return False
        finally:
            _db.close()

    def _terminate_subprocess(process, job_id: str) -> None:
        if process.poll() is not None:
            return

        try:
            pgid = _os.getpgid(process.pid)
            _os.killpg(pgid, signal.SIGTERM)
            logging_utility.info(
                f"[{job_id}] SIGTERM sent to process group {pgid}, "
                f"waiting up to {SIGTERM_GRACE_SECONDS}s for graceful exit"
            )
        except ProcessLookupError:
            logging_utility.info(f"[{job_id}] Process already exited before SIGTERM")
            return
        except Exception as e:
            logging_utility.warning(f"[{job_id}] SIGTERM send failed: {e}")

        try:
            process.wait(timeout=SIGTERM_GRACE_SECONDS)
            logging_utility.info(f"[{job_id}] Subprocess exited cleanly after SIGTERM")
        except _subprocess.TimeoutExpired:
            logging_utility.warning(
                f"[{job_id}] SIGTERM grace period expired — escalating to SIGKILL"
            )
            try:
                pgid = _os.getpgid(process.pid)
                _os.killpg(pgid, signal.SIGKILL)
                process.wait(timeout=5)
                logging_utility.info(f"[{job_id}] Subprocess SIGKILLed")
            except Exception as e:
                logging_utility.error(
                    f"[{job_id}] SIGKILL failed — subprocess may be orphaned: {e}"
                )

    def _cleanup_partial_artifact(output_path: str, job_id: str) -> None:
        if not output_path or not _os.path.exists(output_path):
            return
        try:
            shutil.rmtree(output_path)
            logging_utility.info(
                f"[{job_id}] Deleted partial adapter output: {output_path}"
            )
        except Exception as e:
            logging_utility.warning(
                f"[{job_id}] Could not delete partial adapter {output_path}: {e}"
            )

    def _finalize_cancelled(job_id: str) -> None:
        _db = _SessionLocal()
        try:
            _job = _db.query(_TrainingJob).filter(_TrainingJob.id == job_id).first()
            if _job:
                _job.status = _StatusEnum.cancelled
                if not _job.cancelled_at:
                    _job.cancelled_at = int(_time.time())
                _job.updated_at = int(_time.time())
                _db.commit()
                logging_utility.info(f"[{job_id}] Job finalized as CANCELLED")
        except Exception as e:
            logging_utility.error(
                f"[{job_id}] Failed to finalize cancelled status: {e}"
            )
        finally:
            _db.close()

        try:
            _r.delete(_cancel_key(job_id))
        except Exception:
            pass

    db: Session = _SessionLocal()
    local_data_path = None
    full_output_path = None
    job = None

    try:
        job = db.query(_TrainingJob).filter(_TrainingJob.id == job_id).first()

        # Pre-start cancel check — handles cancelled-while-queued
        if job and job.status in (_StatusEnum.cancelled, _StatusEnum.cancelling):
            logging_utility.info(
                f"[{job_id}] Skipping — job was cancelled before execution "
                f"(status={job.status.value})"
            )
            _finalize_cancelled(job_id)
            return

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
            "-u",
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
            cmd,
            stdout=_subprocess.PIPE,
            stderr=_subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )

        # ─── STDOUT LOOP WITH PROGRESS PARSING (THREADED DB WRITES) ──────────
        metrics_queue: queue.Queue = queue.Queue()
        writer_stop = threading.Event()
        write_counter = {"attempted": 0, "succeeded": 0, "failed": 0}

        def _metrics_writer():
            while not writer_stop.is_set() or not metrics_queue.empty():
                try:
                    m = metrics_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                write_counter["attempted"] += 1
                t_start = _time.time()
                _db = _SessionLocal()
                try:
                    _job = (
                        _db.query(_TrainingJob)
                        .filter(_TrainingJob.id == job_id)
                        .first()
                    )
                    if _job:
                        _job.metrics = m
                        _job.updated_at = int(_time.time())
                        _db.commit()
                        write_counter["succeeded"] += 1
                        logging_utility.info(
                            f"[{job_id}] DB_WRITE_OK step={m.get('step')} "
                            f"elapsed={(_time.time() - t_start) * 1000:.1f}ms "
                            f"attempted={write_counter['attempted']} "
                            f"succeeded={write_counter['succeeded']}"
                        )
                    else:
                        write_counter["failed"] += 1
                        logging_utility.warning(
                            f"[{job_id}] DB_WRITE_MISS — TrainingJob not found"
                        )
                except Exception as write_err:
                    write_counter["failed"] += 1
                    logging_utility.warning(
                        f"[{job_id}] DB_WRITE_FAIL step={m.get('step')} "
                        f"err={write_err}"
                    )
                finally:
                    _db.close()
                    metrics_queue.task_done()

        writer_thread = threading.Thread(
            target=_metrics_writer, daemon=True, name=f"metrics_writer_{job_id}"
        )
        writer_thread.start()
        logging_utility.info(f"[{job_id}] Metrics writer thread started")

        cancel_detected = False
        last_db_cancel_check = _time.time()

        try:
            for line in process.stdout:
                line = line.strip()
                logging_utility.info(f"[{job_id}] {line}")

                if line.startswith("PROGRESS:"):
                    try:
                        metrics = json.loads(line[9:])
                        metrics_queue.put(metrics)
                        logging_utility.info(
                            f"[{job_id}] QUEUED step={metrics.get('step')} "
                            f"qsize={metrics_queue.qsize()}"
                        )
                    except Exception as parse_err:
                        logging_utility.warning(
                            f"[{job_id}] Failed to parse PROGRESS line: {parse_err}"
                        )

                # Cancel check — fast path (Redis) + periodic slow path (DB)
                if _is_cancel_requested(job_id):
                    cancel_detected = True
                    logging_utility.info(f"[{job_id}] Cancel signal detected via Redis")
                    break

                now = _time.time()
                if now - last_db_cancel_check >= DB_CANCEL_POLL_INTERVAL:
                    last_db_cancel_check = now
                    if _db_confirms_cancel(job_id):
                        cancel_detected = True
                        logging_utility.info(
                            f"[{job_id}] Cancel signal detected via DB backstop"
                        )
                        break

        finally:
            logging_utility.info(
                f"[{job_id}] Stopping metrics writer — "
                f"attempted={write_counter['attempted']} "
                f"succeeded={write_counter['succeeded']} "
                f"failed={write_counter['failed']} "
                f"qsize={metrics_queue.qsize()}"
            )
            writer_stop.set()
            writer_thread.join(timeout=10)
            logging_utility.info(
                f"[{job_id}] Metrics writer stopped — "
                f"final attempted={write_counter['attempted']} "
                f"succeeded={write_counter['succeeded']} "
                f"failed={write_counter['failed']}"
            )

        # Cancel path
        if cancel_detected:
            logging_utility.info(f"[{job_id}] Cancelling subprocess")
            _terminate_subprocess(process, job_id)
            _cleanup_partial_artifact(full_output_path, job_id)
            _finalize_cancelled(job_id)
            return

        # Normal completion path
        process.wait()

        # Late-cancel race check
        if _db_confirms_cancel(job_id):
            logging_utility.info(
                f"[{job_id}] Late cancel detected post-subprocess — "
                f"discarding artifact"
            )
            _cleanup_partial_artifact(full_output_path, job_id)
            _finalize_cancelled(job_id)
            return

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
            job.failed_at = int(_time.time())
            db.commit()
        if full_output_path:
            _cleanup_partial_artifact(full_output_path, job_id)
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
