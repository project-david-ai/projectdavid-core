# src/api/entities_api/cache/training_cache.py
#
# Redis cache for the fine-tuning pipeline.
#
# Two responsibilities:
#
#   1. Job queue  — LPUSH/BRPOP pattern for the training worker.
#      The GPU is a shared resource so a single global queue is used.
#      The worker blocks on BRPOP, processes one job at a time, and
#      updates job status via the status cache on completion.
#
#   2. Status cache — lightweight poll target so the API can return
#      current job status without hitting the DB on every request.
#      The DB remains the source of truth — this is read-through only.
#
# Key schema:
#   training:queue                       — List  (global job queue)
#   training:job:{job_id}:status         — String (status payload JSON)
#
# TTL:
#   Queue entries have no TTL — they are consumed by the worker.
#   Status cache entries expire after REDIS_TRAINING_STATUS_TTL seconds.
#   If a status key is missing the caller falls back to the DB.

import asyncio
import json
import os
import time
from typing import Dict, List, Optional, Union

from projectdavid_common.utilities.logging_service import LoggingUtility
from redis import Redis as SyncRedis

try:
    from redis.asyncio import Redis as AsyncRedis
except ImportError:

    class AsyncRedis:
        pass


LOG = LoggingUtility()

REDIS_TRAINING_STATUS_TTL = int(os.getenv("REDIS_TRAINING_STATUS_TTL_SECONDS", "86400"))
TRAINING_QUEUE_KEY = "training:queue"


class TrainingCache:
    """
    Redis-backed queue and status cache for the fine-tuning pipeline.

    Queue contract:
      - Producer (API): LPUSH job envelope on job submission
      - Consumer (worker): BRPOP — blocks until a job is available,
        processes it, then calls set_job_status() with the result.

    Status contract:
      - Written by the worker on every status transition.
      - Read by the API on GET /v1/training-jobs/{job_id}.
      - Expires after REDIS_TRAINING_STATUS_TTL — callers fall back to DB.
    """

    def __init__(self, redis: Union[SyncRedis, "AsyncRedis"]):
        self.redis = redis

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    def _status_key(self, job_id: str) -> str:
        """Key for a job's cached status payload."""
        return f"training:job:{job_id}:status"

    # ------------------------------------------------------------------
    # Queue — producer side (called by TrainingService)
    # ------------------------------------------------------------------

    async def enqueue_job(self, job_id: str, user_id: str, config: dict) -> None:
        """
        Push a training job onto the global queue.

        Envelope schema:
          {
            "job_id":    str,
            "user_id":   str,
            "framework": str,   # axolotl | unsloth
            "config":    dict,  # full training config
            "enqueued_at": float
          }
        """
        envelope = {
            "job_id": job_id,
            "user_id": user_id,
            "framework": config.get("framework", "axolotl"),
            "config": config,
            "enqueued_at": time.time(),
        }
        data = json.dumps(envelope)

        if isinstance(self.redis, AsyncRedis):
            await self.redis.lpush(TRAINING_QUEUE_KEY, data)
        else:
            await asyncio.to_thread(self.redis.lpush, TRAINING_QUEUE_KEY, data)

        LOG.info("TrainingCache: enqueued job %s for user %s", job_id, user_id)

    async def queue_depth(self) -> int:
        """Return the number of jobs currently waiting in the queue."""
        if isinstance(self.redis, AsyncRedis):
            return await self.redis.llen(TRAINING_QUEUE_KEY)
        return await asyncio.to_thread(self.redis.llen, TRAINING_QUEUE_KEY)

    # ------------------------------------------------------------------
    # Queue — consumer side (called by TrainingWorker)
    # ------------------------------------------------------------------

    def dequeue_job_blocking(self, timeout: int = 30) -> Optional[Dict]:
        """
        Blocking pop from the queue. Intended for the training worker loop.

        Uses the sync Redis client — the worker runs in its own process/thread
        and does not share the async event loop with the API.

        Returns the job envelope dict, or None on timeout.

        Usage in worker:
            while True:
                job = cache.dequeue_job_blocking(timeout=30)
                if job:
                    run_training(job)
        """
        if not isinstance(self.redis, SyncRedis):
            raise RuntimeError(
                "dequeue_job_blocking requires a synchronous Redis client. "
                "Instantiate TrainingCache with redis.Redis (not redis.asyncio.Redis) "
                "in the worker process."
            )

        result = self.redis.brpop(TRAINING_QUEUE_KEY, timeout=timeout)
        if result is None:
            return None

        _, raw = result
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            LOG.error("TrainingCache: failed to decode job envelope: %s", e)
            return None

    # ------------------------------------------------------------------
    # Status cache — written by worker, read by API
    # ------------------------------------------------------------------

    async def set_job_status(
        self,
        job_id: str,
        status: str,
        *,
        started_at: Optional[float] = None,
        completed_at: Optional[float] = None,
        failed_at: Optional[float] = None,
        last_error: Optional[str] = None,
        metrics: Optional[Dict] = None,
        output_path: Optional[str] = None,
    ) -> None:
        """
        Write or update the status cache entry for a job.

        Called by the worker on every status transition:
          queued → in_progress → completed | failed
        """
        key = self._status_key(job_id)

        payload = {
            "job_id": job_id,
            "status": status,
            "updated_at": time.time(),
        }
        if started_at is not None:
            payload["started_at"] = started_at
        if completed_at is not None:
            payload["completed_at"] = completed_at
        if failed_at is not None:
            payload["failed_at"] = failed_at
        if last_error is not None:
            payload["last_error"] = last_error
        if metrics is not None:
            payload["metrics"] = metrics
        if output_path is not None:
            payload["output_path"] = output_path

        data = json.dumps(payload)

        if isinstance(self.redis, AsyncRedis):
            await self.redis.set(key, data, ex=REDIS_TRAINING_STATUS_TTL)
        else:
            await asyncio.to_thread(
                self.redis.set, key, data, ex=REDIS_TRAINING_STATUS_TTL
            )

        LOG.info("TrainingCache: job %s → %s", job_id, status)

    def set_job_status_sync(
        self,
        job_id: str,
        status: str,
        **kwargs,
    ) -> None:
        """
        Synchronous version for use in the worker process.
        Accepts the same keyword arguments as set_job_status().
        """
        key = self._status_key(job_id)

        payload = {
            "job_id": job_id,
            "status": status,
            "updated_at": time.time(),
            **{k: v for k, v in kwargs.items() if v is not None},
        }
        data = json.dumps(payload)

        if isinstance(self.redis, SyncRedis):
            self.redis.set(key, data, ex=REDIS_TRAINING_STATUS_TTL)
        else:
            asyncio.run(self.set_job_status(job_id, status, **kwargs))

    async def get_job_status(self, job_id: str) -> Optional[Dict]:
        """
        Retrieve the cached status for a job.

        Returns None on cache miss — the caller should fall back to the DB.
        """
        key = self._status_key(job_id)

        if isinstance(self.redis, AsyncRedis):
            raw = await self.redis.get(key)
        else:
            raw = await asyncio.to_thread(self.redis.get, key)

        if not raw:
            return None

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    async def delete_job_status(self, job_id: str) -> None:
        """Remove a job's status cache entry (e.g. on hard delete)."""
        key = self._status_key(job_id)
        if isinstance(self.redis, AsyncRedis):
            await self.redis.delete(key)
        else:
            await asyncio.to_thread(self.redis.delete, key)

    async def get_queue_snapshot(self) -> List[Dict]:
        """
        Return all jobs currently waiting in the queue without removing them.
        Useful for admin endpoints or debugging.
        """
        if isinstance(self.redis, AsyncRedis):
            raw_list = await self.redis.lrange(TRAINING_QUEUE_KEY, 0, -1)
        else:
            raw_list = await asyncio.to_thread(
                self.redis.lrange, TRAINING_QUEUE_KEY, 0, -1
            )

        result = []
        for raw in raw_list:
            try:
                result.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
        return result
