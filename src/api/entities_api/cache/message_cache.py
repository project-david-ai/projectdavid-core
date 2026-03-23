# src/api/entities_api/cache/message_cache.py
import asyncio
import json
import os
from typing import Any, Dict, List, Union

from projectdavid_common.utilities.logging_service import LoggingUtility
from redis import Redis as SyncRedis

try:
    from redis.asyncio import Redis as AsyncRedis
except ImportError:

    class AsyncRedis:
        pass


LOG = LoggingUtility()
REDIS_HISTORY_TTL = int(os.getenv("REDIS_HISTORY_TTL_SECONDS", "3600"))


class MessageCache:
    """
    Redis-backed message history cache.

    CONTRACT — what the cache stores vs what the LLM receives:

    Redis stores LEAN messages:
        {"role": "user", "content": "...", "attachments": [{"type": "image", "file_id": "file_xxx"}]}

    The "attachments" key carries only file_id references — never base64 bytes.
    Image hydration (file_id → base64 Qwen array) happens in ContextMixin
    just before LLM dispatch via NativeExecutionService.hydrate_messages().

    This means:
      - Redis stays small regardless of how many images a thread has
      - Expired-file handling is always fresh (hydration at dispatch time)
      - No TTL mismatch between Redis and Samba file expiry
    """

    def __init__(self, redis: Union[SyncRedis, "AsyncRedis"]):
        self.redis = redis

    # ------------------------------------------------------------------
    # Lazy service accessors
    # ------------------------------------------------------------------

    @property
    def _message_svc(self):
        if not hasattr(self, "_message_svc_instance") or self._message_svc_instance is None:
            from src.api.entities_api.services.message_service import MessageService

            self._message_svc_instance = MessageService()
        return self._message_svc_instance

    @property
    def _native_exec(self):
        if not hasattr(self, "_native_exec_instance") or self._native_exec_instance is None:
            from src.api.entities_api.services.native_execution_service import (
                NativeExecutionService,
            )

            self._native_exec_instance = NativeExecutionService()
        return self._native_exec_instance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cache_key(self, thread_id: str) -> str:
        return f"thread:{thread_id}:history"

    # ------------------------------------------------------------------
    # Asynchronous Methods
    # ------------------------------------------------------------------

    async def get_history(self, thread_id: str) -> List[Dict]:
        """
        Retrieve lean message history from Redis, falling back to DB on a
        cache miss.

        Returns lean dicts — attachment file_ids preserved, no base64.
        Callers must hydrate images before LLM dispatch.
        """
        key = self._cache_key(thread_id)

        if isinstance(self.redis, AsyncRedis):
            raw_list = await self.redis.lrange(key, 0, -1)
        else:
            raw_list = await asyncio.to_thread(self.redis.lrange, key, 0, -1)

        if raw_list:
            return [json.loads(m) for m in raw_list]

        LOG.debug(f"[CACHE] Miss for thread {thread_id}. Performing async cold load.")

        try:
            full_hist = await self._native_exec.get_raw_messages(thread_id)
        except Exception as e:
            LOG.warning(
                "[CACHE] Async cold load failed for thread %s (%s). Returning empty history.",
                thread_id,
                e,
            )
            return []

        if full_hist:
            await self.set_history(thread_id, full_hist)

        return full_hist or []

    async def set_history(self, thread_id: str, messages: List[Dict]):
        """Overwrite / initialise the cache for a thread."""
        key = self._cache_key(thread_id)
        serialized = [json.dumps(m) for m in messages[-200:]]

        if isinstance(self.redis, AsyncRedis):
            async with self.redis.pipeline(transaction=True) as pipe:
                await pipe.delete(key)
                if serialized:
                    await pipe.rpush(key, *serialized)
                await pipe.expire(key, REDIS_HISTORY_TTL)
                await pipe.execute()
        else:
            await asyncio.to_thread(self.redis.delete, key)
            if serialized:
                await asyncio.to_thread(self.redis.rpush, key, *serialized)
            await asyncio.to_thread(self.redis.expire, key, REDIS_HISTORY_TTL)

    async def append_message(self, thread_id: str, message: Dict):
        key = self._cache_key(thread_id)
        data = json.dumps(message)

        if isinstance(self.redis, AsyncRedis):
            await self.redis.rpush(key, data)
            await self.redis.ltrim(key, -200, -1)
            await self.redis.expire(key, REDIS_HISTORY_TTL)
        else:
            await asyncio.to_thread(self.redis.rpush, key, data)
            await asyncio.to_thread(self.redis.ltrim, key, -200, -1)
            await asyncio.to_thread(self.redis.expire, key, REDIS_HISTORY_TTL)

    async def delete_history(self, thread_id: str):
        key = self._cache_key(thread_id)
        if isinstance(self.redis, AsyncRedis):
            await self.redis.delete(key)
        else:
            await asyncio.to_thread(self.redis.delete, key)

    # ------------------------------------------------------------------
    # Synchronous Methods
    # ------------------------------------------------------------------

    def get_history_sync(self, thread_id: str) -> List[Dict]:
        """
        Retrieve lean message history from Redis, falling back to DB on a
        cache miss.

        Returns lean dicts — attachment file_ids preserved, no base64.
        Callers must hydrate images before LLM dispatch.
        """
        if not isinstance(self.redis, SyncRedis):
            return []

        key = self._cache_key(thread_id)
        raw_list = self.redis.lrange(key, 0, -1)

        if raw_list:
            return [json.loads(m) for m in raw_list]

        LOG.debug(f"[CACHE-SYNC] Miss for thread {thread_id}. Performing sync cold load.")

        try:
            full_hist = self._message_svc.get_raw_messages_internal(thread_id)
            if full_hist:
                self.set_history_sync(thread_id, full_hist)
            return full_hist or []
        except Exception as e:
            LOG.warning(
                "[CACHE-SYNC] Cold load failed for thread %s (%s). Returning empty history.",
                thread_id,
                e,
            )
            return []

    def set_history_sync(self, thread_id: str, messages: List[Dict]):
        if isinstance(self.redis, SyncRedis):
            key = self._cache_key(thread_id)
            serialized = [json.dumps(m) for m in messages[-200:]]
            self.redis.delete(key)
            if serialized:
                self.redis.rpush(key, *serialized)
            self.redis.expire(key, REDIS_HISTORY_TTL)
        else:
            asyncio.run(self.set_history(thread_id, messages))

    def delete_history_sync(self, thread_id: str):
        key = self._cache_key(thread_id)
        if isinstance(self.redis, SyncRedis):
            self.redis.delete(key)
        else:
            asyncio.run(self.delete_history(thread_id))

    def append_message_sync(self, thread_id: str, message: Dict):
        if isinstance(self.redis, SyncRedis):
            key = self._cache_key(thread_id)
            data = json.dumps(message)
            self.redis.rpush(key, data)
            self.redis.ltrim(key, -200, -1)
            self.redis.expire(key, REDIS_HISTORY_TTL)
        else:
            asyncio.run(self.append_message(thread_id, message))


# ------------------------------------------------------------------
# Standalone factory
# ------------------------------------------------------------------
def get_sync_message_cache() -> MessageCache:
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    client = SyncRedis.from_url(redis_url, decode_responses=True)
    return MessageCache(redis=client)
