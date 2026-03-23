import asyncio
import json
import os
from typing import Any, Dict, Union

from redis import Redis as SyncRedis

try:
    from redis.asyncio import Redis as AsyncRedis
except ImportError:

    class AsyncRedis:
        pass


from projectdavid_common.utilities.logging_service import LoggingUtility

LOG = LoggingUtility()

REDIS_ASSISTANT_TTL = int(os.getenv("REDIS_ASSISTANT_TTL_SECONDS", "300"))


class AssistantCache:

    def __init__(self, redis: Union[SyncRedis, "AsyncRedis"]):
        """
        pd_base_url / pd_api_key removed — assistant data is now fetched
        directly via NativeExecutionService, eliminating the internal HTTP
        round-trip and any API-key dependency.
        """
        self.redis = redis
        self._native_exec_svc = None  # lazy-initialised on first access

    @property
    def _native_exec(self):
        # Mirrors the lazy-load pattern used across mixin classes so
        # NativeExecutionService is only instantiated when actually needed,
        # and the deferred import keeps circular dependencies at bay.
        if self._native_exec_svc is None:
            from src.api.entities_api.services.native_execution_service import \
                NativeExecutionService

            self._native_exec_svc = NativeExecutionService()
        return self._native_exec_svc

    def _cache_key(self, assistant_id: str) -> str:
        return f"assistant:{assistant_id}:config"

    async def get(self, assistant_id: str):
        key = self._cache_key(assistant_id)
        if isinstance(self.redis, AsyncRedis):
            raw = await self.redis.get(key)
        else:
            raw = await asyncio.to_thread(self.redis.get, key)
        return json.loads(raw) if raw else None

    async def set(self, assistant_id: str, payload: dict):
        key = self._cache_key(assistant_id)
        data = json.dumps(payload)
        if isinstance(self.redis, AsyncRedis):
            await self.redis.set(key, data, ex=REDIS_ASSISTANT_TTL)
        else:
            await asyncio.to_thread(self.redis.set, key, data, ex=REDIS_ASSISTANT_TTL)

    def _normalize_bool(self, value: Union[str, bool, int]) -> bool:
        """Helper to safely convert metadata strings/ints to boolean."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        if isinstance(value, int):
            return value == 1
        return False

    async def retrieve(self, assistant_id: str) -> Dict[str, Any]:
        """
        Fetches assistant details and caches them.

        Previously used Entity (SDK) for an internal HTTP round-trip.
        Now delegates to NativeExecutionService.retrieve_assistant which
        calls AssistantService → DB directly. No network hop, no API key,
        no ownership concern (read-only config fetch).
        """
        # 1. Cache hit
        cached = await self.get(assistant_id)
        if cached:
            return cached

        # 2. Fetch directly from DB via NativeExecutionService
        assistant = await self._native_exec.retrieve_assistant(assistant_id)

        # 3. Extract Tools (AssistantRead already returns a plain list)
        raw_tools = getattr(assistant, "tools", []) or []

        clean_tools = []
        for t in raw_tools:
            if isinstance(t, dict):
                clean_tools.append(t)
            elif hasattr(t, "model_dump"):
                clean_tools.append(t.model_dump())
            elif hasattr(t, "dict"):
                clean_tools.append(t.dict())
            else:
                clean_tools.append(dict(t))

        # 4. Metadata
        raw_meta = getattr(assistant, "meta_data", {}) or {}

        is_research_worker = self._normalize_bool(raw_meta.get("research_worker_calling", False))
        is_junior_engineer = self._normalize_bool(raw_meta.get("junior_engineer_calling", False))

        # 5. Tool resources
        raw_tool_resources = getattr(assistant, "tool_resources", {}) or {}

        # 6. Build payload
        payload = {
            "instructions": assistant.instructions,
            "tools": clean_tools,
            "tool_resources": raw_tool_resources,
            "agent_mode": getattr(assistant, "agent_mode", False),
            "decision_telemetry": getattr(assistant, "decision_telemetry", False),
            "web_access": getattr(assistant, "web_access", False),
            "deep_research": getattr(assistant, "deep_research", False),
            "is_engineer": getattr(assistant, "engineer", False),
            "meta_data": raw_meta,
            "is_research_worker": is_research_worker,
            "junior_engineer": is_junior_engineer,
        }

        # 7. Populate cache
        await self.set(assistant_id, payload)
        return payload

    async def store(self, assistant_id: str, payload: dict):
        """Alias used by OrchestratorCore._ensure_config_loaded rehydration path."""
        await self.set(assistant_id, payload)

    async def delete(self, assistant_id: str):
        key = self._cache_key(assistant_id)
        if isinstance(self.redis, AsyncRedis):
            await self.redis.delete(key)
        else:
            await asyncio.to_thread(self.redis.delete, key)

    def invalidate_sync(self, assistant_id: str):
        key = self._cache_key(assistant_id)
        if hasattr(self.redis, "delete") and not asyncio.iscoroutinefunction(self.redis.delete):
            self.redis.delete(key)
        else:
            try:
                asyncio.run(self.delete(assistant_id))
            except RuntimeError:
                pass

    def retrieve_sync(self, assistant_id: str):
        return asyncio.run(self.retrieve(assistant_id))
