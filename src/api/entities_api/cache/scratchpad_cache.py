# src/api/entities_api/cache/scratchpad_cache.py

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Dict, Union

from redis import Redis as SyncRedis

try:
    from redis.asyncio import Redis as AsyncRedis
except ImportError:

    class AsyncRedis:
        pass


# Scratchpad working state is intentionally ephemeral/reconstructable.
REDIS_SCRATCHPAD_TTL = int(
    os.getenv(
        "REDIS_SCRATCHPAD_TTL_SECONDS",
        "86400",
    )
)


class ScratchpadCache:
    """
    Redis data plane for Scratchpads.

    Canonical first-class keys:

        scratchpad:{owner_id}:{scratchpad_id}:content
        scratchpad:{owner_id}:{scratchpad_id}:entries

    ``content`` is the supervisor-editable working body.

    ``entries`` is an ordered append-only ledger. Entries use Redis RPUSH,
    avoiding the legacy GET -> concatenate -> SET lost-update race.

    The legacy thread-keyed notebook methods remain temporarily for the
    existing Deep Research tool path. They will be removed once that path
    resolves and propagates first-class Scratchpad identity.
    """

    def __init__(
        self,
        redis: Union[SyncRedis, "AsyncRedis"],
    ) -> None:
        self.redis = redis

    # ------------------------------------------------------------------
    # Redis execution helpers
    # ------------------------------------------------------------------

    async def _call(
        self,
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        method = getattr(
            self.redis,
            method_name,
        )

        if isinstance(
            self.redis,
            AsyncRedis,
        ):
            return await method(
                *args,
                **kwargs,
            )

        return await asyncio.to_thread(
            method,
            *args,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Canonical first-class keyspace
    # ------------------------------------------------------------------

    @staticmethod
    def _content_key(
        owner_id: str,
        scratchpad_id: str,
    ) -> str:
        return f"scratchpad:{owner_id}:" f"{scratchpad_id}:content"

    @staticmethod
    def _entries_key(
        owner_id: str,
        scratchpad_id: str,
    ) -> str:
        return f"scratchpad:{owner_id}:" f"{scratchpad_id}:entries"

    async def get_content(
        self,
        *,
        owner_id: str,
        scratchpad_id: str,
    ) -> Dict[str, Any]:
        """
        Return the current working body.

        Missing Redis state is represented as an empty, unmaterialized
        Scratchpad body rather than an error.
        """

        raw = await self._call(
            "get",
            self._content_key(
                owner_id,
                scratchpad_id,
            ),
        )

        if not raw:
            return {
                "content": "",
                "updated_at": None,
            }

        data = json.loads(raw)

        return {
            "content": data.get(
                "content",
                "",
            ),
            "updated_at": data.get("updated_at"),
        }

    async def set_content(
        self,
        *,
        owner_id: str,
        scratchpad_id: str,
        content: str,
    ) -> Dict[str, Any]:
        """Replace the supervisor-editable working body."""

        updated_at = time.time()

        payload = {
            "content": content,
            "updated_at": updated_at,
        }

        await self._call(
            "set",
            self._content_key(
                owner_id,
                scratchpad_id,
            ),
            json.dumps(payload),
            ex=REDIS_SCRATCHPAD_TTL,
        )

        return payload

    async def append_entry(
        self,
        *,
        owner_id: str,
        scratchpad_id: str,
        content: str,
    ) -> Dict[str, Any]:
        """
        Atomically append one immutable ledger entry using Redis RPUSH.
        """

        created_at = time.time()

        payload = {
            "content": content,
            "created_at": created_at,
        }

        key = self._entries_key(
            owner_id,
            scratchpad_id,
        )

        await self._call(
            "rpush",
            key,
            json.dumps(payload),
        )

        # Refresh ledger lifetime whenever new work arrives.
        await self._call(
            "expire",
            key,
            REDIS_SCRATCHPAD_TTL,
        )

        return payload

    async def list_entries(
        self,
        *,
        owner_id: str,
        scratchpad_id: str,
    ) -> list[Dict[str, Any]]:
        """Return ledger entries in append order."""

        raw_entries = await self._call(
            "lrange",
            self._entries_key(
                owner_id,
                scratchpad_id,
            ),
            0,
            -1,
        )

        if not raw_entries:
            return []

        return [json.loads(raw) for raw in raw_entries]

    async def clear_content(
        self,
        *,
        owner_id: str,
        scratchpad_id: str,
    ) -> None:
        """Delete only the mutable working body."""

        await self._call(
            "delete",
            self._content_key(
                owner_id,
                scratchpad_id,
            ),
        )

    async def clear_entries(
        self,
        *,
        owner_id: str,
        scratchpad_id: str,
    ) -> None:
        """Delete only the append ledger."""

        await self._call(
            "delete",
            self._entries_key(
                owner_id,
                scratchpad_id,
            ),
        )

    async def delete_all_scratchpad_data(
        self,
        *,
        owner_id: str,
        scratchpad_id: str,
    ) -> None:
        """Purge all Redis state for one Scratchpad resource."""

        await self._call(
            "delete",
            self._content_key(
                owner_id,
                scratchpad_id,
            ),
            self._entries_key(
                owner_id,
                scratchpad_id,
            ),
        )

    # ------------------------------------------------------------------
    # Legacy thread-keyed compatibility surface
    # ------------------------------------------------------------------
    #
    # Existing Deep Research currently addresses the notebook by the
    # parent/shared thread. Keep this operational until the orchestration
    # path is migrated to Scratchpad identity.
    # ------------------------------------------------------------------

    def _cache_key(
        self,
        thread_id: str,
    ) -> str:
        return f"scratchpad:{thread_id}:notebook"

    async def get_scratchpad(
        self,
        thread_id: str,
    ) -> Dict[str, Any]:
        raw = await self._call(
            "get",
            self._cache_key(thread_id),
        )

        if not raw:
            return {
                "content": "",
                "last_updated": 0.0,
            }

        return json.loads(raw)

    async def overwrite_scratchpad(
        self,
        thread_id: str,
        content: str,
    ) -> None:
        payload = {
            "content": content,
            "last_updated": time.time(),
        }

        await self._call(
            "set",
            self._cache_key(thread_id),
            json.dumps(payload),
            ex=REDIS_SCRATCHPAD_TTL,
        )

    async def append_to_scratchpad(
        self,
        thread_id: str,
        new_notes: str,
    ) -> None:
        """
        Legacy compatibility append.

        This remains GET -> modify -> SET only until the existing tool path
        has been migrated to first-class Scratchpad identity. New callers
        must use ``append_entry``.
        """

        current_data = await self.get_scratchpad(thread_id)

        existing_content = current_data.get(
            "content",
            "",
        )

        updated_content = (f"{existing_content}\n\n{new_notes}").strip()

        await self.overwrite_scratchpad(
            thread_id,
            updated_content,
        )

    async def clear_scratchpad(
        self,
        thread_id: str,
    ) -> None:
        await self._call(
            "delete",
            self._cache_key(thread_id),
        )


__all__ = [
    "REDIS_SCRATCHPAD_TTL",
    "ScratchpadCache",
]
