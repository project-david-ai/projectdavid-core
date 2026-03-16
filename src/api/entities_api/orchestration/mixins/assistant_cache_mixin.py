from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from entities_api.cache.assistant_cache import AssistantCache
    from redis.asyncio import Redis


class AssistantCacheMixin:
    """
    Hybrid Mixin:
    1. Prioritizes explicit injection (via init).
    2. Falls back to ServiceRegistry/Lazy loading (for legacy consistency).
    """

    _assistant_cache: Optional[AssistantCache] = None

    @property
    def assistant_cache(self) -> AssistantCache:
        # 1. Fast Path: It was injected via __init__ (The "New Style")
        if self._assistant_cache:
            return self._assistant_cache

        # 2. Lazy Path: Use ServiceRegistryMixin logic (The "Old/Current Style")
        if hasattr(self, "_get_service"):
            return self._get_service("AssistantCache")

        # 3. Last Resort: Construct it if 'redis' is available on 'self'
        if hasattr(self, "redis") and self.redis:
            from entities_api.cache.assistant_cache import AssistantCache

            self._assistant_cache = AssistantCache(redis=self.redis)
            return self._assistant_cache

        # 4. Configuration error
        raise ValueError(
            f"AssistantCache could not be resolved in {self.__class__.__name__}. "
            "Ensure it is injected via __init__, or that the class inherits ServiceRegistryMixin."
        )
