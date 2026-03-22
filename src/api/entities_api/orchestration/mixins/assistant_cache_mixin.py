from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from entities_api.cache.assistant_cache import AssistantCache
    from redis.asyncio import Redis


class AssistantCacheMixin:
    """
    Single source of truth for AssistantCache resolution.

    Resolution priority:
        1. Injected via __init__ (fast path — new style)
        2. ServiceRegistryMixin._get_service() (legacy consistency)
        3. Lazy construction from self.redis (fallback — requires OrchestratorCore
           to have set self.redis before first cache access)
        4. Hard ValueError — configuration error, nothing can recover.

    Owns:
        _assistant_cache        — backing attribute
        assistant_cache         — property with four-level resolution chain
        get_assistant_cache()   — method accessor used by ContextMixin and
                                  OrchestratorCore._ensure_config_loaded()

    Contract:
        ContextMixin calls self.get_assistant_cache() throughout.
        OrchestratorCore calls self.assistant_cache directly in
        _ensure_config_loaded() and load_assistant_config().
        Both resolve through this mixin — do not redefine either on
        OrchestratorCore or any worker base class.
    """

    _assistant_cache: Optional[AssistantCache] = None

    @property
    def assistant_cache(self) -> AssistantCache:
        # 1. Fast Path: injected via __init__
        if self._assistant_cache:
            return self._assistant_cache

        # 2. Lazy Path: ServiceRegistryMixin._get_service() expects a class,
        #    not a string — import here since TYPE_CHECKING guard above
        #    makes the top-level import unavailable at runtime.
        if hasattr(self, "_get_service"):
            from entities_api.cache.assistant_cache import AssistantCache

            return self._get_service(AssistantCache)

        # 3. Last Resort: construct from self.redis
        if hasattr(self, "redis") and self.redis:
            from entities_api.cache.assistant_cache import AssistantCache

            self._assistant_cache = AssistantCache(redis=self.redis)
            return self._assistant_cache

        # 4. Configuration error
        raise ValueError(
            f"AssistantCache could not be resolved in {self.__class__.__name__}. "
            "Ensure it is injected via __init__, or that the class inherits ServiceRegistryMixin."
        )

    @assistant_cache.setter
    def assistant_cache(self, value: AssistantCache) -> None:
        self._assistant_cache = value

    def get_assistant_cache(self) -> AssistantCache:
        """
        Method accessor for ContextMixin compatibility.
        Delegates to the assistant_cache property — all resolution
        logic lives there.
        """
        return self.assistant_cache
