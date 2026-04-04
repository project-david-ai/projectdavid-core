import os

from entities_api.cache.assistant_cache import AssistantCache
from entities_api.cache.message_cache import MessageCache
from redis import Redis as SyncRedis


def get_sync_invalidator() -> AssistantCache:
    """
    Creates a short-lived sync connection to Redis for invalidation purposes.
    """
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    client = SyncRedis.from_url(redis_url, decode_responses=True)

    return AssistantCache(redis=client)


def get_sync_message_cache() -> MessageCache:
    """
    Creates a synchronous MessageCache instance for cache invalidation
    within sync CRUD services.
    """
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    client = SyncRedis.from_url(redis_url, decode_responses=True)

    return MessageCache(redis=client)
