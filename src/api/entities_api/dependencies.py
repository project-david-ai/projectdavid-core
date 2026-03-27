import os
from datetime import datetime
from typing import Optional

import redis.asyncio as aioredis

# --- CACHE IMPORTS ---
from entities_api.cache.assistant_cache import AssistantCache
from entities_api.cache.inventory_cache import InventoryCache
from entities_api.cache.message_cache import MessageCache
from entities_api.cache.scratchpad_cache import ScratchpadCache
from entities_api.cache.web_cache import WebSessionCache

# --- SERVICE IMPORTS ---
from entities_api.services.scratchpad_service import ScratchpadService
from entities_api.services.web_reader import UniversalWebReader
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from redis.asyncio import Redis
from sqlalchemy.orm import Session

# --- DB & MODEL IMPORTS ---
from src.api.entities_api.db.database import get_db
from src.api.entities_api.models.models import ApiKey, User

API_KEY_NAME = "X-API-Key"
_api_key_scheme = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# -----------------------------------------------------------------------------
# FIX: Global Redis Connection Pool & Sync Factory
# -----------------------------------------------------------------------------
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# 1. Create Pool Globally
redis_pool = aioredis.ConnectionPool.from_url(REDIS_URL, decode_responses=True)


# 2. Synchronous Factory (Use this in __init__ methods)
def get_redis_sync() -> Redis:
    """
    Creates and returns a Redis client synchronously.
    The client handles async operations, but creation is instant.
    """
    return aioredis.Redis(connection_pool=redis_pool)


# 3. Async Dependency (Use this in FastAPI Routes/Depends)
async def get_redis() -> Redis:
    """
    FastAPI dependency version.
    """
    return get_redis_sync()


# -----------------------------------------------------------------------------
# Auth & User Dependencies
# -----------------------------------------------------------------------------
async def get_api_key(
    api_key_header: Optional[str] = Security(_api_key_scheme),
    db: Session = Depends(get_db),
) -> ApiKey:
    if not api_key_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key in 'X-API-Key' header.",
            headers={"WWW-Authenticate": "APIKey"},
        )

    prefix = api_key_header[:8]
    if len(api_key_header) <= len(prefix):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key format.",
            headers={"WWW-Authenticate": "APIKey"},
        )

    key = (
        db.query(ApiKey)
        .filter(ApiKey.prefix == prefix, ApiKey.is_active.is_(True))
        .first()
    )

    if not key or not key.verify_key(api_key_header):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or inactive API Key.",
            headers={"WWW-Authenticate": "APIKey"},
        )

    if key.expires_at and key.expires_at < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key has expired.",
            headers={"WWW-Authenticate": "APIKey"},
        )
    return key


async def get_current_user(api_key_data: ApiKey = Depends(get_api_key)) -> User:
    if not api_key_data.user:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not resolve user for the provided API key.",
        )
    return api_key_data.user


# -----------------------------------------------------------------------------
# Cache Dependencies
# -----------------------------------------------------------------------------
async def get_assistant_cache(redis: Redis = Depends(get_redis)) -> AssistantCache:
    return AssistantCache(
        redis=redis,
        pd_base_url=os.getenv("ASSISTANTS_BASE_URL"),
        pd_api_key=os.getenv("ADMIN_API_KEY"),
    )


async def get_message_cache(redis: Redis = Depends(get_redis)) -> MessageCache:
    return MessageCache(redis=redis)


async def get_web_cache(redis: Redis = Depends(get_redis)) -> WebSessionCache:
    """
    Provides the WebSessionCache for managing browsing contexts.
    """
    return WebSessionCache(redis=redis)


async def get_inventory_cache(redis: Redis = Depends(get_redis)) -> InventoryCache:
    """
    Provides the InventoryCache for storing/retrieving network device maps.
    """
    return InventoryCache(redis=redis)


async def get_scratchpad_cache(redis: Redis = Depends(get_redis)) -> ScratchpadCache:
    """
    Provides the ScratchpadCache for the Deep Research Agent's persistent working memory.
    """
    return ScratchpadCache(redis=redis)


# -----------------------------------------------------------------------------
# Service Dependencies
# -----------------------------------------------------------------------------
async def get_web_reader(
    cache: WebSessionCache = Depends(get_web_cache),
) -> UniversalWebReader:
    """
    Injects the UniversalWebReader service.
    Automatically handles the Redis connection via WebSessionCache.
    """
    return UniversalWebReader(cache_service=cache)


async def get_scratchpad_service(
    cache: ScratchpadCache = Depends(get_scratchpad_cache),
) -> ScratchpadService:
    return ScratchpadService(cache=cache)
