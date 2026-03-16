from functools import lru_cache
from typing import Optional, TypeVar

from dotenv import load_dotenv
# Import your AsyncHyperbolicClient definition
from entities_api.clients.unified_async_client import (
    _ACTIVE_CLIENTS, AsyncUnifiedInferenceClient)
from projectdavid import Entity
from projectdavid_common.utilities.logging_service import LoggingUtility

load_dotenv()
LOG = LoggingUtility()

T = TypeVar("T")


# -----------------------------------------------------------------------------
# CLIENT FACTORY
# -----------------------------------------------------------------------------
class ClientFactoryMixin:
    """
    Factory / cache for external SDK clients.
    """

    @lru_cache(maxsize=32)
    def _get_project_david_client(
        self, *, api_key: Optional[str], base_url: Optional[str]
    ) -> Entity:
        if not api_key or not base_url:
            raise RuntimeError("api_key + base_url required for Entity client")
        try:
            return Entity(api_key=api_key, base_url=base_url)
        except Exception as exc:
            LOG.error("Project-David client init failed: %s", exc, exc_info=True)
            raise

    @lru_cache(maxsize=32)
    def _get_unified_client(
        self, *, api_key: Optional[str], base_url: Optional[str]
    ) -> AsyncUnifiedInferenceClient:
        """
        Returns a cached AsyncHyperbolicClient.
        NOTE: This caches the CONNECTION POOL, which improves performance.
        """
        if not api_key or not base_url:
            raise RuntimeError("api_key + base_url required for Hyperbolic client")
        try:
            return AsyncUnifiedInferenceClient(api_key=api_key, base_url=base_url)
        except Exception as exc:
            LOG.error("Hyperbolic client init failed: %s", exc, exc_info=True)
            raise

    def _get_cached_unified_client(
        self, api_key: str, base_url: str, enable_logging: bool = False
    ) -> AsyncUnifiedInferenceClient:
        """
        Returns a cached client instance for the given API key/Base URL combo.
        This prevents SSL Handshake overhead on every request.
        """
        cache_key = f"{api_key[-6:]}@{base_url}"  # Simple hash key

        if cache_key not in _ACTIVE_CLIENTS:
            client = AsyncUnifiedInferenceClient(
                api_key=api_key, base_url=base_url, enable_chunk_logging=enable_logging
            )
            _ACTIVE_CLIENTS[cache_key] = client

        # Check if the event loop is closed (rare edge case in some runners)
        client = _ACTIVE_CLIENTS[cache_key]
        if client.client.is_closed:
            # Re-create if closed
            client = AsyncUnifiedInferenceClient(
                api_key=api_key, base_url=base_url, enable_chunk_logging=enable_logging
            )
            _ACTIVE_CLIENTS[cache_key] = client

        return client
