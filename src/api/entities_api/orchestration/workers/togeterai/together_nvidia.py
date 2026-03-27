import os

from entities_api.clients.unified_async_client import get_cached_client

from src.api.entities_api.orchestration.workers.base_workers.nvidia_base import (
    NvidiaBaseWorker,
)


class TogetherNvidiaWorker(NvidiaBaseWorker):
    """TogetherAI-specific implementation of Nvidia."""

    def _get_client_instance(self, api_key: str):
        """
        Returns an async-ready Hyperbolic client.
        """
        return get_cached_client(
            api_key=api_key,
            base_url=os.getenv("TOGETHER_BASE_URL"),
            enable_logging=False,
        )

        # Note: the base class calls client.stream_chat_completion directly.
        # _execute_stream_request is likely not needed anymore, but keeping it
        # for signature compatibility if other parts of your stack call it.

    async def _execute_stream_request(self, client, payload: dict):
        return client.stream_chat_completion(**payload)
