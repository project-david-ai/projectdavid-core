# src/api/entities_api/orchestration/workers/ollama_default_worker.py
import os

from src.api.entities_api.orchestration.workers.base_workers.vllm_raw_worker import \
    VLLMDefaultBaseWorker


class VllmDefaultWorker(VLLMDefaultBaseWorker):
    """
    Concrete Ollama worker.

    Unlike cloud-backed workers (e.g. TogetherDs1, HyperbolicNemotron), this
    worker does NOT use the unified async client. Streaming is handled natively
    by OllamaNativeStream via a direct httpx POST to /api/chat, which gives
    us access to the `thinking` field that the OpenAI-compat /v1 endpoint drops.

    _get_client_instance and _execute_stream_request satisfy the architectural
    contract but are not invoked — the base class stream() calls
    self._stream_ollama_raw() directly.
    """

    def _get_client_instance(self, api_key: str):
        """
        No unified client needed for the native Ollama path.
        Returns None — the base class never calls this for Ollama workers.
        """
        return None

    async def _execute_stream_request(self, client, payload: dict):
        """
        Not used in the native Ollama path.
        Streaming is handled by OllamaNativeStream._stream_ollama_raw().
        OLLAMA_BASE_URL is read from the environment (default: http://localhost:11434).
        """
        raise NotImplementedError(
            "OllamaDefaultWorker streams via _stream_ollama_raw(), "
            "not _execute_stream_request(). "
            f"OLLAMA_BASE_URL={os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}"
        )
