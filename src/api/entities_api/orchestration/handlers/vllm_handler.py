# src/api/entities_api/orchestration/handlers/vllm_handler.py

from typing import Any, AsyncGenerator, Optional, Type

from projectdavid_common.utilities.logging_service import LoggingUtility

from src.api.entities_api.orchestration.engine.inference_arbiter import InferenceArbiter
from src.api.entities_api.orchestration.workers.vllm.vllm_default import VllmDefaultWorker

LOG = LoggingUtility()


class VllmHandler:
    """
    Pure synchronous dispatcher for **TogetherAI** model requests.
    Consolidated to use provider/family prefixes for better maintainability.
    """

    SUBMODEL_CLASS_MAP: dict[str, Type[Any]] = {
        # --- All Families ---
        "vllm/": VllmDefaultWorker,
        "": VllmDefaultWorker,
    }

    def __init__(self, arbiter: InferenceArbiter):
        self.arbiter = arbiter
        # Sort keys by length descending. This ensures that if we ever add a specific
        # long-form override, it matches before the generic family prefix.
        self._sorted_sub_routes = sorted(self.SUBMODEL_CLASS_MAP.keys(), key=len, reverse=True)
        LOG.info("OllamaHandler consolidated dispatcher initialized.")

    def _get_specific_handler_instance(self, unified_model_id: str) -> Any:
        """
        Resolves the concrete worker instance based on the unified model ID.
        """
        prefix = "vllm/"
        lower_id = unified_model_id.lower()

        # Strip the platform prefix if present
        if lower_id.startswith(prefix):
            sub_model_id = lower_id[len(prefix) :]
        else:
            sub_model_id = lower_id
            LOG.warning(f"Model ID '{unified_model_id}' missing expected prefix '{prefix}'.")

        specific_cls: Optional[Type[Any]] = None

        for route_key in self._sorted_sub_routes:
            key_lc = route_key.lower()

            # Match 1: Folder-style prefix (e.g., "meta-llama/")
            if key_lc.endswith("/") and sub_model_id.startswith(key_lc):
                specific_cls = self.SUBMODEL_CLASS_MAP[route_key]
                break

            # Match 2: Substring/Identifier match (e.g., "nvidia")
            if not key_lc.endswith("/") and key_lc in sub_model_id:
                specific_cls = self.SUBMODEL_CLASS_MAP[route_key]
                break

        if specific_cls is None:
            LOG.error(f"No handler found for Vllm sub-model: '{sub_model_id}'")
            raise ValueError(f"Unsupported Vllm model: {unified_model_id}")

        LOG.debug(f"Routing '{sub_model_id}' to: {specific_cls.__name__}")
        return self.arbiter.get_provider_instance(specific_cls)

    async def process_conversation(
        self,
        thread_id: str,
        message_id: Optional[str],
        run_id: str,
        assistant_id: str,
        model: Any,
        stream_reasoning: bool = False,
        api_key: Optional[str] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        worker = self._get_specific_handler_instance(model)

        # 'yield from' does not work with async generators.
        # We must iterate asynchronously and yield the chunks.
        async for chunk in worker.process_conversation(
            thread_id=thread_id,
            message_id=message_id,
            run_id=run_id,
            assistant_id=assistant_id,
            model=model,
            stream_reasoning=stream_reasoning,
            api_key=api_key,
            **kwargs,
        ):
            yield chunk

    async def stream(
        self,
        thread_id: str,
        message_id: Optional[str],
        run_id: str,
        assistant_id: str,
        model: Any,
        stream_reasoning: bool = False,
        api_key: Optional[str] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        worker = self._get_specific_handler_instance(model)

        async for chunk in worker.stream(
            thread_id=thread_id,
            message_id=message_id,
            run_id=run_id,
            assistant_id=assistant_id,
            model=model,
            stream_reasoning=stream_reasoning,
            api_key=api_key,
            **kwargs,
        ):
            yield chunk

    async def process_function_calls(
        self,
        thread_id: str,
        run_id: str,
        assistant_id: str,
        model: Any = None,
        api_key: Optional[str] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        worker = self._get_specific_handler_instance(model)

        # Internal workers typically use 'process_tool_calls'
        async for chunk in worker.process_tool_calls(
            thread_id=thread_id,
            run_id=run_id,
            assistant_id=assistant_id,
            model=model,
            api_key=api_key,
            **kwargs,
        ):
            yield chunk
