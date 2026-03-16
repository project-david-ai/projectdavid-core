# src/api/entities_api/orchestration/workers/hyperbolic/hb_handler.py
from typing import Any, AsyncGenerator, Optional, Type

from projectdavid_common.utilities.logging_service import LoggingUtility

# Worker Imports
from src.api.entities_api.orchestration.engine.inference_arbiter import InferenceArbiter
from src.api.entities_api.orchestration.workers.hyperbolic.hb_deepseek import HyperbolicDs1
from src.api.entities_api.orchestration.workers.hyperbolic.hb_gpt_oss import HyperbolicGptOssWorker
from src.api.entities_api.orchestration.workers.hyperbolic.hb_llama import HyperbolicLlamaWorker
from src.api.entities_api.orchestration.workers.hyperbolic.hb_qwen import HyperbolicQwenWorker

LOG = LoggingUtility()


class HyperbolicHandler:
    """
    Asynchronous dispatcher for Hyperbolic model requests.
    Consolidated to use family prefixes for DeepSeek, Llama, and Qwen.
    """

    SUBMODEL_CLASS_MAP: dict[str, Type[Any]] = {
        # --- DeepSeek Family (covers V3, R1, and distilled versions) ---
        "deepseek": HyperbolicDs1,
        # --- Llama Family ---
        "meta-llama/": HyperbolicLlamaWorker,
        # --- Qwen Family (covers Coder, QwQ, etc.) ---
        "qwen/": HyperbolicQwenWorker,
        # --- Specialized Handlers ---
        "gpt-oss": HyperbolicGptOssWorker,
    }

    def __init__(self, arbiter: InferenceArbiter):
        self.arbiter = arbiter
        # Sort by length descending to match most specific key first
        self._sorted_sub_routes = sorted(self.SUBMODEL_CLASS_MAP.keys(), key=len, reverse=True)
        LOG.info("HyperbolicHandler consolidated dispatcher initialized.")

    def _get_specific_handler_instance(self, unified_model_id: str) -> Any:
        """
        Resolves the concrete Hyperbolic worker based on model ID prefix or substring.
        """
        prefix = "hyperbolic/"
        lower_id = unified_model_id.lower()

        # Strip platform prefix
        sub_model_id = lower_id[len(prefix) :] if lower_id.startswith(prefix) else lower_id

        specific_cls: Optional[Type[Any]] = None
        for route_key in self._sorted_sub_routes:
            key_lc = route_key.lower()

            # Match 1: Folder/Prefix style (e.g., "meta-llama/")
            if key_lc.endswith("/") and sub_model_id.startswith(key_lc):
                specific_cls = self.SUBMODEL_CLASS_MAP[route_key]
                break

            # Match 2: Substring style (e.g., "deepseek" or "gpt-oss")
            if not key_lc.endswith("/") and key_lc in sub_model_id:
                specific_cls = self.SUBMODEL_CLASS_MAP[route_key]
                break

        if specific_cls is None:
            LOG.error(f"No handler found for Hyperbolic sub-model: '{sub_model_id}'")
            raise ValueError(f"Unsupported Hyperbolic model: {unified_model_id}")

        LOG.debug(f"Routing '{sub_model_id}' to: {specific_cls.__name__}")
        return self.arbiter.get_provider_instance(specific_cls)

    # -------------------------------------------------------------------------
    # FIXED: Changed to async def + AsyncGenerator + async for loops
    # -------------------------------------------------------------------------
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
