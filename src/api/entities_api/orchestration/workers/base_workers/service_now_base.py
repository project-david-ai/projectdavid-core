from __future__ import annotations

import asyncio
import json
import os
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from dotenv import load_dotenv
from entities_api.cache.assistant_cache import AssistantCache
from entities_api.clients.delta_normalizer import DeltaNormalizer
from projectdavid import StreamEvent
from projectdavid_common.utilities.logging_service import LoggingUtility
from projectdavid_common.validation import StatusEnum

# --- DEPENDENCIES ---
from src.api.entities_api.dependencies import get_redis, get_redis_sync
from src.api.entities_api.orchestration.engine.orchestrator_core import \
    OrchestratorCore
# --- MIXINS ---
from src.api.entities_api.orchestration.mixins.provider_mixins import \
    _ProviderMixins

load_dotenv()
LOG = LoggingUtility()


class ServiceNowBaseWorker(
    _ProviderMixins,
    OrchestratorCore,
    ABC,
):
    """
    Async Base for ServiceNow Providers (Apriel-1.6-15b-Thinker).
    Migrated to Async-First Architecture.
    """

    def __init__(
        self,
        *,
        assistant_id: str | None = None,
        thread_id: str | None = None,
        redis=None,
        base_url: str | None = None,
        api_key: str | None = None,
        assistant_cache_service: Optional[AssistantCache] = None,
        **extra,
    ) -> None:

        # 2. Setup Redis (Critical for the Mixin fallback)
        # We use get_redis_sync() if no client is provided, ensuring we have a connection.
        self.redis = redis or get_redis_sync()

        # 3. Setup the Cache Service (The "New Way")
        # If passed explicitly, store it. If not, the Mixin will lazy-load it using self.redis
        if assistant_cache_service:
            self._assistant_cache = assistant_cache_service
        elif "assistant_cache" in extra and isinstance(extra["assistant_cache"], AssistantCache):
            # Handle case where it might be passed via **extra
            self._assistant_cache = extra["assistant_cache"]

        # 4. Setup the Data/Config (The "Old Way" renamed)
        # We rename this to avoid overwriting the Mixin's property.
        # We check if a raw dict was passed in 'extra' (legacy support)
        legacy_config = extra.get("assistant_config") or extra.get("assistant_cache")
        self.assistant_config: Dict[str, Any] = (
            legacy_config if isinstance(legacy_config, dict) else {}
        )

        self._david_client: Any = None
        self.redis = redis or get_redis()
        self.assistant_id = assistant_id
        self.thread_id = thread_id
        self.base_url = base_url or os.getenv("BASE_URL")
        self.api_key = api_key or extra.get("api_key")

        self.model_name = extra.get("model_name", "ServiceNow-AI/Apriel-1.5-15b-Thinker")
        self.max_context_window = extra.get("max_context_window", 128000)
        self.threshold_percentage = extra.get("threshold_percentage", 0.8)

        self._current_tool_call_id: str | None = None
        self._pending_tool_payload: Optional[Dict[str, Any]] = None
        self._decision_payload: Optional[Dict[str, Any]] = None

        self.setup_services()

        # Ensure mixin stubs exist if failed to load (Standardized from GptOss)
        if not hasattr(self, "get_function_call_state"):
            LOG.error("CRITICAL: ToolRoutingMixin failed to load.")
            self.get_function_call_state = lambda: None
            self.set_function_call_state = lambda x: None
            self.set_tool_response_state = lambda x: None

        LOG.debug("Hyperbolic-Ds1 provider ready (assistant=%s)", assistant_id)

    @abstractmethod
    def _get_client_instance(self, api_key: str):
        pass

    async def stream(
        self,
        thread_id: str,
        message_id: str | None,
        run_id: str,
        assistant_id: str,
        model: Any,
        *,
        force_refresh: bool = False,
        stream_reasoning: bool = True,
        api_key: str | None = None,
        **kwargs,
    ) -> AsyncGenerator[Union[str, StreamEvent], None]:
        """
        Level 3 Agentic Stream (Native Mode):
        - Uses raw XML/Tag persistence to prevent Llama/DeepSeek persona breakage.
        - Maintains internal batching for parallel tool execution.
        """
        redis = self.redis
        stream_key = f"stream:{run_id}"
        stop_event = self.start_cancellation_monitor(run_id)

        # Early Variable Initialization
        self._current_tool_call_id = None
        self._decision_payload = None
        self._tool_queue: List[Dict] = []

        accumulated: str = ""
        assistant_reply: str = ""
        reasoning_reply: str = ""
        decision_buffer: str = ""
        plan_buffer: str = ""
        current_block: str | None = None

        try:
            if hasattr(self, "_get_model_map") and (mapped := self._get_model_map(model)):
                model = mapped

            # [NEW] Ensure cache is hot before starting
            self.assistant_id = assistant_id
            await self._ensure_config_loaded()
            agent_mode_setting = self.assistant_config.get("agent_mode", False)
            decision_telemetry = self.assistant_config.get("decision_telemetry", True)
            web_access_setting = self.assistant_config.get("decision_telemetry", False)

            LOG.debug(f"Test_cache -> Agent: {agent_mode_setting}, Telemetry: {decision_telemetry}")

            ctx = await self._set_up_context_window(
                assistant_id,
                thread_id,
                trunk=True,
                force_refresh=force_refresh,
                agent_mode=agent_mode_setting,
                decision_telemetry=decision_telemetry,
                web_access=web_access_setting,
            )

            if not api_key:
                yield json.dumps({"type": "error", "content": "Missing API key."})
                return

            yield json.dumps({"type": "status", "status": "started", "run_id": run_id})

            client = self._get_client_instance(api_key=api_key)

            # --- [DEBUG] RAW CONTEXT DUMP ---
            LOG.info(f"\nRAW_CTX_DUMP:\n{json.dumps(ctx, indent=2, ensure_ascii=False)}")

            raw_stream = client.stream_chat_completion(
                messages=ctx,
                model=model,
                max_tokens=10000,
                temperature=kwargs.get("temperature", 0.6),
                stream=True,
            )

            async for chunk in DeltaNormalizer.async_iter_deltas(raw_stream, run_id):
                if stop_event.is_set():
                    break

                ctype = chunk.get("type")
                ccontent = chunk.get("content") or ""

                # --- REAL-TIME STATE MACHINE ---
                if ctype == "content":
                    if current_block == "fc":
                        accumulated += "</fc>"
                    elif current_block == "think":
                        accumulated += "</think>"
                    elif current_block == "plan":
                        accumulated += "</plan>"
                    current_block = None
                    assistant_reply += ccontent
                    accumulated += ccontent
                elif ctype == "call_arguments":
                    if current_block != "fc":
                        if current_block == "think":
                            accumulated += "</think>"
                        elif current_block == "plan":
                            accumulated += "</plan>"
                        accumulated += "<fc>"
                        current_block = "fc"
                    accumulated += ccontent
                elif ctype == "reasoning":
                    if current_block != "think":
                        if current_block == "fc":
                            accumulated += "</fc>"
                        elif current_block == "plan":
                            accumulated += "</plan>"
                        accumulated += "<think>"
                        current_block = "think"
                    reasoning_reply += ccontent
                elif ctype == "plan":
                    if current_block != "plan":
                        if current_block == "fc":
                            accumulated += "</fc>"
                        elif current_block == "think":
                            accumulated += "</think>"
                        accumulated += "<plan>"
                        current_block = "plan"
                    plan_buffer += ccontent
                    accumulated += ccontent
                elif ctype == "decision":
                    decision_buffer += ccontent
                    if current_block == "fc":
                        accumulated += "</fc>"
                    elif current_block == "think":
                        accumulated += "</think>"
                    elif current_block == "plan":
                        accumulated += "</plan>"
                    current_block = "decision"

                if ctype == "call_arguments":
                    continue
                yield json.dumps(chunk)
                await self._shunt_to_redis_stream(redis, stream_key, chunk)

            # Cleanup open tags
            if current_block == "fc":
                accumulated += "</fc>"
            elif current_block == "think":
                accumulated += "</think>"
            elif current_block == "plan":
                accumulated += "</plan>"

        except Exception as exc:
            LOG.error(f"DEBUG: Stream Exception: {exc}")
            err = {"type": "error", "content": f"Stream error: {exc}", "run_id": run_id}
            yield json.dumps(err)
            await self._shunt_to_redis_stream(redis, stream_key, err)
        finally:
            stop_event.set()

        # --- POST-STREAM: BATCH VALIDATION ---
        if decision_buffer:
            try:
                self._decision_payload = json.loads(decision_buffer.strip())
            except Exception:
                pass

        yield json.dumps({"type": "status", "status": "processing", "run_id": run_id})

        # --- [LEVEL 3] NATIVE PERSISTENCE ---
        # The parser finds the tools to drive the backend (Action records).
        tool_calls_batch = self.parse_and_set_function_calls(accumulated, assistant_reply)

        # [THE FIX]: We save the RAW text emitted by Llama.
        # No formal JSON structure, no ID injection into the dialogue content.
        message_to_save = accumulated
        final_status = StatusEnum.completed.value

        if tool_calls_batch:
            # We still keep the tool_queue so the dispatcher knows what to execute
            self._tool_queue = tool_calls_batch
            final_status = StatusEnum.pending_action.value

            # [LOGGING]
            LOG.info(f"🚀 [L3 NATIVE MODE] Turn 1 Batch size: {len(tool_calls_batch)}")

        # Persistence: Save the raw <plan> and <fc> text exactly as Llama intended
        if message_to_save:
            await self.finalize_conversation(message_to_save, thread_id, assistant_id, run_id)

        if self.project_david_client:
            await asyncio.to_thread(
                self.project_david_client.runs.update_run_status, run_id, final_status
            )

        if not tool_calls_batch:
            yield json.dumps({"type": "status", "status": "complete", "run_id": run_id})
