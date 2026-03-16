# src/api/entities_api/workers/vllm_raw_worker.py
"""
VLLMDefaultBaseWorker
=====================
Mirrors OllamaDefaultBaseWorker exactly.

The ONLY difference:

    OllamaDefaultBaseWorker:   inherits OllamaNativeStream
                                calls    self._stream_ollama_raw()

    VLLMDefaultBaseWorker:     inherits VLLMRawStream
                                calls    self._stream_vllm_raw()

Everything else — DeltaNormalizer, OrchestratorCore, tool routing,
identity swap, state management — is identical and untouched.

Environment variables:
    VLLM_BASE_URL   vLLM server URL (default: http://localhost:8000)
    VLLM_TIMEOUT    Request timeout in seconds (default: 120)
"""

from __future__ import annotations

import asyncio
import json
import os
import queue as _queue_mod
import threading
import time
import uuid
from abc import ABC
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Union

from dotenv import load_dotenv
from entities_api.cache.assistant_cache import AssistantCache
from entities_api.clients.delta_normalizer import DeltaNormalizer
from entities_api.clients.vllm_raw_stream import VLLMRawStream  # ← swap
from entities_api.platform_tools.delegated_model_map.delegation_model_map import \
    get_delegated_model
from projectdavid import StreamEvent
from projectdavid_common.utilities.logging_service import LoggingUtility
from projectdavid_common.validation import StatusEnum

from src.api.entities_api.dependencies import get_redis, get_redis_sync
from src.api.entities_api.orchestration.engine.orchestrator_core import \
    OrchestratorCore
from src.api.entities_api.orchestration.mixins.provider_mixins import \
    _ProviderMixins

load_dotenv()
LOG = LoggingUtility()


class VLLMDefaultBaseWorker(
    VLLMRawStream,  # ← swap: was OllamaNativeStream
    _ProviderMixins,
    OrchestratorCore,
    ABC,
):
    """
    Async base worker for vLLM raw inference.

    Stream pipeline:
        vLLM /v1/completions  →  _stream_vllm_raw()  →  DeltaNormalizer  →  StreamState
    """

    def __init__(
        self,
        *,
        assistant_id: str | None = None,
        thread_id: str | None = None,
        redis=None,
        base_url: str | None = None,
        api_key: str | None = None,
        delete_ephemeral_thread: bool = False,
        assistant_cache_service: Optional[AssistantCache] = None,
        **extra,
    ) -> None:

        # ── Role / identity state ─────────────────────────────────────────
        self.is_deep_research: Optional[bool] = None
        self.is_engineer: Optional[bool] = None
        self._scratch_pad_thread: Optional[str] = None
        self._batfish_owner_user_id: Optional[str] = None
        self._run_user_id: Optional[str] = None

        # ── Delegation / ephemeral state ──────────────────────────────────
        self._delete_ephemeral_thread = delete_ephemeral_thread or extra.get(
            "delete_ephemeral_thread", False
        )
        self.ephemeral_supervisor_id: Optional[str] = None
        self._research_worker_thread: Optional[str] = None
        self._worker_thread: Optional[str] = None

        # ── Tool / decision state ─────────────────────────────────────────
        self._current_tool_call_id: Optional[str] = None
        self._pending_tool_payload: Optional[Dict[str, Any]] = None
        self._decision_payload: Optional[Dict[str, Any]] = None

        # ── Infrastructure ────────────────────────────────────────────────
        self.redis = redis or get_redis_sync()

        if assistant_cache_service:
            self._assistant_cache = assistant_cache_service
        elif "assistant_cache" in extra and isinstance(extra["assistant_cache"], AssistantCache):
            self._assistant_cache = extra["assistant_cache"]

        legacy_config = extra.get("assistant_config") or extra.get("assistant_cache")
        self.assistant_config: Dict[str, Any] = (
            legacy_config if isinstance(legacy_config, dict) else {}
        )

        self._david_client: Any = None
        self.redis = redis or get_redis()
        self.assistant_id = assistant_id
        self.thread_id = thread_id

        # vLLM base URL — prefer explicit arg, then env, then default
        self.base_url = base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000")
        self.api_key = api_key or extra.get("api_key")

        self.model_name = extra.get("model_name", "Qwen/Qwen2.5-3B-Instruct")
        self.max_context_window = extra.get("max_context_window", 128_000)
        self.threshold_percentage = extra.get("threshold_percentage", 0.8)

        self.setup_services()

        if not hasattr(self, "get_function_call_state"):
            LOG.error("CRITICAL: ToolRoutingMixin failed to load.")
            self.get_function_call_state = lambda: None
            self.set_function_call_state = lambda x: None
            self.set_tool_response_state = lambda x: None

        LOG.debug("VLLMDefaultBaseWorker ready (assistant=%s)", assistant_id)

    # ─────────────────────────────────────────────────────────────────────
    # Primary async stream
    # ─────────────────────────────────────────────────────────────────────

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
        Agentic stream pipeline:

          vLLM /v1/completions
              ↓  _stream_vllm_raw()           — raw vLLM text chunks (adapted)
              ↓  DeltaNormalizer              — typed, normalised chunk dicts
              ↓  _handle_chunk_accumulation() — accumulates text, reasoning, tool calls
              ↓  yield JSON strings           — to the SSE response
        """

        # ── Reset per-run mutable state ───────────────────────────────────
        self._run_user_id = None
        self.ephemeral_supervisor_id = None
        self._scratch_pad_thread = None
        self._current_tool_call_id = None
        self._decision_payload = None
        self._tool_queue: List[Dict] = []

        _original_assistant_id = assistant_id

        redis = self.redis
        stream_key = f"stream:{run_id}"
        stop_event = self.start_cancellation_monitor(run_id)

        accumulated: str = ""
        assistant_reply: str = ""
        decision_buffer: str = ""
        current_block: str | None = None
        pre_mapped_model = model

        try:
            if hasattr(self, "_get_model_map") and (mapped := self._get_model_map(model)):
                model = mapped

            self.assistant_id = assistant_id
            await self._ensure_config_loaded()

            self.is_deep_research = self.assistant_config.get("deep_research", False)
            self.is_engineer = self.assistant_config.get("is_engineer", False)

            agent_mode_setting = self.assistant_config.get("agent_mode", False)
            decision_telemetry = self.assistant_config.get("decision_telemetry", True)
            web_access_setting = self.assistant_config.get("web_access", False)

            raw_meta = self.assistant_config.get("meta_data", {})

            is_worker_val = raw_meta.get(
                "is_research_worker", raw_meta.get("research_worker_calling", False)
            )
            research_worker_setting = str(
                is_worker_val
            ).lower() == "true" or self.assistant_config.get("is_research_worker", False)

            is_junior_val = raw_meta.get(
                "junior_engineer", raw_meta.get("junior_engineer_calling", False)
            )
            junior_engineer_setting = str(is_junior_val).lower() == "true"

            if self.is_engineer:
                web_access_setting = False
                research_worker_setting = False
                junior_engineer_setting = False
                self.is_deep_research = False
            elif self.is_deep_research:
                web_access_setting = False
                research_worker_setting = False
                junior_engineer_setting = False
            elif research_worker_setting:
                web_access_setting = True
                junior_engineer_setting = False
                delegation_model = get_delegated_model(requested_model=pre_mapped_model)
                await self._native_exec.update_run_fields(
                    run_id, meta_data={"api_key": api_key, "delegated_model": delegation_model}
                )
            elif junior_engineer_setting:
                web_access_setting = False
                research_worker_setting = False

            LOG.critical(
                "██████ [ROLE CONFIG] SeniorEngineer=%s | DeepResearch=%s | "
                "ResearchWorker=%s | JuniorEngineer=%s | WebAccess=%s ██████",
                self.is_engineer,
                self.is_deep_research,
                research_worker_setting,
                junior_engineer_setting,
                web_access_setting,
            )

            request_meta = kwargs.get("meta_data", {})
            custom_vllm_url = request_meta.get("vllm_base_url")  # ← vllm key

            try:
                run = await self._native_exec.retrieve_run(run_id)
                self._run_user_id = run.user_id
                meta = run.meta_data or {}

                if not custom_vllm_url:
                    custom_vllm_url = meta.get("vllm_base_url")

                if self._batfish_owner_user_id is None:
                    self._batfish_owner_user_id = meta.get("batfish_owner_user_id") or run.user_id

                if self._scratch_pad_thread is None and meta.get("scratch_pad_thread"):
                    self._scratch_pad_thread = meta["scratch_pad_thread"]

            except Exception as exc:
                self._run_user_id = None
                LOG.warning("STREAM ▸ Could not resolve run_user_id: %s", exc)

            await self._handle_role_based_identity_swap(requested_model=pre_mapped_model)

            if self.assistant_id != _original_assistant_id:
                await self._ensure_config_loaded()

            if not self._scratch_pad_thread:
                self._scratch_pad_thread = thread_id

            ctx = await self._set_up_context_window(
                assistant_id=self.assistant_id,
                thread_id=thread_id,
                trunk=True,
                force_refresh=force_refresh,
                agent_mode=agent_mode_setting,
                decision_telemetry=decision_telemetry,
                web_access=web_access_setting,
                deep_research=self.is_deep_research,
                engineer=self.is_engineer,
                research_worker=research_worker_setting,
                junior_engineer=junior_engineer_setting,
            )

            yield json.dumps({"type": "status", "status": "started", "run_id": run_id})

            # ── Stream: vLLM → DeltaNormalizer → StreamState ─────────────
            # ↓ ONE LINE different from Ollama worker ↓
            async for chunk in DeltaNormalizer.async_iter_deltas(
                self._stream_vllm_raw(  # ← was _stream_ollama_raw
                    messages=ctx,
                    model=model,
                    temperature=kwargs.get("temperature", 0.6),
                    max_tokens=kwargs.get("max_tokens", 1024),
                    think=kwargs.get("think", False),
                    base_url=custom_vllm_url or self.base_url,
                ),
                run_id,
            ):
                if stop_event.is_set():
                    break

                (
                    current_block,
                    accumulated,
                    assistant_reply,
                    decision_buffer,
                    should_skip,
                ) = self._handle_chunk_accumulation(
                    chunk,
                    current_block,
                    accumulated,
                    assistant_reply,
                    decision_buffer,
                )

                if should_skip:
                    continue

                yield json.dumps(chunk)

            if current_block:
                accumulated += f"</{current_block}>"

            if decision_buffer:
                try:
                    self._decision_payload = json.loads(decision_buffer.strip())
                except Exception:
                    LOG.warning("Failed to parse decision buffer: %s", decision_buffer[:50])

            tool_calls_batch = self.parse_and_set_function_calls(accumulated, assistant_reply)

            message_to_save = assistant_reply
            final_status = StatusEnum.completed.value

            if tool_calls_batch:
                self._tool_queue = tool_calls_batch
                final_status = StatusEnum.pending_action.value

                tool_calls_structure = []
                for tool in tool_calls_batch:
                    t_id = tool.get("id") or f"call_{uuid.uuid4().hex[:8]}"
                    tool_calls_structure.append(
                        {
                            "id": t_id,
                            "type": "function",
                            "function": {
                                "name": tool.get("name"),
                                "arguments": json.dumps(tool.get("arguments", {})),
                            },
                        }
                    )

                message_to_save = json.dumps(tool_calls_structure)

            yield json.dumps({"type": "status", "status": "processing", "run_id": run_id})

            if message_to_save:
                await self.finalize_conversation(
                    message_to_save, thread_id, self.assistant_id, run_id
                )

            await self._native_exec.update_run_status(run_id, final_status)

            if not tool_calls_batch:
                yield json.dumps({"type": "status", "status": "complete", "run_id": run_id})

        except Exception as exc:
            LOG.error("Stream exception: %s", exc, exc_info=True)
            err = {
                "type": "error",
                "content": f"Stream error: {exc}",
                "run_id": run_id,
            }
            yield json.dumps(err)
            await self._shunt_to_redis_stream(redis, stream_key, err)

        finally:
            stop_event.set()
            self.assistant_id = _original_assistant_id

    # ─────────────────────────────────────────────────────────────────────
    # Synchronous wrapper — identical to Ollama worker
    # ─────────────────────────────────────────────────────────────────────

    def stream_sync(
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
    ) -> Generator[str, None, None]:
        """Synchronous wrapper — identical path A/B logic as Ollama worker."""
        kwargs.update(
            force_refresh=force_refresh,
            stream_reasoning=stream_reasoning,
            api_key=api_key,
        )

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            agen = self.stream(thread_id, message_id, run_id, assistant_id, model, **kwargs)
            try:
                while True:
                    try:
                        yield loop.run_until_complete(agen.__anext__())
                    except StopAsyncIteration:
                        break
            finally:
                try:
                    loop.run_until_complete(agen.aclose())
                except Exception:
                    pass
                loop.close()
                asyncio.set_event_loop(None)
            return

        _SENTINEL = object()
        queue_ref: list = []
        stop_flag = threading.Event()

        def _run_in_thread() -> None:
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            q: _queue_mod.Queue = _queue_mod.Queue()
            queue_ref.append(q)

            async def _drain() -> None:
                agen = self.stream(thread_id, message_id, run_id, assistant_id, model, **kwargs)
                try:
                    async for item in agen:
                        if stop_flag.is_set():
                            break
                        q.put(item)
                finally:
                    try:
                        await agen.aclose()
                    except Exception:
                        pass
                    q.put(_SENTINEL)

            try:
                new_loop.run_until_complete(_drain())
            finally:
                new_loop.close()

        t = threading.Thread(target=_run_in_thread, daemon=True)
        t.start()

        while not queue_ref:
            time.sleep(0.001)

        q = queue_ref[0]
        try:
            while True:
                item = q.get()
                if item is _SENTINEL:
                    break
                yield item
        finally:
            stop_flag.set()

        t.join()
