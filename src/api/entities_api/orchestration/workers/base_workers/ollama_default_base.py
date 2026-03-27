# src/api/entities_api/workers/default_worker.py
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
from entities_api.clients.ollama_client import OllamaNativeStream
from entities_api.platform_tools.delegated_model_map.delegation_model_map import (
    get_delegated_model,
)
from projectdavid import StreamEvent
from projectdavid_common.utilities.logging_service import LoggingUtility
from projectdavid_common.validation import StatusEnum

from src.api.entities_api.dependencies import get_redis, get_redis_sync
from src.api.entities_api.orchestration.engine.orchestrator_core import OrchestratorCore
from src.api.entities_api.orchestration.mixins.provider_mixins import _ProviderMixins

load_dotenv()
LOG = LoggingUtility()


class OllamaDefaultBaseWorker(
    OllamaNativeStream,
    _ProviderMixins,
    OrchestratorCore,
    ABC,
):
    """
    Async base worker for local Ollama models.

    Stream pipeline:
        Ollama /api/chat  →  _stream_ollama_raw()  →  DeltaNormalizer  →  StreamState
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
        elif "assistant_cache" in extra and isinstance(
            extra["assistant_cache"], AssistantCache
        ):
            self._assistant_cache = extra["assistant_cache"]

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

        self.model_name = extra.get("model_name", "qwen3:4b")
        self.max_context_window = extra.get("max_context_window", 128_000)
        self.threshold_percentage = extra.get("threshold_percentage", 0.8)

        self.setup_services()

        if not hasattr(self, "get_function_call_state"):
            LOG.error("CRITICAL: ToolRoutingMixin failed to load.")
            self.get_function_call_state = lambda: None
            self.set_function_call_state = lambda x: None
            self.set_tool_response_state = lambda x: None

        LOG.debug("DefaultBaseWorker ready (assistant=%s)", assistant_id)

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

          Ollama /api/chat
              ↓  _stream_ollama_raw()      — raw Ollama dict chunks (no pre-processing)
              ↓  DeltaNormalizer           — typed, normalized chunk dicts
              ↓  _handle_chunk_accumulation()    — accumulates text, reasoning, tool calls
              ↓  yield JSON strings        — to the SSE response
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
            # ── Model alias resolution ────────────────────────────────────
            if hasattr(self, "_get_model_map") and (
                mapped := self._get_model_map(model)
            ):
                model = mapped

            # ── Config ───────────────────────────────────────────────────
            self.assistant_id = assistant_id
            await self._ensure_config_loaded()

            # ------------------------------------------------------------------
            # 3. ROLE FLAG EXTRACTION
            # Read all role signals from the assistant's normalized config.
            # ------------------------------------------------------------------
            self.is_deep_research = self.assistant_config.get("deep_research", False)
            self.is_engineer = self.assistant_config.get("is_engineer", False)

            agent_mode_setting = self.assistant_config.get("agent_mode", False)
            decision_telemetry = self.assistant_config.get("decision_telemetry", True)

            # Default web_access from config
            web_access_setting = self.assistant_config.get("web_access", False)

            # Extract from meta_data for dynamic ephemeral flags
            raw_meta = self.assistant_config.get("meta_data", {})

            # Worker role flags
            is_worker_val = raw_meta.get(
                "is_research_worker", raw_meta.get("research_worker_calling", False)
            )
            research_worker_setting = str(
                is_worker_val
            ).lower() == "true" or self.assistant_config.get(
                "is_research_worker", False
            )

            # Check for "junior_engineer"
            is_junior_val = raw_meta.get(
                "junior_engineer", raw_meta.get("junior_engineer_calling", False)
            )
            junior_engineer_setting = str(is_junior_val).lower() == "true"

            # ------------------------------------------------------------------
            # 4. ROLE CONFLICT RESOLUTION
            # ------------------------------------------------------------------
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
                # --------------------------------------------------------
                # Pass the inference api key through the run
                # Pass the delegated research model through the run
                # object — trusted internally write, no ownership check.
                # --------------------------------------------------------
                delegation_model = get_delegated_model(requested_model=pre_mapped_model)
                await self._native_exec.update_run_fields(
                    run_id,
                    meta_data={"api_key": api_key, "delegated_model": delegation_model},
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

            # ------------------------------------------------------------------
            # CAPTURE REAL USER ID — before any identity swap mutates state.
            # ------------------------------------------------------------------

            # Extract dynamically injected metadata from the stream kwargs
            request_meta = kwargs.get("meta_data", {})
            custom_ollama_url = request_meta.get("ollama_base_url")

            try:
                # [FIXED SDK REMOVAL] Replaced HTTP SDK usage with native execution DB lookup
                run = await self._native_exec.retrieve_run(run_id)
                self._run_user_id = run.user_id

                meta = run.meta_data or {}

                # Fallback to run-level metadata if not passed directly in the stream
                if not custom_ollama_url:
                    custom_ollama_url = meta.get("ollama_base_url")

                if self._batfish_owner_user_id is None:
                    self._batfish_owner_user_id = (
                        meta.get("batfish_owner_user_id") or run.user_id
                    )

                if self._scratch_pad_thread is None and meta.get("scratch_pad_thread"):
                    self._scratch_pad_thread = meta["scratch_pad_thread"]

                LOG.info(
                    "STREAM ▸ Captured run_user_id=%s | batfish_owner=%s | scratch_pad_thread=%s",
                    self._run_user_id,
                    self._batfish_owner_user_id,
                    self._scratch_pad_thread,
                )
            except Exception as exc:
                self._run_user_id = None
                LOG.warning("STREAM ▸ Could not resolve run_user_id: %s", exc)

            # ------------------------------------------------------------------
            # 5. IDENTITY SWAP & RELOAD (Supervisor roles only)
            # ------------------------------------------------------------------
            await self._handle_role_based_identity_swap(
                requested_model=pre_mapped_model
            )

            # --- CRITICAL FIX: Reload config if identity was swapped! ---
            if self.assistant_id != _original_assistant_id:
                LOG.info(
                    f"Identity swapped from {_original_assistant_id} to {self.assistant_id}. Reloading config."
                )
                await self._ensure_config_loaded()

            # Scratchpad: prefer meta_data value; fall back to thread_id
            if not self._scratch_pad_thread:
                self._scratch_pad_thread = thread_id
            LOG.info("STREAM ▸ Scratchpad pinned to: %s", self._scratch_pad_thread)

            # ── Build context window ──────────────────────────────────────
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
            LOG.info(
                "RAW_CTX_DUMP (%d messages):\n%s",
                len(ctx),
                json.dumps(ctx, indent=2, ensure_ascii=False),
            )

            yield json.dumps({"type": "status", "status": "started", "run_id": run_id})

            # ── Stream: Ollama → DeltaNormalizer → StreamState ────────────
            async for chunk in DeltaNormalizer.async_iter_deltas(
                self._stream_ollama_raw(
                    messages=ctx,
                    model=model,
                    temperature=kwargs.get("temperature", 0.6),
                    max_tokens=kwargs.get("max_tokens", 10_000),
                    think=kwargs.get("think", False),
                    base_url=custom_ollama_url,  # <-- Pass dynamically extracted custom URL down
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

            # Ensure any dangling XML tag is closed cleanly at end of stream
            if current_block:
                accumulated += f"</{current_block}>"

            # 8a. Extract Decision Payload from buffered XML block (if any)
            if decision_buffer:
                try:
                    self._decision_payload = json.loads(decision_buffer.strip())
                except Exception:
                    LOG.warning(
                        f"Failed to parse decision buffer: {decision_buffer[:50]}..."
                    )

            # 8b. Extract Tool Calls from accumulated stream output
            tool_calls_batch = self.parse_and_set_function_calls(
                accumulated, assistant_reply
            )

            message_to_save = assistant_reply
            final_status = StatusEnum.completed.value

            # ------------------------------------------------------------------
            # 9. TOOL CALL ENVELOPE CONSTRUCTION
            # If the assistant emitted tool calls, build the standardised envelope
            # and flag the run as pending_action so the orchestrator picks it up.
            # ------------------------------------------------------------------
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

                # Persist the structural representation, not the raw text
                message_to_save = json.dumps(tool_calls_structure)

            yield json.dumps(
                {"type": "status", "status": "processing", "run_id": run_id}
            )

            if message_to_save:
                # [FIX]: Use self.assistant_id to save under the supervisor's ID (if applicable)
                await self.finalize_conversation(
                    message_to_save, thread_id, self.assistant_id, run_id
                )

            await self._native_exec.update_run_status(run_id, final_status)

            if not tool_calls_batch:
                yield json.dumps(
                    {"type": "status", "status": "complete", "run_id": run_id}
                )

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
            LOG.info(
                "ORCHESTRATOR ▸ Identity restored to %s. Ephemeral state cleared.",
                _original_assistant_id,
            )

    # ─────────────────────────────────────────────────────────────────────
    # Synchronous wrapper
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
        """
        Synchronous wrapper around stream().

        Path A — no running event loop (CLI, Celery, test runner):
            Creates a dedicated loop, drives the async generator, then tears
            it down cleanly.

        Path B — a loop is already running (called from a sync shim inside
            an async context):
            Spins a background thread with its own loop and ferries items
            back through a thread-safe queue, keeping both loops isolated.
        """
        kwargs.update(
            force_refresh=force_refresh,
            stream_reasoning=stream_reasoning,
            api_key=api_key,
        )

        # ── Path A ────────────────────────────────────────────────────────
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            agen = self.stream(
                thread_id, message_id, run_id, assistant_id, model, **kwargs
            )
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
                try:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                except Exception:
                    pass
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    try:
                        loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )
                    except Exception:
                        pass
                loop.close()
                asyncio.set_event_loop(None)
            return

        # ── Path B ────────────────────────────────────────────────────────
        _SENTINEL = object()
        queue_ref: list = []
        stop_flag = threading.Event()

        def _run_in_thread() -> None:
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            q: _queue_mod.Queue = _queue_mod.Queue()
            queue_ref.append(q)

            async def _drain() -> None:
                agen = self.stream(
                    thread_id, message_id, run_id, assistant_id, model, **kwargs
                )
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
                try:
                    new_loop.run_until_complete(new_loop.shutdown_asyncgens())
                except Exception:
                    pass
                pending = asyncio.all_tasks(new_loop)
                for task in pending:
                    task.cancel()
                if pending:
                    try:
                        new_loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )
                    except Exception:
                        pass
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
