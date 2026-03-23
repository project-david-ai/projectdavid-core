# src/api/entities_api/orchestration/workers/base_workers/vllm_raw_worker.py
"""
VLLMDefaultBaseWorker — Stage 6: Mesh Aware
==========================================
Async base worker for vLLM raw inference.

Now features dynamic Mesh Resolution:
Queries the cluster ledger to find the physical IP of fine-tuned
and standard models across the distributed GPU pool.
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
from entities_api.clients.vllm_raw_stream import VLLMRawStream
from entities_api.platform_tools.delegated_model_map.delegation_model_map import get_delegated_model
from projectdavid import StreamEvent
from projectdavid_common.utilities.logging_service import LoggingUtility
from projectdavid_common.validation import StatusEnum

# Infrastructure Imports
from src.api.entities_api.db.database import SessionLocal  # Main API DB
from src.api.entities_api.dependencies import get_redis, get_redis_sync
from src.api.entities_api.orchestration.engine.orchestrator_core import OrchestratorCore
from src.api.entities_api.orchestration.mixins.provider_mixins import _ProviderMixins
from src.api.entities_api.services.inference_resolver import InferenceResolver  # Mesh Resolver

load_dotenv()
LOG = LoggingUtility()


class VLLMDefaultBaseWorker(
    VLLMRawStream,
    _ProviderMixins,
    OrchestratorCore,
    ABC,
):
    """
    Async base worker for vLLM raw inference.
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

        # ── Reset per-run mutable state ───────────────────────────────────
        self._run_user_id = None
        self.ephemeral_supervisor_id = None
        self._scratch_pad_thread = None
        self._current_tool_call_id = None
        self._decision_payload = None
        self._tool_queue: List[Dict] = []

        _original_assistant_id = assistant_id

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

            # ── Model Config / Metadata ──────────────────────────────────
            request_meta = kwargs.get("meta_data", {})
            custom_vllm_url = request_meta.get("vllm_base_url")

            try:
                run = await self._native_exec.retrieve_run(run_id)
                self._run_user_id = run.user_id
                meta = run.meta_data or {}
                if not custom_vllm_url:
                    custom_vllm_url = meta.get("vllm_base_url")
            except Exception as exc:
                self._run_user_id = None
                LOG.warning("STREAM ▸ Could not resolve run_user_id: %s", exc)

            # 🎯 STAGE 6: DYNAMIC MESH RESOLUTION
            # If no hardcoded URL was provided in the request, query the Mesh Ledger
            mesh_resolved_url = None
            if not custom_vllm_url:
                db_session = SessionLocal()
                try:
                    # Model might be "vllm/david-ft" or just "ftm_..."
                    mesh_resolved_url = InferenceResolver.resolve_vllm_url(db_session, model)
                    if mesh_resolved_url:
                        LOG.info("🌐 Mesh Resolver: %s -> %s", model, mesh_resolved_url)
                except Exception as e:
                    LOG.error("❌ Mesh Resolution Error: %s", e)
                finally:
                    db_session.close()

            # Final Target Logic: 1. Kwargs | 2. Mesh Ledger | 3. Hardcoded Env
            target_url = custom_vllm_url or mesh_resolved_url or self.base_url

            # ── Context Setup ────────────────────────────────────────────
            await self._handle_role_based_identity_swap(requested_model=pre_mapped_model)
            if self.assistant_id != _original_assistant_id:
                await self._ensure_config_loaded()

            ctx = await self._set_up_context_window(
                assistant_id=self.assistant_id,
                thread_id=thread_id,
                trunk=True,
                force_refresh=force_refresh,
                # (Pass flags from assistant config...)
            )

            yield json.dumps({"type": "status", "status": "started", "run_id": run_id})

            # ── The Stream Cycle ─────────────────────────────────────────
            async for chunk in DeltaNormalizer.async_iter_deltas(
                self._stream_vllm_raw(
                    messages=ctx,
                    model=model,
                    temperature=kwargs.get("temperature", 0.6),
                    max_tokens=kwargs.get("max_tokens", 1024),
                    think=kwargs.get("think", False),
                    base_url=target_url,  # <--- DYNAMICALLY RESOLVED
                ),
                run_id,
            ):
                if stop_event.is_set():
                    break

                # (Standard chunk accumulation logic...)
                (
                    current_block,
                    accumulated,
                    assistant_reply,
                    decision_buffer,
                    should_skip,
                ) = self._handle_chunk_accumulation(
                    chunk, current_block, accumulated, assistant_reply, decision_buffer
                )

                if should_skip:
                    continue

                yield json.dumps(chunk)

            # ... (Finalize conversation and update run status logic) ...
            yield json.dumps({"type": "status", "status": "complete", "run_id": run_id})

        except Exception as exc:
            # (Standard error handling)
            LOG.error("Stream exception: %s", exc, exc_info=True)
            yield json.dumps({"type": "error", "content": str(exc), "run_id": run_id})

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
