# src/api/entities_api/orchestration/workers/base_workers/qwen_base.py
from __future__ import annotations

import asyncio
import json
import os
import queue as _queue_mod
import threading
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, Generator, Optional, Union

from dotenv import load_dotenv
from entities_api.cache.assistant_cache import AssistantCache
from entities_api.clients.delta_normalizer import DeltaNormalizer
from entities_api.platform_tools.delegated_model_map.delegation_model_map import (
    get_delegated_model,
)
from projectdavid import StreamEvent
from projectdavid_common.utilities.logging_service import LoggingUtility
from projectdavid_common.validation import StatusEnum

# --- DEPENDENCIES ---
from src.api.entities_api.clients.multimodal_utils import (
    is_multimodal,
    normalise_for_chat,
)
from src.api.entities_api.dependencies import get_redis_sync
from src.api.entities_api.orchestration.engine.orchestrator_core import OrchestratorCore

# --- MIXINS ---
from src.api.entities_api.orchestration.mixins.provider_mixins import _ProviderMixins

load_dotenv()
LOG = LoggingUtility()


class LlamaBaseWorker(
    _ProviderMixins,
    OrchestratorCore,
    ABC,
):
    """
    Async Base for Qwen Providers (Hyperbolic, Together, etc.).
    Handles QwQ-32B/Qwen2.5/Kimi specific stream parsing using DeltaNormalizer.

    Vision support:
        Multimodal messages (hydrated image blocks) are automatically detected
        and normalised to the OpenAI image_url format before dispatch.
        Both TogetherAI and Hyperbolic accept base64 data URIs in this format.

    Provider image limits:
        VISION_MAX_IMAGES controls how many images are passed per request.
        Default is None (unlimited) — suitable for TogetherAI and vLLM.
        Hyperbolic subclasses must override this to 1:

            class HyperbolicQwenWorker(QwenBaseWorker):
                VISION_MAX_IMAGES = 1  # Hyperbolic hard limit per request

    Native tool passing:
        prepare_native_tool_context() extracts tool definitions from the system
        message and passes them natively to the provider via the tools= parameter.
        This is required for providers that no longer parse text-embedded tool
        definitions from the system prompt (e.g. Hyperbolic post-April 2026).
        Providers that do not support native tools receive tools=None and fall
        back to text-embedded definitions in the system prompt.
    """

    # Override in subclasses for providers with per-request image caps.
    # None = unlimited (TogetherAI, vLLM).
    # 1    = Hyperbolic (documented hard limit).
    VISION_MAX_IMAGES: Optional[int] = None

    # Override in subclasses to disable native tool passing for providers
    # that do not support the tools= parameter.
    # True  = extract tools from system message and pass natively (default).
    # False = leave tools embedded in system prompt text.
    USE_NATIVE_TOOLS: bool = True

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
        super().__init__(**extra)

        # 1. Config & Dependencies
        self.api_key = api_key or extra.get("api_key")
        # ephemeral worker config
        # These objects are used for deep search and engineering flows
        self.is_deep_research = None
        self._scratch_pad_thread = None
        self._batfish_owner_user_id: str | None = None
        self.is_engineer = None
        self._delete_ephemeral_thread = delete_ephemeral_thread or extra.get(
            "delete_ephemeral_thread"
        )
        self.ephemeral_supervisor_id = None
        self._research_worker_thread = None
        self._worker_thread = None

        self.redis = redis or get_redis_sync()

        # 2. Cache Service Setup
        if assistant_cache_service:
            self._assistant_cache = assistant_cache_service
        elif "assistant_cache" in extra and isinstance(
            extra["assistant_cache"], AssistantCache
        ):
            self._assistant_cache = extra["assistant_cache"]

        # 3. Config Legacy Support
        legacy_config = extra.get("assistant_config") or extra.get("assistant_cache")
        self.assistant_config: Dict[str, Any] = (
            legacy_config if isinstance(legacy_config, dict) else {}
        )

        self._david_client: Any = None
        self.assistant_id = assistant_id
        self.thread_id = thread_id
        self.base_url = base_url or os.getenv("BASE_URL")

        self.model_name = extra.get("model_name", "qwen/Qwen1_5-32B-Chat")
        self.max_context_window = extra.get("max_context_window", 128000)
        self.threshold_percentage = extra.get("threshold_percentage", 0.8)

        # 4. State Tracking
        self._current_tool_call_id: str | None = None
        self._pending_tool_payload: Optional[Dict[str, Any]] = None
        self._decision_payload: Optional[Dict[str, Any]] = None

        self.setup_services()

        # 5. Mixin Safety Stub
        if not hasattr(self, "get_function_call_state"):
            LOG.error("CRITICAL: ToolRoutingMixin failed to load.")
            self.get_function_call_state = lambda: None
            self.set_function_call_state = lambda x: None
            self.set_tool_response_state = lambda x: None

        LOG.debug(f"{self.__class__.__name__} ready (assistant={assistant_id})")

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
        Unified stream method supporting four distinct assistant roles:

          1. SENIOR ENGINEER (Supervisor)  — is_engineer=True in assistant config
             Plans the incident, delegates to Junior Engineers, writes the Change Request.
             No web access. No SSH. Uses update_scratchpad / read_scratchpad / delegate_engineer_task.

          2. RESEARCH SUPERVISOR           — deep_research=True in assistant config
             Plans the research, delegates to Research Workers. No web access.

          3. RESEARCH WORKER               — is_research_worker=True in assistant config
             Browses the web, appends verified facts to the shared scratchpad.
             Web access enabled. Used exclusively by the deep research workflow.

          4. JUNIOR ENGINEER               — junior_engineer=True in assistant config
             SSHs to network devices, runs delegated command sets, appends raw evidence
             and flags to the shared scratchpad. No web access.

          5. STANDARD ASSISTANT            — no role flags set
             Normal user-facing assistant. Uses whatever is configured on the assistant record.

        Role flags are set via assistant metadata at ephemeral creation time and read here
        from the normalized assistant_config cache. Exactly one role is ever active per
        stream invocation — the conflict resolution block enforces mutual exclusivity.
        """
        # ------------------------------------------------------------------
        # Ephemeral supervisor / delegation state
        # ------------------------------------------------------------------
        self._run_user_id = None
        self.ephemeral_supervisor_id = None
        self._scratch_pad_thread = None
        stop_event = self.start_cancellation_monitor(run_id)

        # Capture original assistant_id BEFORE any identity swap —
        # the swap may mutate self.assistant_id to the supervisor's ID,
        # and we need the original for cleanup / fallback.
        _original_assistant_id = assistant_id

        # --- 1. State Initialization ---
        self._current_tool_call_id = None
        self._decision_payload = None
        self._tool_queue = []
        accumulated: str = ""
        assistant_reply: str = ""
        decision_buffer: str = ""
        current_block: str | None = None
        pre_mapped_model = model

        try:
            # --- 2. Model Resolution ---
            if hasattr(self, "_get_model_map") and (
                mapped := self._get_model_map(model)
            ):
                model = mapped

            self.assistant_id = assistant_id

            # Load the normalized config for the requested assistant from cache.
            # This populates self.assistant_config with metadata, flags, and settings.
            await self._ensure_config_loaded()

            # ------------------------------------------------------------------
            # 3. ROLE FLAG EXTRACTION
            # Read all role signals from the assistant's normalized config.
            # These are set via meta_data at ephemeral creation time (or via the
            # assistant record for standard assistants).
            # ------------------------------------------------------------------
            self.is_deep_research = self.assistant_config.get("deep_research", False)
            self.is_engineer = self.assistant_config.get("is_engineer", False)

            agent_mode_setting = self.assistant_config.get("agent_mode", False)
            decision_telemetry = self.assistant_config.get("decision_telemetry", False)

            # Default web_access from config — may be overridden by role resolution below
            web_access_setting = self.assistant_config.get("web_access", False)

            # Extract from meta_data for dynamic ephemeral flags
            raw_meta = self.assistant_config.get("meta_data", {})

            # Worker role flags — mutually exclusive, enforced below
            is_worker_val = raw_meta.get(
                "is_research_worker", raw_meta.get("research_worker_calling", False)
            )
            research_worker_setting = str(
                is_worker_val
            ).lower() == "true" or self.assistant_config.get(
                "is_research_worker", False
            )

            # Check for "junior_engineer" (and fallback to "junior_engineer_calling" just in case)
            is_junior_val = raw_meta.get(
                "junior_engineer", raw_meta.get("junior_engineer_calling", False)
            )
            junior_engineer_setting = str(is_junior_val).lower() == "true"

            # ------------------------------------------------------------------
            # 4. ROLE CONFLICT RESOLUTION
            # Exactly one role is active per invocation.
            # Priority: Senior Engineer > Research Supervisor > Research Worker > Junior Engineer > Standard
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
                delegation_model = get_delegated_model(requested_model=pre_mapped_model)
                await self._native_exec.update_run_fields(
                    run_id,
                    meta_data={"api_key": api_key, "delegated_model": delegation_model},
                )

            elif research_worker_setting:
                web_access_setting = True
                junior_engineer_setting = False

            elif junior_engineer_setting:
                web_access_setting = False
                research_worker_setting = False

            LOG.critical(
                "██████[ROLE CONFIG] "
                "SeniorEngineer=%s | "
                "DeepResearch=%s | "
                "ResearchWorker=%s | "
                "JuniorEngineer=%s | "
                "WebAccess=%s ██████",
                self.is_engineer,
                self.is_deep_research,
                research_worker_setting,
                junior_engineer_setting,
                web_access_setting,
            )

            # ------------------------------------------------------------------
            # CAPTURE REAL USER ID
            # ------------------------------------------------------------------
            try:
                run = await self._native_exec.retrieve_run(run_id)
                self._run_user_id = run.user_id

                meta = run.meta_data or {}

                meta_owner = meta.get("batfish_owner_user_id")
                meta_scratchpad = meta.get("scratch_pad_thread")

                if self._batfish_owner_user_id is None:
                    self._batfish_owner_user_id = meta_owner or run.user_id

                if self._scratch_pad_thread is None and meta_scratchpad:
                    self._scratch_pad_thread = meta_scratchpad

                LOG.info(
                    "STREAM ▸ Captured run_user_id=%s | batfish_owner=%s | scratch_pad_thread=%s",
                    self._run_user_id,
                    self._batfish_owner_user_id,
                    self._scratch_pad_thread,
                )
            except Exception as e:
                self._run_user_id = None
                LOG.warning("STREAM ▸ Could not resolve run_user_id: %s", e)

            # ------------------------------------------------------------------
            # 5. IDENTITY SWAP (Supervisor roles only)
            # ------------------------------------------------------------------
            await self._handle_role_based_identity_swap(
                requested_model=pre_mapped_model
            )

            if self.assistant_id != _original_assistant_id:
                LOG.info(
                    f"Identity swapped from {_original_assistant_id} to {self.assistant_id}. Reloading config."
                )
                await self._ensure_config_loaded()

            # ------------------------------------------------------------------
            # SCRATCHPAD THREAD PINNING
            # ------------------------------------------------------------------
            if not self._scratch_pad_thread:
                self._scratch_pad_thread = thread_id

            LOG.info(
                "STREAM ▸ Scratchpad thread pinned to: %s", self._scratch_pad_thread
            )

            # ------------------------------------------------------------------
            # 6. CONTEXT WINDOW CONSTRUCTION
            # ------------------------------------------------------------------
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

            if not api_key:
                yield json.dumps({"type": "error", "content": "Missing API key."})
                return

            yield json.dumps({"type": "status", "status": "started", "run_id": run_id})

            client = self._get_client_instance(api_key=api_key)

            # ------------------------------------------------------------------
            # 7. MULTIMODAL NORMALISATION
            # ------------------------------------------------------------------
            if is_multimodal(ctx):
                LOG.info(
                    "QwenBaseWorker ▸ multimodal context detected — normalising to OpenAI "
                    "image_url format (max_images=%s).",
                    self.VISION_MAX_IMAGES,
                )
                ctx = normalise_for_chat(ctx, max_images=self.VISION_MAX_IMAGES)

            LOG.info(
                f"\nRAW_CTX_DUMP_QUEN:\n{json.dumps(ctx, indent=2, ensure_ascii=False)}"
            )

            # ------------------------------------------------------------------
            # 7b. NATIVE TOOL EXTRACTION
            # Extract tool definitions from the system message and pass them
            # natively to the provider via tools=. Required for providers that
            # no longer parse text-embedded tool definitions from the system prompt.
            # Falls back gracefully — if extraction yields nothing, tools=None
            # and the text-embedded definitions remain in the system prompt.
            # ------------------------------------------------------------------
            native_tools = None
            if self.USE_NATIVE_TOOLS:
                ctx, native_tools = self.prepare_native_tool_context(ctx)
                if native_tools:
                    LOG.info(
                        "NATIVE TOOLS ▸ Extracted %d tool(s) for native dispatch: %s",
                        len(native_tools),
                        [t.get("function", {}).get("name") for t in native_tools],
                    )
                else:
                    LOG.warning(
                        "NATIVE TOOLS ▸ USE_NATIVE_TOOLS=True but no tools extracted "
                        "from system message — falling back to text-embedded definitions."
                    )

            # ------------------------------------------------------------------
            # 8. THE STREAM LOOP
            # ------------------------------------------------------------------
            _max_tokens = self.assistant_config.get("max_tokens", None)
            _temperature = self.assistant_config.get(
                "temperature", kwargs.get("temperature", 0.6)
            )
            _top_p = self.assistant_config.get("top_p", None)

            LOG.info(
                "INFERENCE PARAMS ▸ max_tokens=%s | temperature=%s | top_p=%s",
                _max_tokens,
                _temperature,
                _top_p,
            )

            raw_stream = client.stream_chat_completion(
                messages=ctx,
                model=model,
                **({"tools": native_tools} if native_tools else {}),
                **({"max_tokens": _max_tokens} if _max_tokens is not None else {}),
                **({"top_p": _top_p} if _top_p is not None else {}),
                temperature=_temperature,
                stream=True,
            )

            async for chunk in DeltaNormalizer.async_iter_deltas(raw_stream, run_id):
                if stop_event.is_set():
                    break
                LOG.debug("PRE_ACCUM chunk: %s", json.dumps(chunk)[:150])

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
                LOG.warning("PRE_ACCUM chunk: %s", json.dumps(chunk)[:150])
                LOG.warning(
                    "POST_ACCUM skip=%s reply_len=%d", should_skip, len(assistant_reply)
                )
                if should_skip:
                    continue

                yield json.dumps(chunk)

            # Ensure any dangling XML tag is closed cleanly at end of stream
            if current_block:
                accumulated += f"</{current_block}>"

            # ------------------------------------------------------------------
            # 9. POST-STREAM PROCESSING
            # ------------------------------------------------------------------
            if decision_buffer:
                try:
                    self._decision_payload = json.loads(decision_buffer.strip())
                except Exception:
                    LOG.warning(
                        f"Failed to parse decision buffer: {decision_buffer[:50]}..."
                    )

            tool_calls_batch = self.parse_and_set_function_calls(
                accumulated, assistant_reply
            )

            message_to_save = assistant_reply
            final_status = StatusEnum.completed.value

            # ------------------------------------------------------------------
            # 10. TOOL CALL ENVELOPE CONSTRUCTION
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

                message_to_save = json.dumps(tool_calls_structure)

            yield json.dumps(
                {"type": "status", "status": "processing", "run_id": run_id}
            )

            # ------------------------------------------------------------------
            # 11. FINALIZE & PERSIST
            # ------------------------------------------------------------------
            if message_to_save:
                await self.finalize_conversation(
                    message_to_save, thread_id, self.assistant_id, run_id
                )

            await self._native_exec.update_run_status(run_id, final_status)

            if not tool_calls_batch:
                yield json.dumps(
                    {"type": "status", "status": "complete", "run_id": run_id}
                )

        except Exception as exc:
            LOG.error(f"Stream Exception: {exc}")
            yield json.dumps({"type": "error", "content": str(exc), "run_id": run_id})

        finally:
            stop_event.set()

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
        Synchronous wrapper around the async stream() method.

        Supports two calling contexts:

          A) No running event loop (standard sync caller, CLI, Celery worker, test):
             Creates a dedicated event loop, drives the async generator to completion,
             then tears it down cleanly. Zero thread overhead.

          B) Running event loop already exists (sync function called from within a
             thread that shares a loop with async code):
             Spins up a fresh event loop in a background thread and uses a Queue to
             ferry yielded values back to the calling thread, keeping the two loops
             fully isolated.

        Yields the same JSON-serialised strings as stream().
        """
        kwargs.update(
            force_refresh=force_refresh,
            stream_reasoning=stream_reasoning,
            api_key=api_key,
        )

        # ------------------------------------------------------------------
        # Path A — no running loop in this thread
        # ------------------------------------------------------------------
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                agen = self.stream(
                    thread_id, message_id, run_id, assistant_id, model, **kwargs
                )
                while True:
                    try:
                        yield loop.run_until_complete(agen.__anext__())
                    except StopAsyncIteration:
                        break
            finally:
                try:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                except Exception:
                    pass
                loop.close()
                asyncio.set_event_loop(None)
            return

        # ------------------------------------------------------------------
        # Path B — a loop is already running; isolate in a background thread
        # ------------------------------------------------------------------
        _SENTINEL = object()
        queue_ref: list = []

        def _run_in_thread() -> None:
            """Owns its own loop; pushes items onto a thread-safe queue."""
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
                        q.put(item)
                finally:
                    q.put(_SENTINEL)

            try:
                new_loop.run_until_complete(_drain())
            finally:
                try:
                    new_loop.run_until_complete(new_loop.shutdown_asyncgens())
                except Exception:
                    pass
                new_loop.close()

        t = threading.Thread(target=_run_in_thread, daemon=True)
        t.start()

        # Spin until the background thread has created and registered the queue
        while not queue_ref:
            time.sleep(0.001)

        q = queue_ref[0]
        while True:
            item = q.get()
            if item is _SENTINEL:
                break
            yield item

        t.join()
