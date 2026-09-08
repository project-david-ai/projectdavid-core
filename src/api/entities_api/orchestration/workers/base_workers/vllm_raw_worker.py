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
from entities_api.platform_tools.delegated_model_map.delegation_model_map import (
    get_delegated_model,
)
from projectdavid import StreamEvent
from projectdavid_common.utilities.logging_service import LoggingUtility
from projectdavid_common.validation import StatusEnum

# Infrastructure Imports
from src.api.entities_api.db.database import SessionLocal
from src.api.entities_api.dependencies import get_redis, get_redis_sync
from src.api.entities_api.orchestration.engine.orchestrator_core import OrchestratorCore
from src.api.entities_api.orchestration.mixins.provider_mixins import _ProviderMixins
from src.api.entities_api.services.inference_resolver import InferenceResolver

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

        # vLLM base URL — prefer explicit arg, then env, then default
        self.base_url = base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000")
        self.api_key = api_key or extra.get("api_key")

        self.model_name = extra.get("model_name", "Qwen/Qwen2.5-3B-Instruct")
        self.max_context_window = extra.get("max_context_window", 128_000)
        self._configured_max_context_window = self.max_context_window
        self.threshold_percentage = extra.get("threshold_percentage", 0.8)

        self.setup_services()

        if not hasattr(self, "get_function_call_state"):
            LOG.error("CRITICAL: ToolRoutingMixin failed to load.")
            self.get_function_call_state = lambda: None
            self.set_function_call_state = lambda x: None
            self.set_tool_response_state = lambda x: None

        LOG.debug("VLLMDefaultBaseWorker ready (assistant=%s)", assistant_id)

    @staticmethod
    def _compact_json_schema(value: Any) -> Any:
        """Remove prose-only schema metadata while preserving call validity."""
        if isinstance(value, dict):
            return {
                key: VLLMDefaultBaseWorker._compact_json_schema(item)
                for key, item in value.items()
                if key
                not in {
                    "description",
                    "title",
                    "examples",
                    "example",
                }
            }
        if isinstance(value, list):
            return [VLLMDefaultBaseWorker._compact_json_schema(item) for item in value]
        return value

    @classmethod
    def _compact_tool_schema(cls, tool: Dict[str, Any]) -> Dict[str, Any]:
        """Keep tool name/argument contract while dropping token-heavy prose."""
        if not isinstance(tool, dict):
            return tool

        compact = cls._compact_json_schema(tool)
        function = compact.get("function")
        source_function = tool.get("function")
        if isinstance(function, dict) and isinstance(source_function, dict):
            description = source_function.get("description")
            if isinstance(description, str) and description.strip():
                first_line = description.strip().splitlines()[0].strip()
                first_sentence = first_line.split(". ", 1)[0].rstrip(".")
                if first_sentence:
                    function["description"] = first_sentence[:160] + "."
        return compact

    def _compact_local_tool_context(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Replace embedded full tool prose with compact schemas for small contexts."""
        extractor = getattr(self, "prepare_native_tool_context", None)
        if not callable(extractor):
            return messages

        try:
            stripped_messages, native_tools = extractor(messages)
        except Exception as exc:
            LOG.warning("LOCAL CONTEXT TOOLS ▸ extraction failed: %s", exc)
            return messages

        if not native_tools:
            return messages

        compact_tools = [self._compact_tool_schema(tool) for tool in native_tools]
        compact_json = json.dumps(
            compact_tools,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        suffix = "\n\nAvailable tools (compact JSON schemas):\n" + compact_json

        result = [dict(message) for message in stripped_messages]
        system_index = next(
            (
                index
                for index, message in enumerate(result)
                if message.get("role") == "system"
                and isinstance(message.get("content"), str)
            ),
            None,
        )
        if system_index is None:
            result.insert(0, {"role": "system", "content": suffix.lstrip()})
        else:
            result[system_index]["content"] = (
                result[system_index].get("content", "").rstrip() + suffix
            )

        LOG.info(
            "LOCAL CONTEXT TOOLS ▸ compacted %d tool schema(s) to %d chars",
            len(compact_tools),
            len(compact_json),
        )
        return result

    @staticmethod
    def _estimate_local_prompt_tokens(messages: List[Dict[str, Any]]) -> int:
        """Conservative offline estimate when the deployment tokenizer is unavailable."""
        estimate = 0
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                rendered = content
            else:
                rendered = json.dumps(
                    content,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            estimate += max(1, (len(rendered) + 2) // 3) + 16
        return estimate

    @staticmethod
    def _local_completion_limit(
        configured: Any,
        context_window: int,
    ) -> int:
        """Cap requested output for constrained local deployments."""
        try:
            requested = int(configured) if configured is not None else 2_048
        except (TypeError, ValueError):
            requested = 2_048

        requested = max(1, requested)

        if context_window <= 2_048:
            capacity_cap = 256
        elif context_window <= 4_096:
            capacity_cap = 512
        else:
            capacity_cap = min(
                2_048,
                max(
                    512,
                    context_window // 4,
                ),
            )

        return min(
            requested,
            capacity_cap,
        )

    @classmethod
    def _admit_local_context(
        cls,
        messages: List[Dict[str, Any]],
        *,
        context_window: int,
        completion_tokens: int,
        safety_tokens: int = 64,
        minimum_completion_tokens: int = 32,
    ) -> tuple[List[Dict[str, Any]], int]:
        """Protect the current turn and elastically allocate completion capacity."""
        try:
            completion_tokens = int(completion_tokens)
        except (TypeError, ValueError):
            completion_tokens = 256

        completion_tokens = max(
            1,
            completion_tokens,
        )

        safety_tokens = max(
            0,
            int(safety_tokens),
        )

        minimum_completion_tokens = max(
            1,
            int(minimum_completion_tokens),
        )

        usable_window = context_window - safety_tokens

        if usable_window <= minimum_completion_tokens:
            raise ValueError(
                "LOCAL_CONTEXT_CAPACITY_EXCEEDED: "
                "deployment leaves insufficient capacity "
                "after the local safety reserve"
            )

        system_indexes = {
            index
            for index, message in enumerate(messages)
            if message.get("role") == "system"
        }

        latest_user_index = next(
            (
                index
                for index in range(
                    len(messages) - 1,
                    -1,
                    -1,
                )
                if (messages[index].get("role") == "user")
            ),
            None,
        )

        required_indexes = set(system_indexes)

        if latest_user_index is not None:
            # Preserve the complete current turn, including anything
            # occurring after the latest user message.
            required_indexes.update(
                range(
                    latest_user_index,
                    len(messages),
                )
            )
        else:
            latest_non_system = next(
                (
                    index
                    for index in range(
                        len(messages) - 1,
                        -1,
                        -1,
                    )
                    if index not in system_indexes
                ),
                None,
            )

            if latest_non_system is not None:
                required_indexes.update(
                    range(
                        latest_non_system,
                        len(messages),
                    )
                )

        protected = [
            message
            for index, message in enumerate(messages)
            if index in required_indexes
        ]

        protected_tokens = cls._estimate_local_prompt_tokens(protected)

        protected_completion_room = usable_window - protected_tokens

        if protected_completion_room < minimum_completion_tokens:
            raise ValueError(
                "LOCAL_CONTEXT_CAPACITY_EXCEEDED: "
                "protected system/tool context requires "
                f"approximately {protected_tokens} input tokens; "
                f"deployment window {context_window} leaves only "
                f"{max(0, protected_completion_room)} "
                "completion tokens after "
                f"a {safety_tokens}-token safety reserve"
            )

        # This is the important V3 behaviour:
        # shrink the completion reservation instead of rejecting
        # a protected prompt that otherwise fits.
        completion_tokens = min(
            completion_tokens,
            protected_completion_room,
        )

        input_budget = usable_window - completion_tokens

        full_tokens = cls._estimate_local_prompt_tokens(messages)

        if full_tokens <= input_budget:
            admitted = list(messages)

        else:
            chosen = set(required_indexes)

            history_end = (
                latest_user_index
                if latest_user_index is not None
                else min(
                    required_indexes,
                    default=len(messages),
                )
            )

            history_indexes = [
                index for index in range(history_end) if index not in system_indexes
            ]

            # Keep complete historical conversational turns rather
            # than arbitrary isolated messages.
            groups: List[List[int]] = []
            group: List[int] = []

            for index in history_indexes:
                role = messages[index].get("role")

                if role == "user" and group:
                    groups.append(group)
                    group = []

                group.append(index)

            if group:
                groups.append(group)

            for history_group in reversed(groups):
                candidate_indexes = chosen | set(history_group)

                candidate = [
                    message
                    for index, message in enumerate(messages)
                    if index in candidate_indexes
                ]

                candidate_tokens = cls._estimate_local_prompt_tokens(candidate)

                if candidate_tokens > input_budget:
                    break

                chosen.update(history_group)

            admitted = [
                message for index, message in enumerate(messages) if index in chosen
            ]

        estimated_input = cls._estimate_local_prompt_tokens(admitted)

        available_completion = usable_window - estimated_input

        completion_tokens = min(
            completion_tokens,
            available_completion,
        )

        if completion_tokens < minimum_completion_tokens:
            raise ValueError(
                "LOCAL_CONTEXT_CAPACITY_EXCEEDED: "
                "admitted local context leaves only "
                f"{max(0, completion_tokens)} completion tokens "
                f"after a {safety_tokens}-token safety reserve"
            )

        LOG.info(
            "LOCAL CONTEXT BUDGET ? "
            "admitted=%d/%d "
            "estimated_input=%d "
            "completion=%d "
            "requested_cap=%d "
            "safety=%d "
            "window=%d",
            len(admitted),
            len(messages),
            estimated_input,
            completion_tokens,
            min(
                completion_tokens,
                (
                    256
                    if context_window <= 2_048
                    else (512 if context_window <= 4_096 else completion_tokens)
                ),
            ),
            safety_tokens,
            context_window,
        )

        return admitted, completion_tokens

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
        local_max_model_len: Optional[int] = None

        try:
            if hasattr(self, "_get_model_map") and (
                mapped := self._get_model_map(model)
            ):
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

            # ── Stage 6: Dynamic Mesh Resolution ─────────────────────────
            mesh_resolved_url = None
            if not custom_vllm_url:
                db_session = SessionLocal()
                try:
                    mesh_route = InferenceResolver.resolve_vllm_route(db_session, model)
                    mesh_resolved_url = mesh_route.get("url") if mesh_route else None
                    local_max_model_len = (
                        mesh_route.get("max_model_len") if mesh_route else None
                    )
                    if mesh_resolved_url:
                        LOG.info("🌐 Mesh Resolver: %s -> %s", model, mesh_resolved_url)
                    if local_max_model_len:
                        LOG.info(
                            "LOCAL CONTEXT CAPACITY ▸ model=%s max_model_len=%d",
                            model,
                            local_max_model_len,
                        )
                except Exception as e:
                    LOG.error("❌ Mesh Resolution Error: %s", e)
                    if InferenceResolver.requires_registered_route(model):
                        raise
                finally:
                    db_session.close()

            if (
                not custom_vllm_url
                and not mesh_resolved_url
                and InferenceResolver.requires_registered_route(model)
            ):
                raise ValueError(
                    "MODEL_ROUTE_UNAVAILABLE: no active registered endpoint "
                    "for the requested model; activate it before sending"
                )

            # Final Target Logic: 1. Kwargs | 2. Mesh Ledger | 3. Hardcoded Env
            target_url = custom_vllm_url or mesh_resolved_url or self.base_url

            # ── Context Setup ────────────────────────────────────────────
            self.max_context_window = (
                local_max_model_len or self._configured_max_context_window
            )

            await self._handle_role_based_identity_swap(
                requested_model=pre_mapped_model
            )
            if self.assistant_id != _original_assistant_id:
                await self._ensure_config_loaded()

            ctx = await self._set_up_context_window(
                assistant_id=self.assistant_id,
                thread_id=thread_id,
                trunk=True,
                force_refresh=force_refresh,
            )

            # ── Inference parameters from assistant cache ─────────────────
            _max_tokens = self.assistant_config.get("max_tokens", None)
            _temperature = self.assistant_config.get(
                "temperature", kwargs.get("temperature", 0.6)
            )
            _top_p = self.assistant_config.get("top_p", None)

            if local_max_model_len:
                # LOCAL_CONTEXT_ADMISSION_V2
                ctx = self._compact_local_tool_context(ctx)

                _max_tokens = self._local_completion_limit(
                    _max_tokens,
                    local_max_model_len,
                )

                ctx, _max_tokens = self._admit_local_context(
                    ctx,
                    context_window=local_max_model_len,
                    completion_tokens=_max_tokens,
                )

                LOG.info(
                    "LOCAL CONTEXT BUDGET ? " "max_model_len=%d max_tokens=%d",
                    local_max_model_len,
                    _max_tokens,
                )

            LOG.info(
                "INFERENCE PARAMS ▸ max_tokens=%s | temperature=%s | top_p=%s",
                _max_tokens,
                _temperature,
                _top_p,
            )

            # Admission happens before the Ray Serve SSE request is opened.
            yield json.dumps({"type": "status", "status": "started", "run_id": run_id})

            # ── The Stream Cycle ─────────────────────────────────────────
            async for chunk in DeltaNormalizer.async_iter_deltas(
                self._stream_vllm_raw(
                    messages=ctx,
                    model=model,
                    temperature=_temperature,
                    **({"max_tokens": _max_tokens} if _max_tokens is not None else {}),
                    **({"top_p": _top_p} if _top_p is not None else {}),
                    think=kwargs.get("think", False),
                    base_url=target_url,
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
                    chunk, current_block, accumulated, assistant_reply, decision_buffer
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
                    LOG.warning(
                        "Failed to parse decision buffer: %s...", decision_buffer[:50]
                    )

            tool_calls_batch = self.parse_and_set_function_calls(
                accumulated, assistant_reply
            )

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

            yield json.dumps(
                {"type": "status", "status": "processing", "run_id": run_id}
            )

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
            LOG.error("Stream exception: %s", exc, exc_info=True)
            yield json.dumps({"type": "error", "content": str(exc), "run_id": run_id})

        finally:
            stop_event.set()
            self.assistant_id = _original_assistant_id
            self.max_context_window = self._configured_max_context_window

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
