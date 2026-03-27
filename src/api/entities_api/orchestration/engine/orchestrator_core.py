"""
orchestrator_core.py
────────────────────
The ultra-thin root class.

• Glues together every functional mix-in in the correct MRO order
• Carries only a few shared state fields
• Leaves `stream` + `process_conversation` abstract so each provider
  (Hyperbolic, OpenAI, Together, …) can implement its own wire-format.

Cache ownership
───────────────
AssistantCacheMixin is the single source of truth for _assistant_cache
resolution. It provides the `assistant_cache` property and
`get_assistant_cache()` accessor with a four-level fallback chain:
    1. Injected via __init__ (fast path)
    2. ServiceRegistry (_get_service)
    3. Lazy construction from self.redis
    4. Hard ValueError

OrchestratorCore.__init__ sets self.redis before any cache access so
AssistantCacheMixin path 3 always resolves correctly if paths 1 and 2
are unavailable.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from entities_api.cache.assistant_cache import AssistantCache
from entities_api.dependencies import get_redis_sync
from entities_api.platform_tools.delegated_model_map.delegation_model_map import (
    get_delegated_model,
)
from entities_api.utils.assistant_manager import AssistantManager

# -------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------
from projectdavid import StreamEvent
from projectdavid_common import ToolValidator
from projectdavid_common.utilities.logging_service import LoggingUtility

from src.api.entities_api.constants.platform import PLATFORM_TOOLS

# Mixins
from src.api.entities_api.orchestration.mixins.assistant_cache_mixin import (
    AssistantCacheMixin,
)
from src.api.entities_api.orchestration.mixins.code_interpreter_mixin import (
    CodeInterpreterMixin,
)
from src.api.entities_api.orchestration.mixins.consumer_tool_handlers_mixin import (
    ConsumerToolHandlersMixin,
)
from src.api.entities_api.orchestration.mixins.context_mixin import ContextMixin
from src.api.entities_api.orchestration.mixins.json_utils_mixin import JsonUtilsMixin
from src.api.entities_api.orchestration.mixins.platform_tool_handlers_mixin import (
    PlatformToolHandlersMixin,
)
from src.api.entities_api.orchestration.mixins.service_registry_mixin import (
    ServiceRegistryMixin,
)
from src.api.entities_api.orchestration.mixins.shell_execution_mixin import (
    ShellExecutionMixin,
)
from src.api.entities_api.orchestration.mixins.streaming_mixin import StreamingMixin
from src.api.entities_api.orchestration.mixins.tool_routing_mixin import (
    ToolRoutingMixin,
)

LOG = LoggingUtility()


@dataclass
class StreamState:
    """Container for mutable state during the streaming process."""

    accumulated: str = ""
    assistant_reply: str = ""
    reasoning_reply: str = ""
    decision_buffer: str = ""
    plan_buffer: str = ""
    current_block: str | None = None


class OrchestratorCore(
    ServiceRegistryMixin,
    AssistantCacheMixin,  # Owns: assistant_cache property, get_assistant_cache(),
    # _assistant_cache resolution. Replaces the duplicated
    # property/setter/getter that previously lived here.
    JsonUtilsMixin,
    ContextMixin,
    ToolRoutingMixin,
    PlatformToolHandlersMixin,
    ConsumerToolHandlersMixin,
    StreamingMixin,
    CodeInterpreterMixin,
    ShellExecutionMixin,
    ABC,
):

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

        self.is_deep_research = None
        self._scratch_pad_thread = None

        # 1. Setup Redis — must be set before any AssistantCacheMixin access
        #    so that the lazy-construction fallback (path 3) can resolve correctly.
        self.redis = redis or get_redis_sync()

        # 2. Seed _assistant_cache if explicitly injected.
        #    AssistantCacheMixin owns the property and resolution chain —
        #    we only set the backing attribute here when the caller passes
        #    a ready-made instance, bypassing all fallback paths.
        if assistant_cache_service:
            self._assistant_cache = assistant_cache_service
        elif "assistant_cache" in extra and isinstance(
            extra["assistant_cache"], AssistantCache
        ):
            self._assistant_cache = extra["assistant_cache"]
        # If neither is provided, AssistantCacheMixin will resolve lazily
        # via ServiceRegistry or self.redis on first access.

        # 3. Setup the Data/Config
        legacy_config = extra.get("assistant_config") or extra.get("assistant_cache")
        self.assistant_config: Dict[str, Any] = (
            legacy_config if isinstance(legacy_config, dict) else {}
        )

        # 4. Core State
        self._current_tool_call_id: Optional[str] = None
        self._decision_payload: Any = None

    """
    All behaviour resides in the mix-ins.
    Concrete provider classes only need to implement:
        • stream()
    """

    tool_response: Optional[bool] = None
    function_call: Optional[dict] = None
    _cancelled: bool = False
    code_mode: bool = False

    # --------------------------------------------------------------------------
    # ABSTRACT METHODS
    # --------------------------------------------------------------------------

    @abstractmethod
    def _get_client_instance(self, api_key: str):
        """Return the specific provider client."""
        pass

    @abstractmethod
    def _execute_stream_request(self, client, payload: dict) -> Any:
        """Execute the stream request (Sync or Async-to-Sync)."""
        pass

    @abstractmethod
    async def stream(
        self,
        thread_id: str,
        message_id: str,
        run_id: str,
        assistant_id: str,
        model: Any,
        *,
        force_refresh: bool = False,
        stream_reasoning: bool = False,
        api_key: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Must open a streaming connection to the underlying LLM provider,
        parse deltas into the mix-in JSON chunk format and `yield` them.
        """
        yield ""

    # --------------------------------------------------------------------------
    # CONFIG HELPERS
    # --------------------------------------------------------------------------

    async def load_assistant_config(self):
        """
        Populates self.assistant_config from Redis if not already set.
        """
        if not self.assistant_config and self.assistant_id:
            self.assistant_config = (
                await self.assistant_cache.retrieve(self.assistant_id) or {}
            )
            LOG.debug(f"Loaded config for {self.assistant_id}")

    async def _handle_role_based_identity_swap(self, requested_model: Any) -> None:
        """
        Performs the 'Hot Swap' for specialized agent loops:
        Deep Research (Supervisor -> Workers) OR Engineering (Senior -> Junior)

        user_id is resolved from self._run_user_id (stamped in stream() before
        this method is called) so ephemeral assistants are correctly owned.
        """
        is_engineer_active = getattr(self, "is_engineer", False)

        # 1. ENFORCE MUTUAL EXCLUSIVITY
        if self.is_deep_research and is_engineer_active:
            LOG.error(
                "CRITICAL: Assistant configured with BOTH Deep Research and Engineer. "
                "Defaulting to Engineer."
            )
            self.is_deep_research = False

        # 2. No-op for standard assistants
        if not self.is_deep_research and not is_engineer_active:
            return

        # 3. Resolve owner for ephemeral assistants — must be set before this call
        user_id = getattr(self, "_run_user_id", None)
        if not user_id:
            LOG.error(
                "ORCHESTRATOR ▸ _handle_role_based_identity_swap: "
                "_run_user_id is not set — ephemeral assistants cannot be created without an owner."
            )
            return

        assistant_manager = AssistantManager()

        # ==========================================
        # PATH A: DEEP RESEARCH SWAP
        # ==========================================
        if self.is_deep_research:
            LOG.critical("██████ [DEEP_RESEARCH_MODE_ACTIVE] ██████")
            ephemeral_lead = (
                await assistant_manager.create_ephemeral_research_supervisor(
                    user_id=user_id
                )
            )
            self._worker_thread = await assistant_manager.create_ephemeral_thread(
                user_id=user_id
            )

        # ==========================================
        # PATH B: ENGINEER SWAP
        # ==========================================
        elif is_engineer_active:
            LOG.critical("██████ [ENGINEER_MODE_ACTIVE] ██████")
            ephemeral_lead = await assistant_manager.create_ephemeral_senior_engineer(
                user_id=user_id
            )
            self._worker_thread = await assistant_manager.create_ephemeral_thread(
                user_id=user_id
            )

        # ==========================================
        # COMMON IDENTITY SWAP LOGIC
        # ==========================================
        self.assistant_id = ephemeral_lead.id
        self.ephemeral_supervisor_id = ephemeral_lead.id

        # Flush and Reload Configuration (Grabs the Senior/Supervisor system prompt & tools)
        self.assistant_config = {}
        await self._ensure_config_loaded()

        # Set Delegated Inference Model
        self._delegation_model = get_delegated_model(requested_model=requested_model)

    def _handle_chunk_accumulation(
        self,
        chunk: Dict[str, Any],
        current_block: str | None,
        accumulated: str,
        assistant_reply: str,
        decision_buffer: str = "",
    ) -> tuple[str | None, str, str, str, bool]:
        ctype = chunk.get("type")
        ccontent = chunk.get("content") or ""
        should_skip = False

        if ctype == "content":
            if current_block:
                accumulated += f"</{current_block}>"
                current_block = None
            assistant_reply += ccontent
            accumulated += ccontent

        elif ctype == "call_arguments":
            if current_block != "fc":
                if current_block:
                    accumulated += f"</{current_block}>"
                accumulated += "<fc>"
                current_block = "fc"
            accumulated += ccontent
            should_skip = True

        elif ctype == "reasoning":
            if current_block != "think":
                if current_block:
                    accumulated += f"</{current_block}>"
                accumulated += "<think>"
                current_block = "think"
            accumulated += ccontent

        elif ctype == "plan":
            if current_block != "plan":
                if current_block:
                    accumulated += f"</{current_block}>"
                accumulated += "<plan>"
                current_block = "plan"
            accumulated += ccontent

        elif ctype == "decision":
            if current_block != "decision":
                if current_block:
                    accumulated += f"</{current_block}>"
                accumulated += "<decision>"
                current_block = "decision"
            accumulated += ccontent
            decision_buffer += ccontent

        return current_block, accumulated, assistant_reply, decision_buffer, should_skip

    def _update_stream_state(self, chunk: dict, state: StreamState) -> None:
        ctype = chunk.get("type")
        ccontent = chunk.get("content") or ""

        if ctype == "content":
            if state.current_block == "fc":
                state.accumulated += "</fc>"
            elif state.current_block == "think":
                state.accumulated += "</think>"
            elif state.current_block == "plan":
                state.accumulated += "</plan>"
            state.current_block = None
            state.assistant_reply += ccontent
            state.accumulated += ccontent

        elif ctype == "call_arguments":
            if state.current_block != "fc":
                if state.current_block == "think":
                    state.accumulated += "</think>"
                elif state.current_block == "plan":
                    state.accumulated += "</plan>"
                state.accumulated += "<fc>"
                state.current_block = "fc"
            state.accumulated += ccontent

        elif ctype == "reasoning":
            if state.current_block != "think":
                if state.current_block == "fc":
                    state.accumulated += "</fc>"
                elif state.current_block == "plan":
                    state.accumulated += "</plan>"
                state.accumulated += "<think>"
                state.current_block = "think"
            state.reasoning_reply += ccontent

        elif ctype == "plan":
            if state.current_block != "plan":
                if state.current_block == "fc":
                    state.accumulated += "</fc>"
                elif state.current_block == "think":
                    state.accumulated += "</think>"
                state.accumulated += "<plan>"
                state.current_block = "plan"
            state.plan_buffer += ccontent
            state.accumulated += ccontent

        elif ctype == "decision":
            state.decision_buffer += ccontent
            if state.current_block == "fc":
                state.accumulated += "</fc>"
            elif state.current_block == "think":
                state.accumulated += "</think>"
            elif state.current_block == "plan":
                state.accumulated += "</plan>"
            state.current_block = "decision"

    async def _ensure_config_loaded(self):
        if not self.assistant_id:
            LOG.warning("⚠️ Cannot load config: No assistant_id provided.")
            return

        try:
            cached_data = await self.assistant_cache.retrieve(self.assistant_id)
            if cached_data:
                self.assistant_config = cached_data
                LOG.info(
                    f"✅ Config Loaded for {self.assistant_id} | "
                    f"AgentMode: {self.assistant_config.get('agent_mode')}"
                )
            else:
                LOG.warning(
                    f"⚠️ Cache Miss for {self.assistant_id} — fetching from source"
                )
                fresh_config = await self._fetch_assistant_config_from_db(
                    self.assistant_id
                )
                if fresh_config:
                    self.assistant_config = fresh_config
                    await self.assistant_cache.store(self.assistant_id, fresh_config)
                    LOG.info(f"✅ Config rehydrated from DB for {self.assistant_id}")
                else:
                    LOG.error(
                        f"❌ FATAL: No config found anywhere for {self.assistant_id}"
                    )

        except Exception as e:
            LOG.error(f"❌ Error loading assistant config: {e}")

    def _build_tool_structure(self, batch):
        structure = []
        for tool in batch:
            tool_id = tool.get("id") or f"call_{uuid.uuid4().hex[:8]}"
            structure.append(
                {
                    "id": tool_id,
                    "type": "function",
                    "function": {
                        "name": tool.get("name"),
                        "arguments": (
                            json.dumps(tool.get("arguments", {}))
                            if isinstance(tool.get("arguments"), dict)
                            else tool.get("arguments")
                        ),
                    },
                }
            )
        return structure

    # --------------------------------------------------------------------------
    # CORE ORCHESTRATION LOOP (Recursive Level 3)
    # --------------------------------------------------------------------------
    async def process_conversation(
        self,
        thread_id: str,
        message_id: Optional[str],
        run_id: str,
        assistant_id: str,
        model: Any,
        api_key: Optional[str] = None,
        max_turns: int = 200,
        **kwargs,
    ) -> AsyncGenerator[Union[str, StreamEvent], None]:
        """
        Level 3 Recursive Orchestrator (Batch-Aware).

        Run lifecycle guarantees:
        - started_at   : written on Turn 1
        - current_turn : written every turn
        - completed_at : written on clean text-only exit
        - failed_at    : written on stream failure, max-turn failsafe, or outer exception
        - last_error   : written on failure paths
        - incomplete_details : written on max-turn failsafe
        """

        self._scratch_pad_thread = None
        _original_assistant_id = assistant_id

        await self._ensure_config_loaded()

        tool_validator = ToolValidator()
        turn_count = 0
        current_message_id = message_id

        try:
            while turn_count < max_turns:
                turn_count += 1
                LOG.info(f"ORCHESTRATOR ▸ Turn {turn_count} start [Run: {run_id}]")

                # ------------------------------------------------------------------
                # LIFECYCLE STAMP — started_at (turn 1) + current_turn (every turn)
                # ------------------------------------------------------------------
                try:
                    fields = {"current_turn": turn_count}
                    if turn_count == 1:
                        fields["started_at"] = int(time.time())

                    await self._native_exec.update_run_fields(run_id, **fields)

                except Exception as lifecycle_exc:
                    LOG.warning(
                        f"ORCHESTRATOR ▸ Failed to write turn state (non-fatal): {lifecycle_exc}"
                    )

                # ------------------------------------------------------------------
                # RESET INTERNAL TURN STATE
                # ------------------------------------------------------------------
                self.set_tool_response_state(False)
                self.set_function_call_state(None)
                self._current_tool_call_id = None

                # ------------------------------------------------------------------
                # INFERENCE TURN
                # ------------------------------------------------------------------
                try:
                    async for chunk in self.stream(
                        thread_id=thread_id,
                        message_id=current_message_id,
                        run_id=run_id,
                        assistant_id=assistant_id,
                        model=model,
                        force_refresh=(turn_count > 1),
                        api_key=api_key,
                        **kwargs,
                    ):
                        yield chunk

                except Exception as stream_exc:
                    LOG.error(
                        f"ORCHESTRATOR ▸ Turn {turn_count} stream failed: {stream_exc}"
                    )

                    try:
                        await self._native_exec.update_run_fields(
                            run_id,
                            last_error=str(stream_exc),
                            failed_at=int(time.time()),
                        )
                    except Exception as lifecycle_exc:
                        LOG.warning(
                            f"ORCHESTRATOR ▸ Failed to write stream error (non-fatal): {lifecycle_exc}"
                        )

                    yield json.dumps(
                        {
                            "type": "error",
                            "content": f"Stream failure: {stream_exc}",
                            "run_id": run_id,
                        }
                    )
                    break

                # ------------------------------------------------------------------
                # CHECK FOR TOOL CALLS
                # ------------------------------------------------------------------
                batch: List[Dict] = self.get_function_call_state()

                if not batch:
                    LOG.info(f"ORCHESTRATOR ▸ Turn {turn_count} completed with text.")

                    try:
                        await self._native_exec.update_run_fields(
                            run_id,
                            completed_at=int(time.time()),
                        )
                    except Exception as lifecycle_exc:
                        LOG.warning(
                            f"ORCHESTRATOR ▸ Failed to write completed_at (non-fatal): {lifecycle_exc}"
                        )

                    break

                # ------------------------------------------------------------------
                # TOOL VALIDATION + SDK DETECTION
                # ------------------------------------------------------------------
                has_sdk_user_tool = False

                for call in batch:
                    if "function" in call and isinstance(call["function"], dict):
                        tool_name = call["function"].get("name")
                        args = call["function"].get("arguments", {})
                    else:
                        tool_name = call.get("name")
                        args = call.get("arguments", {})

                    if tool_name not in PLATFORM_TOOLS:
                        has_sdk_user_tool = True

                    validation_event = tool_validator.validate_args(
                        tool_name=tool_name,
                        args=args,
                    )

                    if validation_event:
                        yield validation_event

                # ------------------------------------------------------------------
                # TOOL EXECUTION TURN
                # ------------------------------------------------------------------
                async for chunk in self.process_tool_calls(
                    thread_id=thread_id,
                    scratch_pad_thread=self._scratch_pad_thread,
                    run_id=run_id,
                    assistant_id=self.assistant_id,
                    tool_call_id=self._current_tool_call_id,
                    model=model,
                    api_key=api_key,
                    decision=self._decision_payload,
                ):
                    yield chunk

                # ------------------------------------------------------------------
                # RECURSION DECISION
                # ------------------------------------------------------------------
                if has_sdk_user_tool:
                    LOG.info(
                        "ORCHESTRATOR ▸ SDK tool detected. Handing control to SDK."
                    )
                    return

                LOG.info(
                    f"ORCHESTRATOR ▸ Platform batch {turn_count} complete. Looping internally..."
                )
                await asyncio.sleep(0.5)

                current_message_id = None
                continue

            # ----------------------------------------------------------------------
            # FAILSAFE — MAX TURNS
            # ----------------------------------------------------------------------
            if turn_count >= max_turns:
                LOG.error("ORCHESTRATOR ▸ Max turns reached.")

                try:
                    await self._native_exec.update_run_fields(
                        run_id,
                        failed_at=int(time.time()),
                        incomplete_details=(
                            f"Max turns ({max_turns}) reached without clean completion. "
                            f"Last tool batch size: "
                            f"{len(self.get_function_call_state() or [])}."
                        ),
                    )
                except Exception as lifecycle_exc:
                    LOG.warning(
                        f"ORCHESTRATOR ▸ Failed to write max-turns state (non-fatal): {lifecycle_exc}"
                    )

                yield json.dumps(
                    {
                        "type": "error",
                        "content": "Maximum conversation turns exceeded",
                        "run_id": run_id,
                    }
                )

        # ----------------------------------------------------------------------
        # OUTER EXCEPTION GUARD
        # ----------------------------------------------------------------------
        except Exception as exc:
            LOG.error(f"ORCHESTRATOR ▸ Unhandled exception: {exc}")

            try:
                await self._native_exec.update_run_fields(
                    run_id,
                    last_error=str(exc),
                    failed_at=int(time.time()),
                )
            except Exception as lifecycle_exc:
                LOG.warning(
                    f"ORCHESTRATOR ▸ Failed to persist outer exception (non-fatal): {lifecycle_exc}"
                )

            yield json.dumps(
                {
                    "type": "error",
                    "content": str(exc),
                    "run_id": run_id,
                }
            )

        # ----------------------------------------------------------------------
        # IDENTITY TEARDOWN — ALWAYS EXECUTES
        # ----------------------------------------------------------------------
        finally:
            if self.ephemeral_supervisor_id:
                try:
                    await self._ephemeral_clean_up(
                        assistant_id=self.ephemeral_supervisor_id,
                        thread_id=thread_id,
                        delete_thread=False,
                    )
                except Exception as cleanup_exc:
                    LOG.warning(
                        f"ORCHESTRATOR ▸ Ephemeral cleanup failed (non-fatal): {cleanup_exc}"
                    )

            self.assistant_id = _original_assistant_id
            self.ephemeral_supervisor_id = None

            LOG.info(
                f"ORCHESTRATOR ▸ Identity restored to {_original_assistant_id}. "
                f"Ephemeral state cleared."
            )
