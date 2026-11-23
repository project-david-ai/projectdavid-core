"""
orchestrator_core.py
src/api/entities_api/orchestration/engine/orchestrator_core.py

The root orchestration class.

AssistantCacheMixin owns assistant-cache resolution.

Execution ledger:
- One durable supervisor actor slot per root Run epoch.
- Assistant identity restored across inference turns and SDK continuations.
- No supervisor deletion during nonterminal SDK handoff.
- Terminal cleanup records actor retirement before Assistant soft-deletion.
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
from projectdavid import StreamEvent
from projectdavid_common import ToolValidator
from projectdavid_common.utilities.logging_service import LoggingUtility
from projectdavid_common.validation import StatusEnum

from src.api.entities_api.constants.platform import PLATFORM_TOOLS
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
from src.api.entities_api.utilities.assistant_manager import AssistantManager

LOG = LoggingUtility()


@dataclass
class StreamState:
    """Mutable state during streaming."""

    accumulated: str = ""
    assistant_reply: str = ""
    reasoning_reply: str = ""
    decision_buffer: str = ""
    plan_buffer: str = ""
    current_block: str | None = None


def _should_handoff_to_sdk_user_tool(
    *,
    tool_name: Optional[str],
    platform_tools: Any,
    mcp_executor: Any,
    authorized_consumer_tool_names: frozenset[str],
) -> bool:
    """
    Only authorized SDK consumer tools may trigger an SDK handoff.

    Unknown or hallucinated tool names remain inside Core so their
    persisted rejection can drive another inference turn.
    """
    return bool(
        tool_name
        and tool_name not in platform_tools
        and mcp_executor is None
        and tool_name in authorized_consumer_tool_names
    )


class OrchestratorCore(
    ServiceRegistryMixin,
    AssistantCacheMixin,
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

        self._current_tool_call_id: Optional[str] = None
        self._decision_payload: Any = None

        # Ledger references are request-scoped; the database is durable.
        self._ledger_epoch_id: Optional[str] = None
        self._ledger_actor_id: Optional[str] = None
        self._active_run_id: Optional[str] = None

    tool_response: Optional[bool] = None
    function_call: Optional[dict] = None
    _cancelled: bool = False
    code_mode: bool = False

    # ------------------------------------------------------------------
    # Abstract provider interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _get_client_instance(self, api_key: str):
        """Return the provider-specific client."""
        pass

    @abstractmethod
    def _execute_stream_request(self, client, payload: dict) -> Any:
        """Execute the provider-specific streaming request."""
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
        """Provider-specific streaming implementation."""
        yield ""

    # ------------------------------------------------------------------
    # Assistant configuration
    # ------------------------------------------------------------------

    async def load_assistant_config(self):
        if not self.assistant_config and self.assistant_id:
            self.assistant_config = (
                await self.assistant_cache.retrieve(self.assistant_id) or {}
            )
            LOG.debug(f"Loaded config for {self.assistant_id}")

    # ------------------------------------------------------------------
    # Durable role-based supervisor identity
    # ------------------------------------------------------------------

    async def _handle_role_based_identity_swap(
        self,
        requested_model: Any,
        *,
        run_id: Optional[str] = None,
    ) -> None:
        """
        Resolve one supervisor identity per root-Run execution epoch.

        The ledger is authoritative across provider turns and SDK
        continuations.

        A previously activated actor is reused. If its Assistant has
        been deleted or become unavailable, fail rather than silently
        assigning a different identity to the same actor.
        """
        is_engineer_active = bool(getattr(self, "is_engineer", False))

        if self.is_deep_research and is_engineer_active:
            LOG.error(
                "CRITICAL: Assistant configured with BOTH Deep Research "
                "and Engineer. Defaulting to Engineer."
            )
            self.is_deep_research = False

        if not self.is_deep_research and not is_engineer_active:
            return

        user_id = getattr(self, "_run_user_id", None)

        if not user_id:
            raise RuntimeError(
                "Supervisor reservation requires the authenticated " "Run owner."
            )

        # Older provider streams call this method without run_id.
        # Resolve only from the current process_conversation invocation;
        # never fall back to an assistant or cross-run identity.
        active_run_id = self._active_run_id
        if run_id is not None and active_run_id is not None and run_id != active_run_id:
            raise RuntimeError("Supervisor reservation run_id differs from active Run.")
        if run_id is None:
            run_id = active_run_id
        if not run_id:
            raise RuntimeError("Supervisor reservation requires run_id.")

        role = "research_supervisor" if self.is_deep_research else "senior_engineer"

        reservation = await self._native_exec.reserve_execution_actor(
            run_id=run_id,
            slot="supervisor",
            execution_role=role,
            ephemeral=True,
        )

        epoch = reservation["epoch"]
        actor = reservation["actor"]

        self._ledger_epoch_id = epoch["id"]
        self._ledger_actor_id = actor["id"]

        if epoch["run_id"] != run_id or epoch["root_run_id"] != run_id:
            raise RuntimeError("Supervisor identity swap is only valid for a root Run.")

        if actor["state"] == "retired":
            raise RuntimeError(
                "Execution actor was retired; a new execution " "attempt is required."
            )

        assistant_manager = AssistantManager()
        supervisor_id = actor.get("assistant_id_snapshot")

        if supervisor_id:
            # Existing epoch: restore the recorded identity.
            ephemeral_lead = await self._native_exec.retrieve_assistant(supervisor_id)

            if (
                not ephemeral_lead
                or getattr(
                    ephemeral_lead,
                    "deleted_at",
                    None,
                )
                is not None
            ):
                raise RuntimeError(
                    "Previously recorded supervisor Assistant is "
                    "unavailable; explicit epoch recovery is required."
                )

            LOG.info(
                "ORCHESTRATOR ▸ Reusing supervisor %s (epoch=%s)",
                supervisor_id,
                epoch["id"],
            )

        else:
            # No Assistant identity has yet been attached to this slot.
            # Assistant creation happens before ledger activation.
            #
            # A concurrent creator could win the write-once activation.
            # If that happens, reconcile to the winning identity.

            if role == "research_supervisor":
                ephemeral_lead = (
                    await assistant_manager.create_ephemeral_research_supervisor(
                        user_id=user_id
                    )
                )
            else:
                ephemeral_lead = (
                    await assistant_manager.create_ephemeral_senior_engineer(
                        user_id=user_id
                    )
                )

            try:
                activated = await self._native_exec.activate_execution_actor(
                    actor_id=actor["id"],
                    assistant_id=ephemeral_lead.id,
                    model=getattr(
                        ephemeral_lead,
                        "model",
                        None,
                    ),
                )

                recorded_id = activated["assistant_id_snapshot"]

            except Exception:
                # Activation may have committed before its response
                # failed, or another process may have activated first.
                #
                # Read the durable identity before attempting cleanup.

                try:
                    refreshed = await self._native_exec.reserve_execution_actor(
                        run_id=run_id,
                        slot="supervisor",
                        execution_role=role,
                        ephemeral=True,
                    )

                    recorded_id = refreshed["actor"].get("assistant_id_snapshot")

                except Exception:
                    LOG.warning(
                        "ORCHESTRATOR ▸ Unable to reconcile supervisor "
                        "activation for epoch %s; preserving Assistant "
                        "%s for operator recovery.",
                        epoch["id"],
                        ephemeral_lead.id,
                    )
                    raise

                if not recorded_id:
                    LOG.error(
                        "ORCHESTRATOR ▸ Supervisor activation unresolved "
                        "for epoch %s; Assistant %s requires "
                        "reconciliation.",
                        epoch["id"],
                        ephemeral_lead.id,
                    )
                    raise RuntimeError(
                        "Supervisor activation did not persist " "an identity."
                    )

            if recorded_id != ephemeral_lead.id:
                # The other creator won. Delete only our losing
                # Assistant, never the identity recorded in the ledger.
                try:
                    await assistant_manager.delete_assistant(
                        assistant_id=ephemeral_lead.id,
                        user_id=user_id,
                        permanent=True,
                    )
                except Exception as cleanup_exc:
                    LOG.warning(
                        "ORCHESTRATOR ▸ Losing supervisor cleanup " "failed: %s",
                        cleanup_exc,
                    )

                ephemeral_lead = await self._native_exec.retrieve_assistant(recorded_id)

                if (
                    not ephemeral_lead
                    or getattr(
                        ephemeral_lead,
                        "deleted_at",
                        None,
                    )
                    is not None
                ):
                    raise RuntimeError("Winning supervisor Assistant is not available.")

            LOG.info(
                "ORCHESTRATOR ▸ Supervisor identity activated: %s "
                "(epoch=%s, role=%s)",
                recorded_id,
                epoch["id"],
                role,
            )

        # Apply the identity only after resolving its durable snapshot.
        self.assistant_id = ephemeral_lead.id
        self.ephemeral_supervisor_id = ephemeral_lead.id

        # Reload the supervisor's own prompt and tool configuration.
        self.assistant_config = {}
        await self._ensure_config_loaded()

        self._delegation_model = get_delegated_model(requested_model=requested_model)

    # ------------------------------------------------------------------
    # Chunk accumulation
    # ------------------------------------------------------------------

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

        return (
            current_block,
            accumulated,
            assistant_reply,
            decision_buffer,
            should_skip,
        )

    def _update_stream_state(
        self,
        chunk: dict,
        state: StreamState,
    ) -> None:
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

    # ------------------------------------------------------------------
    # Assistant configuration loading
    # ------------------------------------------------------------------

    async def _ensure_config_loaded(self):
        if not self.assistant_id:
            LOG.warning("⚠️ Cannot load config: No assistant_id provided.")
            return

        try:
            config_loaded = False

            cached_data = await self.assistant_cache.retrieve(self.assistant_id)

            if cached_data:
                self.assistant_config = cached_data
                config_loaded = True

                LOG.info(
                    f"✅ Config Loaded for {self.assistant_id} | "
                    f"AgentMode: "
                    f"{self.assistant_config.get('agent_mode')}"
                )

            else:
                LOG.warning(
                    f"⚠️ Cache Miss for {self.assistant_id} — " f"fetching from source"
                )

                fresh_config = await self._fetch_assistant_config_from_db(
                    self.assistant_id
                )

                if fresh_config:
                    self.assistant_config = fresh_config
                    config_loaded = True

                    await self.assistant_cache.store(
                        self.assistant_id,
                        fresh_config,
                    )

                    LOG.info(
                        f"✅ Config rehydrated from DB " f"for {self.assistant_id}"
                    )

                else:
                    LOG.error(
                        f"❌ FATAL: No config found anywhere "
                        f"for {self.assistant_id}"
                    )

        except Exception as e:
            LOG.error(f"❌ Error loading assistant config: {e}")

        if config_loaded:
            self.refresh_mcp_runtime_bindings(self.assistant_id)

    def _build_tool_structure(self, batch):
        structure = []

        for tool in batch:
            tool_id = tool.get("id") or (f"call_{uuid.uuid4().hex[:8]}")

            structure.append(
                {
                    "id": tool_id,
                    "type": "function",
                    "function": {
                        "name": tool.get("name"),
                        "arguments": (
                            json.dumps(tool.get("arguments", {}))
                            if isinstance(
                                tool.get("arguments"),
                                dict,
                            )
                            else tool.get("arguments")
                        ),
                    },
                }
            )

        return structure

    # ------------------------------------------------------------------
    # Recursive orchestration loop
    # ------------------------------------------------------------------

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
        - run_id: durable logical execution identity
        - started_at: written once
        - current_turn: durable across continuation boundaries
        - status: canonical transition through RunService
        - completed_at: derived from completed status
        - failed_at: derived from failed status
        - last_error: written before failure transition
        - incomplete_details: written on max-turn failure

        The execution ledger does not replace RunService.
        """

        self._scratch_pad_thread = None
        self._ledger_epoch_id = None
        self._ledger_actor_id = None
        self.ephemeral_supervisor_id = None
        self._active_run_id = run_id
        # Non-secret child-inference reference; do not forward to provider clients.
        provider_key_id = kwargs.pop("provider_key_id", None)

        _original_assistant_id = assistant_id

        await self._ensure_config_loaded()

        tool_validator = ToolValidator()

        # Resume turn accounting from the durable Run.
        try:
            persisted_run = await self._native_exec.retrieve_run(run_id)

            turn_count = int(
                getattr(
                    persisted_run,
                    "current_turn",
                    0,
                )
                or 0
            )

            started_at = getattr(
                persisted_run,
                "started_at",
                None,
            )

        except Exception as lifecycle_exc:
            LOG.warning(
                "ORCHESTRATOR ▸ Failed to hydrate persisted Run "
                "lifecycle (non-fatal): %s",
                lifecycle_exc,
            )

            turn_count = 0
            started_at = None

        current_message_id = message_id

        clean_exit = False
        stream_failed = False

        try:
            while turn_count < max_turns:
                turn_count += 1

                LOG.info(f"ORCHESTRATOR ▸ Turn {turn_count} start " f"[Run: {run_id}]")

                # ------------------------------------------------------
                # Persist durable turn lifecycle
                # ------------------------------------------------------

                try:
                    fields = {"current_turn": turn_count}

                    if started_at is None:
                        started_at = int(time.time())
                        fields["started_at"] = started_at

                    await self._native_exec.update_run_fields(
                        run_id,
                        **fields,
                    )

                    await self._native_exec.update_run_status(
                        run_id,
                        StatusEnum.in_progress.value,
                    )

                except Exception as lifecycle_exc:
                    LOG.warning(
                        "ORCHESTRATOR ▸ Failed to write turn " "state (non-fatal): %s",
                        lifecycle_exc,
                    )

                # ------------------------------------------------------
                # Reset per-turn state
                # ------------------------------------------------------

                self.set_tool_response_state(False)
                self.set_function_call_state(None)
                self._current_tool_call_id = None

                # ------------------------------------------------------
                # Inference turn
                # ------------------------------------------------------

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
                        "ORCHESTRATOR ▸ Turn %s stream failed: %s",
                        turn_count,
                        stream_exc,
                    )

                    try:
                        await self._native_exec.update_run_fields(
                            run_id,
                            last_error=str(stream_exc),
                        )

                        await self._native_exec.update_run_status(
                            run_id,
                            StatusEnum.failed.value,
                        )

                    except Exception as lifecycle_exc:
                        LOG.warning(
                            "ORCHESTRATOR ▸ Failed to write "
                            "stream error (non-fatal): %s",
                            lifecycle_exc,
                        )

                    stream_failed = True

                    yield json.dumps(
                        {
                            "type": "error",
                            "content": (f"Stream failure: {stream_exc}"),
                            "run_id": run_id,
                        }
                    )

                    break

                # ------------------------------------------------------
                # Determine whether the model produced tool calls
                # ------------------------------------------------------

                batch: List[Dict] = self.get_function_call_state()

                if not batch:
                    LOG.info(
                        "ORCHESTRATOR ▸ Turn %s completed with text.",
                        turn_count,
                    )

                    try:
                        await self._native_exec.update_run_status(
                            run_id,
                            StatusEnum.completed.value,
                        )

                    except Exception as lifecycle_exc:
                        LOG.warning(
                            "ORCHESTRATOR ▸ Failed to mark Run "
                            "completed (non-fatal): %s",
                            lifecycle_exc,
                        )

                    clean_exit = True
                    break

                # ------------------------------------------------------
                # Validate tools and detect authorized SDK handoffs
                # ------------------------------------------------------

                has_sdk_user_tool = False

                authorized_consumer_tool_names = (
                    await self._resolve_authorized_consumer_tool_names(assistant_id)
                )

                for call in batch:
                    if "function" in call and isinstance(
                        call["function"],
                        dict,
                    ):
                        tool_name = call["function"].get("name")
                        args = call["function"].get(
                            "arguments",
                            {},
                        )
                    else:
                        tool_name = call.get("name")
                        args = call.get(
                            "arguments",
                            {},
                        )

                    mcp_executor = self._get_mcp_tool_executor(tool_name)

                    if _should_handoff_to_sdk_user_tool(
                        tool_name=tool_name,
                        platform_tools=PLATFORM_TOOLS,
                        mcp_executor=mcp_executor,
                        authorized_consumer_tool_names=(authorized_consumer_tool_names),
                    ):
                        has_sdk_user_tool = True

                    validation_event = tool_validator.validate_args(
                        tool_name=tool_name,
                        args=args,
                    )

                    if validation_event:
                        yield validation_event

                # ------------------------------------------------------
                # Execute the tool batch
                # ------------------------------------------------------

                async for chunk in self.process_tool_calls(
                    thread_id=thread_id,
                    scratch_pad_thread=self._scratch_pad_thread,
                    run_id=run_id,
                    assistant_id=self.assistant_id,
                    tool_call_id=self._current_tool_call_id,
                    model=model,
                    api_key=api_key,
                    provider_key_id=provider_key_id,
                    project_david_api_key=kwargs.get("project_david_api_key"),
                    decision=self._decision_payload,
                ):
                    yield chunk

                # ------------------------------------------------------
                # Decide whether to return control to the SDK
                # ------------------------------------------------------

                if has_sdk_user_tool:
                    LOG.info(
                        "ORCHESTRATOR ▸ SDK tool detected. " "Handing control to SDK."
                    )

                    # Returning from an async generator still runs
                    # finally. Supervisor cleanup below checks the
                    # authoritative Run status before deleting it.
                    return

                LOG.info(
                    "ORCHESTRATOR ▸ Platform batch %s complete. "
                    "Looping internally...",
                    turn_count,
                )

                await asyncio.sleep(0.5)

                current_message_id = None
                continue

            # ----------------------------------------------------------
            # Max-turn failsafe
            # ----------------------------------------------------------

            if turn_count >= max_turns and not clean_exit and not stream_failed:
                LOG.error("ORCHESTRATOR ▸ Max turns reached.")

                try:
                    details = (
                        f"Max turns ({max_turns}) reached without "
                        f"clean completion. Last tool batch size: "
                        f"{len(self.get_function_call_state() or [])}."
                    )

                    await self._native_exec.update_run_fields(
                        run_id,
                        incomplete_details=details,
                        last_error=("Maximum conversation turns exceeded"),
                    )

                    await self._native_exec.update_run_status(
                        run_id,
                        StatusEnum.failed.value,
                    )

                except Exception as lifecycle_exc:
                    LOG.warning(
                        "ORCHESTRATOR ▸ Failed to write max-turns "
                        "state (non-fatal): %s",
                        lifecycle_exc,
                    )

                yield json.dumps(
                    {
                        "type": "error",
                        "content": ("Maximum conversation turns exceeded"),
                        "run_id": run_id,
                    }
                )

        # --------------------------------------------------------------
        # Outer exception guard
        # --------------------------------------------------------------

        except Exception as exc:
            LOG.error(
                "ORCHESTRATOR ▸ Unhandled exception: %s",
                exc,
            )

            try:
                await self._native_exec.update_run_fields(
                    run_id,
                    last_error=str(exc),
                )

                await self._native_exec.update_run_status(
                    run_id,
                    StatusEnum.failed.value,
                )

            except Exception as lifecycle_exc:
                LOG.warning(
                    "ORCHESTRATOR ▸ Failed to persist outer "
                    "exception (non-fatal): %s",
                    lifecycle_exc,
                )

            yield json.dumps(
                {
                    "type": "error",
                    "content": str(exc),
                    "run_id": run_id,
                }
            )

        # --------------------------------------------------------------
        # Supervisor lifecycle finalization
        # --------------------------------------------------------------

        finally:
            # SDK tool handoff is not a terminal Run outcome.
            # The durable actor and Assistant must survive it.

            if (
                self.ephemeral_supervisor_id
                and self._ledger_epoch_id
                and self._ledger_actor_id
            ):
                try:
                    persisted = await self._native_exec.retrieve_run(run_id)

                    raw_status = getattr(
                        persisted,
                        "status",
                        None,
                    )

                    run_status = str(
                        getattr(
                            raw_status,
                            "value",
                            raw_status,
                        )
                    )

                    if run_status in {
                        "completed",
                        "failed",
                        "cancelled",
                        "expired",
                    }:
                        # Record retirement before deleting the
                        # Assistant, preserving forensic identity.
                        #
                        # If ledger finalization fails, do not
                        # delete the Assistant. Leave it available
                        # for explicit recovery.

                        await self._native_exec.retire_execution_actor(
                            actor_id=self._ledger_actor_id
                        )

                        await self._native_exec.complete_execution_epoch(
                            epoch_id=self._ledger_epoch_id,
                            terminal_status=run_status,
                        )

                        await self._ephemeral_clean_up(
                            assistant_id=(self.ephemeral_supervisor_id),
                            thread_id=thread_id,
                            delete_thread=False,
                        )

                    else:
                        LOG.info(
                            "ORCHESTRATOR ▸ Supervisor retained at "
                            "nonterminal Run status %s (epoch=%s)",
                            run_status,
                            self._ledger_epoch_id,
                        )

                except Exception as lifecycle_exc:
                    LOG.warning(
                        "ORCHESTRATOR ▸ Supervisor finalization "
                        "deferred for recovery (Run=%s): %s",
                        run_id,
                        lifecycle_exc,
                    )

            # Only clear request-scoped Python state.
            # This does not delete ledger records or Assistants.

            self.assistant_id = _original_assistant_id
            self.ephemeral_supervisor_id = None
            self._ledger_epoch_id = None
            self._ledger_actor_id = None
            self._active_run_id = None

            LOG.info(
                "ORCHESTRATOR ▸ Identity restored to %s. "
                "Request-scoped supervisor state cleared.",
                _original_assistant_id,
            )
