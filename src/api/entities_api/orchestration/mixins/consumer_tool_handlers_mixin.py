# src/api/entities_api/orchestration/mixins/consumer_tool_handlers_mixin.py
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, Optional

from dotenv import load_dotenv
from projectdavid_common.validation import StatusEnum

from src.api.entities_api.constants.platform import ERROR_NO_CONTENT
from src.api.entities_api.orchestration.mcp_run_session_manager import (
    McpRunSessionManager,
    get_mcp_run_session_manager,
)
from src.api.entities_api.orchestration.mcp_tool_executor import McpToolExecutor
from src.api.entities_api.orchestration.tool_abi import (
    ToolCallEnvelope,
    ToolResultEnvelope,
)
from src.api.entities_api.services.logging_service import LoggingUtility
from src.api.entities_api.services.mcp_registration_service import (
    McpRegistrationService,
)

load_dotenv()
LOG = LoggingUtility()
logger = logging.getLogger(__name__)


class ConsumerToolHandlersMixin:
    """
    Server-side tool output persistence and run lifecycle management.
    Level 2 refactor: all SDK client calls replaced with self._native_exec.

    Owns:
        submit_tool_output()      — async, persists tool result to thread and
                                    updates Action status
        _submit_fallback_error()  — async, last-resort error persistence
        _handover_to_consumer()   — async generator, records Action and yields
                                    tool_call_manifest for SDK-managed tools
        finalize_conversation()   — async, saves assistant reply and marks run
                                    complete via asyncio.to_thread
        handle_error()            — async, saves terminal error and marks run failed
        _save_assistant_message() — sync internal helper called via to_thread,
                                    uses _native_exec.message_svc and run_svc
                                    directly to avoid async context issues

    Requires on self:
        self._native_exec         — NativeExecMixin, used throughout for all
                                    DB operations (submit_tool_output,
                                    update_action_status, create_action,
                                    update_run_status, message_svc, run_svc)

    Called by:
        PlatformToolHandlersMixin — via _submit_platform_tool_output()
                                    (explicit isinstance guard enforced)
        ToolRoutingMixin          — via _handover_to_consumer()
        DelegationMixin           — via submit_tool_output() directly
        CodeInterpreterMixin      — finalisation path

    Contract:
        submit_tool_output() tolerates action=None — tool output is still
        persisted to the thread but action status update is skipped with a
        warning. This handles the case where create_action() failed upstream
        (e.g. during delegation). Do NOT redefine submit_tool_output() or
        finalize_conversation() on any subclass.
    """

    async def submit_tool_output(
        self,
        *,
        thread_id: str,
        assistant_id: str,
        tool_call_id: Optional[str] = None,
        content: str,
        action: Any,
        is_error: bool = False,
    ) -> None:
        """
        Push tool output to thread and update Action status (Async).
        Level 2: Agnostic to content; assumes the SDK provides either success JSON
        or a formatted 'Level 2' error instruction for the LLM.

        action may be None when the upstream create_action call failed (e.g.
        during delegation). In that case we still persist the tool output to
        the thread so the LLM can continue, but we skip the action status
        update rather than crashing with 'NoneType has no attribute id'.
        """
        if not content:
            content = ERROR_NO_CONTENT

        result = ToolResultEnvelope.from_legacy_result(content, is_error=is_error)
        final_status = StatusEnum.failed if result.is_error else StatusEnum.completed
        action_id = getattr(action, "id", None)

        try:
            # 1. Save the tool result to the message thread (role='tool')
            # ── REPLACED: was self.project_david_client.messages.submit_tool_output(...)
            await self._native_exec.submit_tool_output(
                thread_id=thread_id,
                assistant_id=assistant_id,
                tool_call_id=tool_call_id,
                content=result.content,
                action_id=action_id,
                is_error=result.is_error,
            )

            # 2. Mark the specific Action as finished — only if we have one.
            if action_id:
                # ── REPLACED: was self.project_david_client.actions.update_action(...)
                await self._native_exec.update_action_status(
                    action_id, final_status.value
                )
            else:
                LOG.warning(
                    "submit_tool_output ▸ action is None for tool_call_id=%s — "
                    "tool output saved but action status NOT updated.",
                    tool_call_id,
                )

        except Exception as exc:
            LOG.error("submit_tool_output failed: %s", exc, exc_info=True)
            await self._submit_fallback_error(
                thread_id,
                assistant_id,
                error_msg=f"CRITICAL_SYSTEM_ERROR: {exc}",
                action=action,
            )

    async def submit_tool_result(
        self,
        *,
        thread_id: str,
        assistant_id: str,
        result: ToolResultEnvelope,
        action: Any,
        tool_call_id: Optional[str] = None,
    ) -> None:
        """Submit a typed result through Core's current legacy persistence path.

        Rich fields remain available on ToolResultEnvelope at the orchestration
        boundary. Durable storage is still string-based, so only the legacy
        projection is persisted until a future storage schema explicitly grows
        rich tool-result columns.
        """

        await self.submit_tool_output(
            thread_id=thread_id,
            assistant_id=assistant_id,
            tool_call_id=tool_call_id,
            content=result.content,
            action=action,
            is_error=result.is_error,
        )

    async def _submit_fallback_error(
        self,
        thread_id: str,
        assistant_id: str,
        *,
        tool_call_id: Optional[str] = None,
        error_msg: str,
        action: Any,
    ) -> None:
        """Fallback for critical database or infrastructure failures."""
        action_id = getattr(action, "id", None)

        try:
            # ── REPLACED: was self.project_david_client.messages.submit_tool_output(...)
            await self._native_exec.submit_tool_output(
                thread_id=thread_id,
                assistant_id=assistant_id,
                tool_call_id=tool_call_id,
                content=error_msg,
                action_id=action_id,
                is_error=True,
            )
        finally:
            if action_id:
                # ── REPLACED: was self.project_david_client.actions.update_action(...)
                await self._native_exec.update_action_status(
                    action_id, StatusEnum.failed.value
                )
            else:
                LOG.warning(
                    "_submit_fallback_error ▸ action is None for tool_call_id=%s — "
                    "fallback message saved but action status NOT updated.",
                    tool_call_id,
                )

    # ------------------------------------------------------------------
    # ASYNC TOOL CALL PROCESSOR (Reactive Mode)
    # ------------------------------------------------------------------
    _MCP_RESERVED_ROUTED_NAMES: frozenset[str] = frozenset(
        {
            "code_interpreter",
            "computer",
            "file_search",
            "read_web_page",
            "scroll_web_page",
            "search_web_page",
            "perform_web_search",
            "delegate_research_task",
            "delegate_engineer_task",
            "read_scratchpad",
            "update_scratchpad",
            "append_scratchpad",
        }
    )

    def _mcp_executor_registry(self) -> dict[str, McpToolExecutor]:
        """Return ephemeral MCP bindings owned by this orchestrator instance."""

        registry = self.__dict__.get("_mcp_tool_executors")
        if registry is None:
            registry = {}
            self.__dict__["_mcp_tool_executors"] = registry
        return registry

    def _mcp_run_session_manager(self) -> McpRunSessionManager:
        """Return the application-runtime MCP session owner.

        Provider/orchestrator instances may be recreated across inference
        continuation requests for the same Project David Run. The asyncio
        application loop survives those request boundaries.
        """

        return get_mcp_run_session_manager()

    async def _close_mcp_run_sessions(
        self,
        run_id: str,
        *,
        cancel: bool = False,
    ) -> None:
        """Release all live MCP protocol sessions owned by one Run."""

        await get_mcp_run_session_manager().close_run(
            run_id,
            cancel=cancel,
        )

    def bind_mcp_tool_executor(self, executor: McpToolExecutor) -> None:
        """Bind one provider-visible MCP alias for this orchestrator instance."""

        provider_name = executor.provider_name

        if provider_name in self._MCP_RESERVED_ROUTED_NAMES:
            raise ValueError(
                f"MCP provider alias collides with an internal tool: {provider_name}"
            )

        registry = self._mcp_executor_registry()
        existing = registry.get(provider_name)

        if existing is not None and existing is not executor:
            raise ValueError(f"MCP provider alias is already bound: {provider_name}")

        registry[provider_name] = executor

    def unbind_mcp_tool_executor(self, provider_name: str) -> bool:
        """Remove one ephemeral MCP binding."""

        return self._mcp_executor_registry().pop(provider_name, None) is not None

    def _get_mcp_tool_executor(
        self,
        provider_name: str,
    ) -> McpToolExecutor | None:
        return self._mcp_executor_registry().get(provider_name)

    def refresh_mcp_runtime_bindings(
        self,
        assistant_id: str,
    ) -> None:
        """Rebuild this orchestrator's MCP executor registry."""

        executors = McpRegistrationService().build_runtime_executors(
            assistant_id=assistant_id,
        )

        registry = self._mcp_executor_registry()
        registry.clear()

        for executor in executors:
            self.bind_mcp_tool_executor(executor)

        LOG.info(
            "MCP-RUNTIME hydration complete: assistant=%s bindings=%d",
            assistant_id,
            len(executors),
        )

    _MCP_CANCEL_POLL_SECONDS = 0.5

    async def _execute_mcp_tool_with_run_cancellation(
        self,
        *,
        executor: McpToolExecutor,
        call: ToolCallEnvelope,
        action: Any,
    ) -> ToolResultEnvelope | None:
        """Run MCP tools while Project David remains lifecycle authority.

        The remote request executes in its own asyncio Task. While it is alive,
        Core polls the authoritative Run record. API cancellation therefore
        interrupts a blocked/long-running tools/call rather than waiting for the
        remote server to finish.

        ``None`` means the Project David Run was cancelled and no synthetic tool
        result should be persisted into the conversation.
        """

        action_id = getattr(action, "id", None)

        async def progress_callback(
            progress: float,
            total: float | None,
            message: str | None,
        ) -> None:
            LOG.debug(
                "MCP progress ▸ run=%s action=%s tool=%s progress=%s total=%s message=%s",
                call.run_id,
                action_id,
                call.name,
                progress,
                total,
                message,
            )

        # Stateful MCP servers require protocol-session affinity across
        # multiple native tool calls in the same Project David Run.
        #
        # Scope by authenticated tenant, Run and MCP registration. If an
        # internal/legacy execution path lacks an owner identity, fall back to
        # a run-unique scope rather than ever sharing state across Runs.
        owner_scope = str(
            getattr(self, "_batfish_owner_user_id", None) or f"run:{call.run_id}"
        )

        task = asyncio.create_task(
            self._mcp_run_session_manager().execute(
                owner_scope=owner_scope,
                executor=executor,
                call=call,
                progress_callback=progress_callback,
            )
        )

        try:
            while True:
                done, _ = await asyncio.wait(
                    {task},
                    timeout=self._MCP_CANCEL_POLL_SECONDS,
                )

                if task in done:
                    result = await task

                    # Completion and cancellation may race. Re-read the
                    # authoritative Project David Run before accepting the
                    # remote result. If cancellation was already committed,
                    # cancellation wins and no synthetic tool result is saved.
                    try:
                        run = await self._native_exec.retrieve_run(call.run_id)
                    except Exception as exc:  # noqa: BLE001
                        LOG.warning(
                            "MCP completion-state check failed for run %s: %s",
                            call.run_id,
                            exc,
                        )
                    else:
                        if getattr(run, "status", None) == StatusEnum.cancelled.value:
                            LOG.info(
                                "Discarding completed MCP result because "
                                "Project David run %s is already cancelled "
                                "(action=%s, tool=%s)",
                                call.run_id,
                                action_id,
                                call.name,
                            )

                            if action_id:
                                await self._native_exec.update_action_status(
                                    action_id,
                                    StatusEnum.cancelled.value,
                                )

                            await self._close_mcp_run_sessions(
                                call.run_id,
                                cancel=True,
                            )
                            return None

                    return result

                try:
                    run = await self._native_exec.retrieve_run(call.run_id)
                except Exception as exc:  # noqa: BLE001
                    LOG.warning(
                        "MCP cancellation poll failed for run %s: %s",
                        call.run_id,
                        exc,
                    )
                    continue

                if getattr(run, "status", None) != StatusEnum.cancelled.value:
                    continue

                LOG.info(
                    "MCP execution cancelled by Project David run %s "
                    "(action=%s, tool=%s)",
                    call.run_id,
                    action_id,
                    call.name,
                )

                task.cancel()

                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as exc:  # noqa: BLE001
                    LOG.debug(
                        "MCP cancellation cleanup raised for run %s: %s",
                        call.run_id,
                        exc,
                    )

                await self._close_mcp_run_sessions(
                    call.run_id,
                    cancel=True,
                )

                if action_id:
                    await self._native_exec.update_action_status(
                        action_id,
                        StatusEnum.cancelled.value,
                    )

                return None

        except asyncio.CancelledError:
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # noqa: BLE001
                LOG.debug(
                    "MCP task cleanup raised during parent cancellation "
                    "for run %s: %s",
                    call.run_id,
                    exc,
                )

            await self._close_mcp_run_sessions(
                call.run_id,
                cancel=True,
            )
            raise

    async def _resolve_authorized_consumer_tool_names(
        self,
        assistant_id: str,
    ) -> frozenset[str]:
        """
        Return SDK/MCP function names authorized for this assistant.

        Consumer functions and attached MCP aliases are both represented as
        ``type=function`` entries in the assistant tools configuration.

        Failure to load that authoritative configuration fails closed.
        """
        try:
            cache = self.get_assistant_cache()

            cfg = await cache.retrieve(assistant_id)

        except Exception as exc:
            LOG.warning(
                "CONSUMER-TOOLS ? Failed to load assistant %s "
                "tool configuration for authority validation: %s. "
                "Failing closed.",
                assistant_id,
                exc,
            )

            return frozenset()

        raw_tools = (
            cfg.get("tools")
            if isinstance(
                cfg,
                dict,
            )
            else None
        ) or []

        allowed: set[str] = set()

        for tool in raw_tools:
            if not isinstance(
                tool,
                dict,
            ):
                continue

            if tool.get("type") != "function":
                continue

            function = tool.get("function")

            if not isinstance(
                function,
                dict,
            ):
                continue

            name = function.get("name")

            if (
                isinstance(
                    name,
                    str,
                )
                and name
            ):
                allowed.add(name)

        return frozenset(allowed)

    async def _handover_to_consumer(
        self,
        thread_id: str,
        assistant_id: str,
        content: Dict[str, Any],
        run_id: str,
        *,
        tool_call_id: Optional[str] = None,
        decision: Optional[Dict] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Record the tool intent and either execute a bound MCP tool inside Core
        or preserve the existing SDK-managed consumer-tool handoff.
        """
        call = ToolCallEnvelope.from_legacy_call(
            content,
            run_id=run_id,
            thread_id=thread_id,
            assistant_id=assistant_id,
            tool_call_id=tool_call_id,
        )

        authorized_names = await self._resolve_authorized_consumer_tool_names(
            assistant_id
        )

        if call.name not in authorized_names:
            LOG.warning(
                "CONSUMER-TOOLS ? Rejecting unauthorized consumer/MCP "
                "tool call: %s (assistant=%s)",
                call.name,
                assistant_id,
            )

            try:
                await self._native_exec.submit_tool_output(
                    thread_id=call.thread_id,
                    assistant_id=call.assistant_id,
                    tool_call_id=call.tool_call_id,
                    content=(
                        f"Error: the tool '{call.name}' is not available "
                        "to this assistant. It was not declared in the "
                        "assistant's tools configuration. Do not attempt "
                        "to call it again."
                    ),
                    is_error=True,
                )

            except Exception as exc:
                LOG.error(
                    "CONSUMER-TOOLS ? Failed to persist unauthorized "
                    "tool rejection for %s: %s",
                    call.name,
                    exc,
                )

            return

        action = await self._native_exec.create_action(
            tool_name=call.name,
            run_id=call.run_id,
            tool_call_id=call.tool_call_id,
            function_args=call.arguments,
            decision=decision,
        )

        mcp_executor = self._get_mcp_tool_executor(call.name)

        if mcp_executor is not None:
            await self._native_exec.update_run_status(
                call.run_id,
                StatusEnum.pending_action.value,
            )

            result = await self._execute_mcp_tool_with_run_cancellation(
                executor=mcp_executor,
                call=call,
                action=action,
            )

            if result is None:
                return

            await self.submit_tool_result(
                thread_id=call.thread_id,
                assistant_id=call.assistant_id,
                tool_call_id=call.tool_call_id,
                result=result,
                action=action,
            )
            return

        # Existing SDK-managed consumer-tool behavior.
        if action and action.id:
            yield json.dumps(
                {
                    "type": "tool_call_manifest",
                    "run_id": call.run_id,
                    "action_id": action.id,
                    "tool_call_id": call.tool_call_id,
                    "tool": call.name,
                    "args": call.arguments,
                }
            )

        await self._native_exec.update_run_status(
            call.run_id,
            StatusEnum.pending_action.value,
        )
        return

    async def finalize_conversation(
        self,
        assistant_reply: str,
        thread_id: str,
        assistant_id: str,
        run_id: str,
        final_status: Optional[str] = None,
    ) -> None:
        """Persist assistant output without implicitly completing the Run.

        Assistant/tool-call transcript persistence and Run lifecycle are
        separate concerns. Provider/orchestration code owns the execution
        status transition. ``final_status`` remains as an explicit
        compatibility escape hatch for callers that intentionally want
        this operation to perform a transition.
        """
        if not assistant_reply:
            return

        LOG.info("TOOL-ROUTER Persisting assistant output for run %s", run_id)

        await asyncio.to_thread(
            self._save_assistant_message,
            thread_id,
            assistant_id,
            run_id,
            assistant_reply,
            is_error=False,
        )

        if final_status is not None:
            await self._native_exec.update_run_status(
                run_id,
                final_status,
            )

    async def handle_error(
        self,
        assistant_reply: str,
        thread_id: str,
        assistant_id: str,
        run_id: str,
    ) -> None:
        """Persist terminal error output, fail the epoch, release MCP state."""

        try:
            await asyncio.to_thread(
                self._save_assistant_message,
                thread_id,
                assistant_id,
                run_id,
                assistant_reply or "An unexpected terminal error occurred.",
                is_error=True,
            )

            await self._native_exec.update_run_status(
                run_id,
                StatusEnum.failed.value,
            )
        finally:
            await self._close_mcp_run_sessions(
                run_id,
                cancel=True,
            )

    def _save_assistant_message(
        self,
        thread_id: str,
        assistant_id: str,
        run_id: str,
        content: str = "",
        *,
        is_error: bool,
        forced_status: Optional[str] = None,
    ) -> None:
        self._native_exec.message_svc.save_assistant_message_chunk(
            thread_id=thread_id,
            content=content,
            role="assistant",
            assistant_id=assistant_id,
            sender_id=assistant_id,
            is_last_chunk=True,
        )

        # Push to Redis cache so the next turn's context build sees this message.
        # Without this, Redis returns stale history and Turn 2 misses the <fc>
        # assistant row, producing a dangling tool_call_id on the tool message.
        try:
            from src.api.entities_api.cache.message_cache import get_sync_message_cache

            cache = get_sync_message_cache()
            cache.append_message_sync(
                thread_id,
                {"role": "assistant", "content": content},
            )
        except Exception as exc:
            LOG.warning("Failed to append assistant message to cache: %s", exc)

        # Deliberately lifecycle-neutral.
        #
        # This method also persists assistant tool-call envelopes. Saving
        # one of those rows must never imply that the logical Run has
        # reached a terminal state.
