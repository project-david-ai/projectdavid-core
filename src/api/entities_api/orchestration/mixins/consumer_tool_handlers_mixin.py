# src/api/entities_api/orchestration/mixins/consumer_tool_handlers_mixin.py
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, Optional

from dotenv import load_dotenv
from projectdavid_common.validation import StatusEnum

from src.api.entities_api.constants.platform import ERROR_NO_CONTENT
from src.api.entities_api.services.logging_service import LoggingUtility

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

        final_status = StatusEnum.failed if is_error else StatusEnum.completed
        action_id = getattr(action, "id", None)

        try:
            # 1. Save the tool result to the message thread (role='tool')
            # ── REPLACED: was self.project_david_client.messages.submit_tool_output(...)
            await self._native_exec.submit_tool_output(
                thread_id=thread_id,
                assistant_id=assistant_id,
                tool_call_id=tool_call_id,
                content=content,
                action_id=action_id,
                is_error=is_error,
            )

            # 2. Mark the specific Action as finished — only if we have one.
            if action_id:
                # ── REPLACED: was self.project_david_client.actions.update_action(...)
                await self._native_exec.update_action_status(action_id, final_status.value)
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
                await self._native_exec.update_action_status(action_id, StatusEnum.failed.value)
            else:
                LOG.warning(
                    "_submit_fallback_error ▸ action is None for tool_call_id=%s — "
                    "fallback message saved but action status NOT updated.",
                    tool_call_id,
                )

    # ------------------------------------------------------------------
    # ASYNC TOOL CALL PROCESSOR (Reactive Mode)
    # ------------------------------------------------------------------
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
        Reactive Mode: Records the intent as an Action and yields a manifest.
        The SDK catches this manifest and manages the Turn 1 -> Turn 2 recursion.
        """
        tool_name = content.get("name") or content.get("tool_name")
        tool_args = content.get("arguments") or content.get("args") or {}

        # 1. Record the intent to call a tool in the DB
        # ── REPLACED: was self.project_david_client.actions.create_action(...)
        action = await self._native_exec.create_action(
            tool_name=tool_name,
            run_id=run_id,
            tool_call_id=tool_call_id,
            function_args=tool_args,
            decision=decision,
        )

        # 2. Yield the tool_call_manifest to the SDK
        if action and action.id:
            yield json.dumps(
                {
                    "type": "tool_call_manifest",
                    "run_id": run_id,
                    "action_id": action.id,
                    "tool_call_id": tool_call_id,
                    "tool": tool_name,
                    "args": tool_args,
                }
            )

        # 3. Pause the run state. SDK loop will resume by initiating a new turn.
        # ── REPLACED: was self.project_david_client.runs.update_run_status(...)
        await self._native_exec.update_run_status(run_id, StatusEnum.pending_action.value)
        return

    async def finalize_conversation(
        self,
        assistant_reply: str,
        thread_id: str,
        assistant_id: str,
        run_id: str,
        final_status: Optional[str] = None,
    ) -> None:
        """Saves final output and marks run completed (Async)."""
        if not assistant_reply:
            return

        LOG.info(f"TOOL-ROUTER ▸ Finalizing run {run_id}")
        await asyncio.to_thread(
            self._save_assistant_message,
            thread_id,
            assistant_id,
            run_id,
            assistant_reply,
            is_error=False,
            forced_status=final_status,
        )

    async def handle_error(
        self, assistant_reply: str, thread_id: str, assistant_id: str, run_id: str
    ) -> None:
        """Saves terminal error output and marks run failed (Async)."""
        await asyncio.to_thread(
            self._save_assistant_message,
            thread_id,
            assistant_id,
            run_id,
            assistant_reply or "An unexpected terminal error occurred.",
            is_error=True,
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
        """
        Internal sync helper to persist assistant text and update run status.

        Called via asyncio.to_thread so cannot await.  Accesses MessageService
        and RunService through _native_exec's already-instantiated singletons
        to avoid creating fresh service instances on every call.
        """
        # ── REPLACED: was self.project_david_client.messages.save_assistant_message_chunk(...)
        self._native_exec.message_svc.save_assistant_message_chunk(
            thread_id=thread_id,
            content=content,
            role="assistant",
            assistant_id=assistant_id,
            sender_id=assistant_id,
            is_last_chunk=True,
        )

        # ── REPLACED: was self.project_david_client.runs.update_run_status(...)
        status = forced_status or (
            StatusEnum.failed.value if is_error else StatusEnum.completed.value
        )
        self._native_exec.run_svc.update_run_status(run_id, status)
