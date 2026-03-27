"""
Handlers for *platform-native* tools – i.e. the ones shipped with
Project David itself (web-search, code-interpreter, vector-store search,
remote shell, …).

Asynchronous Version: All blocking I/O is offloaded to threads or awaited.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from projectdavid_common import ValidationInterface

from src.api.entities_api.constants.assistant import (
    WEB_SEARCH_PRESENTATION_FOLLOW_UP_INSTRUCTIONS,
)
from src.api.entities_api.constants.platform import ERROR_NO_CONTENT
from src.api.entities_api.services.logging_service import LoggingUtility

LOG = LoggingUtility()
logger = logging.getLogger(__name__)


class PlatformToolHandlersMixin:
    """
    Handlers for platform-native tools shipped with the substrate
    (web_search, code_interpreter, vector_store_search, computer).

    Owns:
        _submit_platform_tool_output() — thin async wrapper around
                                         ConsumerToolHandlersMixin.submit_tool_output
        _handle_web_search()           — formats search results and appends
                                         follow-up instructions before submission
        _handle_code_interpreter()     — parses result JSON, extracts output text
        _handle_vector_search()        — passes result directly to submission
        _handle_computer()             — passes shell output or ERROR_NO_CONTENT
        _process_platform_tool_calls() — main entry point: creates Action record,
                                         updates run status, calls platform service,
                                         routes to per-tool handler

    Requires on self:
        self.submit_tool_output()      — ConsumerToolHandlersMixin (explicit
                                         isinstance guard enforced in
                                         _submit_platform_tool_output)
        self._native_exec              — NativeExecMixin, used for Action creation
                                         and run status updates in
                                         _process_platform_tool_calls
        self.platform_tool_service     — expected on self, must be set before
                                         _process_platform_tool_calls is called
        self.set_assistant_id()
        self.set_thread_id()           — context setters, ServiceRegistryMixin
                                         or OrchestratorCore

    Contract:
        _submit_platform_tool_output() raises TypeError if
        ConsumerToolHandlersMixin is not in the MRO — explicit guard,
        same pattern as ToolRoutingMixin → JsonUtilsMixin.
        _process_platform_tool_calls() uses asyncio.to_thread for the blocking
        platform service call — do not call it from a sync context.
        All DB operations (Action creation, run status) now use self._native_exec
        consistently with ConsumerToolHandlersMixin and CodeInterpreterMixin.
    """

    async def _submit_platform_tool_output(
        self,
        *,
        thread_id: str,
        assistant_id: str,
        content: str,
        action,
        tool_call_id: Optional[str] = None,
    ):
        """
        Thin wrapper around ConsumerToolHandlersMixin.submit_tool_output.
        Async to match the Consumer refactor.
        """
        from src.api.entities_api.orchestration.mixins.consumer_tool_handlers_mixin import (
            ConsumerToolHandlersMixin,
        )

        if not isinstance(self, ConsumerToolHandlersMixin):
            raise TypeError(
                "PlatformToolHandlersMixin must be combined with ConsumerToolHandlersMixin"
            )

        await self.submit_tool_output(
            thread_id=thread_id,
            assistant_id=assistant_id,
            tool_call_id=tool_call_id,
            content=content,
            action=action,
        )

    async def _handle_web_search(
        self,
        *,
        thread_id: str,
        assistant_id: str,
        tool_call_id: Optional[str] = None,
        output: List[Any],
        action,
    ):
        """Async handler for web_search results."""
        try:
            pretty_content = output[0] if output else "No results found."
            rendered = (
                f"{pretty_content}{WEB_SEARCH_PRESENTATION_FOLLOW_UP_INSTRUCTIONS}"
            )

            await self._submit_platform_tool_output(
                thread_id=thread_id,
                assistant_id=assistant_id,
                tool_call_id=tool_call_id,
                content=rendered,
                action=action,
            )
        except Exception as exc:
            LOG.error("web_search handler failed: %s", exc, exc_info=True)
            await self._submit_platform_tool_output(
                thread_id=thread_id,
                assistant_id=assistant_id,
                tool_call_id=tool_call_id,
                content=f"ERROR: {exc}",
                action=action,
            )

    async def _handle_code_interpreter(
        self,
        *,
        thread_id: str,
        assistant_id: str,
        tool_call_id: Optional[str] = None,
        output: str,
        action,
    ):
        """Async handler for code_interpreter output."""
        try:
            parsed = json.loads(output)
            output_text = parsed.get("result", {}).get("output", str(output))
        except Exception as exc:
            LOG.error("code_interpreter output malformed: %s", exc, exc_info=True)
            output_text = f"ERROR: {exc}"

        await self._submit_platform_tool_output(
            thread_id=thread_id,
            assistant_id=assistant_id,
            tool_call_id=tool_call_id,
            content=output_text,
            action=action,
        )

    async def _handle_vector_search(
        self,
        *,
        thread_id: str,
        assistant_id: str,
        tool_call_id: Optional[str] = None,
        output: Any,
        action,
    ):
        """Async handler for vector store results."""
        await self._submit_platform_tool_output(
            thread_id=thread_id,
            assistant_id=assistant_id,
            tool_call_id=tool_call_id,
            content=str(output),
            action=action,
        )

    async def _handle_computer(
        self,
        *,
        thread_id: str,
        assistant_id: str,
        tool_call_id: Optional[str] = None,
        output: str,
        action,
    ):
        """Async handler for remote shell/computer output."""
        await self._submit_platform_tool_output(
            thread_id=thread_id,
            assistant_id=assistant_id,
            tool_call_id=tool_call_id,
            content=output or ERROR_NO_CONTENT,
            action=action,
        )

    async def _process_platform_tool_calls(
        self,
        thread_id: str,
        assistant_id: str,
        *,
        tool_call_id: Optional[str] = None,
        content: Dict[str, Any],
        run_id: str,
        decision: Optional[Dict] = None,
    ):
        """
        Main entry point for platform tools.
        Blocking platform service call is offloaded to a thread.
        All DB operations use self._native_exec.
        """
        self.set_assistant_id(assistant_id)
        self.set_thread_id(thread_id)

        tool_name = content.get("name")
        arguments = content.get("arguments", {})

        # 1. Create Action record via _native_exec
        try:
            action = await self._native_exec.create_action(
                tool_name=tool_name,
                run_id=run_id,
                tool_call_id=tool_call_id,
                function_args=arguments,
                decision=decision,
            )
        except Exception as e:
            LOG.error(f"PLATFORM-HANDLER ▸ Action creation failed: {e}")
            return

        LOG.debug(
            "Action %s created for %s", getattr(action, "id", "unknown"), tool_name
        )

        # 2. Update Run Status to pending_action via _native_exec
        try:
            await self._native_exec.update_run_status(
                run_id,
                ValidationInterface.StatusEnum.pending_action.value,
            )
        except Exception as e:
            LOG.warning(f"PLATFORM-HANDLER ▸ Run status update failed: {e}")

        # 3. Call the Platform Service — blocking, offloaded to thread
        platform = self.platform_tool_service
        try:
            result = await asyncio.to_thread(
                platform.call_function, tool_name, arguments
            )
        except Exception as e:
            LOG.error(f"PLATFORM-HANDLER ▸ Platform service call failed: {e}")
            result = f"CRITICAL ERROR: Platform service failed to execute {tool_name}"

        # 4. Route to per-tool handler
        handlers = {
            "web_search": self._handle_web_search,
            "code_interpreter": self._handle_code_interpreter,
            "vector_store_search": self._handle_vector_search,
            "computer": self._handle_computer,
        }

        handler = handlers.get(tool_name)
        if handler:
            await handler(
                thread_id=thread_id,
                assistant_id=assistant_id,
                tool_call_id=tool_call_id,
                output=result,
                action=action,
            )
        else:
            await self._submit_platform_tool_output(
                thread_id=thread_id,
                assistant_id=assistant_id,
                tool_call_id=tool_call_id,
                content=str(result),
                action=action,
            )

        LOG.debug("Platform-tool %s finished for run %s", tool_name, run_id)
