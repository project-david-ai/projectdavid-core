from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncGenerator, Dict, Optional

from projectdavid_common import ToolValidator
from projectdavid_common.utilities.logging_service import LoggingUtility
from projectdavid_common.validation import StatusEnum

LOG = LoggingUtility()


def _scratchpad_status(
    run_id: str,
    operation: str,
    state: str,
    tool: Optional[str] = None,
    activity: Optional[str] = None,
    entry: Optional[str] = None,
    assistant_id: Optional[str] = None,
) -> str:
    """
    Emit a status event as raw JSON conforming to the stream EVENT_CONTRACT.

    Shape:
        {
            "type":         "scratchpad_status",
            "run_id":       "<uuid>",
            "operation":    "read" | "update" | "append",
            "state":        "in_progress" | "success" | "completed" | "error",
            "tool":         "<tool_name>",     (optional)
            "activity":     "<human readable>",(optional)
            "entry":        "<entry text>",    (optional)
            "assistant_id": "<asst_id>"        (optional)
        }
    """
    payload = {
        "type": "scratchpad_status",
        "run_id": run_id,
        "operation": operation,
        "state": state,
    }

    if tool is not None:
        payload["tool"] = tool
    if activity is not None:
        payload["activity"] = activity
    if entry is not None:
        payload["entry"] = entry
    if assistant_id is not None:
        payload["assistant_id"] = assistant_id

    return json.dumps(payload)


class ScratchpadMixin:

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scratch_pad_thread: Optional[str] = None

    async def _execute_scratchpad_logic(
        self,
        tool_name: str,
        operation_type: str,
        thread_id: Optional[str],
        run_id: str,
        assistant_id: str,
        arguments_dict: Dict[str, Any],
        tool_call_id: str,
        decision: Any,
        scratch_pad_thread: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:

        if scratch_pad_thread:
            thread_id = scratch_pad_thread

        LOG.info(f"SCRATCHPAD ▸ scratchpad thread id: {scratch_pad_thread}")

        # Injecting assistant_id into the human-readable logs
        # so you can easily track Worker vs Supervisor activity
        _OP_LABELS = {
            "read": (
                f"📖 Reading scratchpad ({assistant_id})...",
                f"📖 Scratchpad read by {assistant_id}.",
            ),
            "update": (
                f"✏️ Updating scratchpad ({assistant_id})...",
                f"✏️ Scratchpad updated by {assistant_id}.",
            ),
            "append": (
                f"📝 Appending to scratchpad ({assistant_id})...",
                f"📝 Scratchpad entry written by {assistant_id}.",
            ),
        }

        label_start, label_done = _OP_LABELS.get(
            operation_type,
            (
                f"Accessing memory ({assistant_id})...",
                f"Memory synchronized by {assistant_id}.",
            ),
        )

        yield _scratchpad_status(
            run_id=run_id,
            operation=operation_type,
            state="in_progress",
            tool=tool_name,
            activity=label_start,
            assistant_id=assistant_id,
        )

        # Validation Logic
        if operation_type != "read":
            required = {"content": str} if operation_type == "update" else {"note": str}

            validator = ToolValidator()
            validator.schema_registry = {tool_name: required}

            if err := validator.validate_args(tool_name, arguments_dict):
                yield _scratchpad_status(
                    run_id=run_id,
                    operation=operation_type,
                    state="error",
                    tool=tool_name,
                    activity=f"Validation error: {err}",
                    assistant_id=assistant_id,
                )

                # Natively register failure action & submit tool output immediately
                await self._native_exec.submit_failed_tool_execution(
                    tool_name=tool_name,
                    run_id=run_id,
                    thread_id=thread_id,
                    assistant_id=assistant_id,
                    tool_call_id=tool_call_id,
                    error_message=f"Error: {err}",
                    function_args=arguments_dict,
                    decision=decision,
                )
                return

        # 1. Create Action natively
        action = await self._native_exec.create_action(
            tool_name=tool_name,
            run_id=run_id,
            tool_call_id=tool_call_id,
            function_args=arguments_dict,
            decision=decision,
        )

        try:
            # 2. Perform Data Plane operations via native ScratchpadService (Direct await, NO to_thread)
            if operation_type == "read":
                res = await self._native_exec.scratchpad_svc.get_formatted_view(
                    thread_id=thread_id,
                )
            elif operation_type == "update":
                res = await self._native_exec.scratchpad_svc.update_content(
                    thread_id=thread_id,
                    content=arguments_dict.get("content"),
                )
            else:
                res = await self._native_exec.scratchpad_svc.append_note(
                    thread_id=thread_id,
                    note=arguments_dict.get("note"),
                )

            # Determine entry text for the frontend scratchpad component
            if operation_type == "append":
                entry_text = arguments_dict.get("note", "")
            elif operation_type == "update":
                entry_text = arguments_dict.get("content", "")
            elif operation_type == "read":
                entry_text = res if isinstance(res, str) else json.dumps(res)
            else:
                entry_text = ""

            # Yield raw scratchpad content event — consumed directly by routes.py
            if entry_text:
                yield _scratchpad_status(
                    run_id=run_id,
                    operation=operation_type,
                    state="success",
                    entry=entry_text,
                    assistant_id=assistant_id,
                )

            # Done-label activity event
            yield _scratchpad_status(
                run_id=run_id,
                operation=operation_type,
                state="completed",
                tool=tool_name,
                activity=label_done,
                assistant_id=assistant_id,
            )

            # 3. Update Action natively
            await self._native_exec.update_action_status(
                action_id=action.id,
                status=StatusEnum.completed.value,
            )

            # 4. Submit Output natively
            content_str = res if isinstance(res, str) else json.dumps(res)
            await self._native_exec.submit_tool_output(
                thread_id=thread_id,
                assistant_id=assistant_id,
                tool_call_id=tool_call_id,
                content=content_str,
                action_id=action.id,
                is_error=False,
            )

        except Exception as e:
            yield _scratchpad_status(
                run_id=run_id,
                operation=operation_type,
                state="error",
                tool=tool_name,
                activity=f"Scratchpad error: {str(e)}",
                assistant_id=assistant_id,
            )

            # Native Error Handling
            if action:
                await self._native_exec.update_action_status(
                    action_id=action.id,
                    status=StatusEnum.failed.value,
                )

            await self._native_exec.submit_tool_output(
                thread_id=thread_id,
                assistant_id=assistant_id,
                tool_call_id=tool_call_id,
                content=f"Error: {e}",
                action_id=action.id if action else None,
                is_error=True,
            )

    async def handle_read_scratchpad(self, *args, **kwargs):
        async for s in self._execute_scratchpad_logic(
            "read_scratchpad", "read", *args, **kwargs
        ):
            yield s

    async def handle_update_scratchpad(self, *args, **kwargs):
        async for s in self._execute_scratchpad_logic(
            "update_scratchpad", "update", *args, **kwargs
        ):
            yield s

    async def handle_append_scratchpad(self, *args, **kwargs):
        async for s in self._execute_scratchpad_logic(
            "append_scratchpad", "append", *args, **kwargs
        ):
            yield s
