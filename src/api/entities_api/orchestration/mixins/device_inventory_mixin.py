# src/api/entities_api/orchestration/mixins/device_inventory_mixin.py
from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncGenerator, Dict, Optional

from projectdavid_common import ToolValidator
from projectdavid_common.utilities.logging_service import LoggingUtility
from projectdavid_common.validation import StatusEnum

LOG = LoggingUtility()


def _status(run_id: str, tool: str, message: str, status: str = "running") -> str:
    return json.dumps(
        {
            "type": "engineer_status",
            "run_id": run_id,
            "tool": tool,
            "status": status,
            "message": message,
        }
    )


class NetworkInventoryMixin:
    """
    Drives the **Agentic Network Engineering Tools** (Inventory Search, Device Lookups).
    """

    # ------------------------------------------------------------------
    # 1. HELPER METHODS & LOGIC
    # ------------------------------------------------------------------

    @staticmethod
    def _format_engineer_tool_error(
        tool_name: str, error_content: str, inputs: Dict[str, Any]
    ) -> str:
        if not error_content:
            error_content = "Unknown Error (Empty Response)"

        error_lower = error_content.lower()

        if "validation" in error_lower or "missing" in error_lower:
            return (
                f"❌ SCHEMA ERROR: Invalid arguments for '{tool_name}'.\n"
                f"ERROR DETAILS: {error_content}\n"
                f"PROVIDED ARGS: {json.dumps(inputs)}\n"
                "SYSTEM INSTRUCTION: Check the tool definition and retry with valid arguments."
            )

        if "not found" in error_lower or "empty" in error_lower:
            return (
                f"⚠️ Tool '{tool_name}' returned no results.\n"
                "SYSTEM INSTRUCTION: The requested device or group could not be found in the current inventory map. "
                "Consider asking the user to verify the hostname/group or upload a new inventory."
            )

        return (
            f"❌ Engineering Tool '{tool_name}' Error: {error_content}\n"
            "SYSTEM INSTRUCTION: Review arguments. If error persists, stop using this tool."
        )

    # ------------------------------------------------------------------
    # 2. CORE EXECUTION LOGIC (The Engine)
    # ------------------------------------------------------------------
    async def _execute_engineer_tool_logic(
        self,
        tool_name: str,
        required_schema: Dict[str, Any],
        thread_id: str,
        run_id: str,
        assistant_id: str,
        arguments_dict: Dict[str, Any],
        tool_call_id: Optional[str],
        decision: Optional[Dict],
        user_id: Optional[str] = None,  # ← ADDED
    ) -> AsyncGenerator[str, None]:
        """
        Shared core logic for Inventory Search and Device Lookup.

        user_id: When the platform client uses an admin API key, pass the
                 owning user's ID here so inventory lookups hit the correct
                 bucket rather than the admin's empty one.
        """
        ts_start = asyncio.get_event_loop().time()

        # --- [1] STATUS: VALIDATING ---
        yield _status(run_id, tool_name, "Validating parameters...")

        # --- [2] VALIDATION ---
        validator = ToolValidator()
        validator.schema_registry = {tool_name: required_schema}
        validation_error = validator.validate_args(tool_name, arguments_dict)

        if validation_error:
            LOG.warning(f"{tool_name} ▸ Validation Failed: {validation_error}")
            yield _status(
                run_id,
                tool_name,
                f"Validation failed: {validation_error}",
                status="error",
            )
            error_feedback = self._format_engineer_tool_error(
                tool_name, f"Validation Error: {validation_error}", arguments_dict
            )
            await self.submit_tool_output(
                thread_id=thread_id,
                assistant_id=assistant_id,
                tool_call_id=tool_call_id,
                content=error_feedback,
                action=None,
                is_error=True,
            )
            return

        # --- [3] STATUS: CREATING ACTION ---
        yield _status(run_id, tool_name, "Initializing tool action...")

        action = await asyncio.to_thread(
            self.project_david_client.actions.create_action,
            tool_name=tool_name,
            run_id=run_id,
            tool_call_id=tool_call_id,
            function_args=arguments_dict,
            decision=decision,
        )

        # --- [4] EXECUTION ---
        try:
            res = None

            if tool_name == "search_inventory_by_group":
                group = arguments_dict["group"]
                yield _status(
                    run_id,
                    tool_name,
                    f"Searching inventory map for group: '{group}'...",
                )
                res = await asyncio.to_thread(
                    self.project_david_client.engineer.search_inventory_by_group,
                    group=group,
                    user_id=user_id,  # ← ADDED
                )

            elif tool_name == "get_device_info":
                hostname = arguments_dict["hostname"]
                yield _status(
                    run_id, tool_name, f"Looking up device details for: '{hostname}'..."
                )
                res = await asyncio.to_thread(
                    self.project_david_client.engineer.get_device_info,
                    hostname=hostname,
                    user_id=user_id,  # ← ADDED
                )

            else:
                raise ValueError(f"Unknown engineer tool: {tool_name}")

            # --- [5] RESULT ANALYSIS & RESPONSE ---
            if res:
                final_content = json.dumps(res, indent=2)
                yield _status(
                    run_id, tool_name, "Data retrieved successfully.", status="success"
                )
                is_error = False
            else:
                final_content = self._format_engineer_tool_error(
                    tool_name, "Empty result (not found)", arguments_dict
                )
                yield _status(
                    run_id, tool_name, "Query yielded no results.", status="warning"
                )
                is_error = True

            # --- [6] UPDATE DB & SUBMIT OUTPUT ---
            await asyncio.to_thread(
                self.project_david_client.actions.update_action,
                action_id=action.id,
                status=(
                    StatusEnum.completed.value
                    if not is_error
                    else StatusEnum.failed.value
                ),
            )
            await self.submit_tool_output(
                thread_id=thread_id,
                assistant_id=assistant_id,
                tool_call_id=tool_call_id,
                content=final_content,
                action=action,
                is_error=is_error,
            )

            LOG.info(
                "[%s] %s completed in %.2fs",
                run_id,
                tool_name,
                asyncio.get_event_loop().time() - ts_start,
            )

        except Exception as exc:
            # --- [7] HARD FAILURE ---
            LOG.error(f"[{run_id}] {tool_name} HARD FAILURE: {exc}", exc_info=True)
            yield _status(
                run_id, tool_name, f"Critical failure: {str(exc)}", status="error"
            )
            error_hint = self._format_engineer_tool_error(
                tool_name, str(exc), arguments_dict
            )
            await asyncio.to_thread(
                self.project_david_client.actions.update_action,
                action_id=action.id,
                status=StatusEnum.failed.value,
            )
            await self.submit_tool_output(
                thread_id=thread_id,
                assistant_id=assistant_id,
                tool_call_id=tool_call_id,
                content=error_hint,
                action=action,
                is_error=True,
            )

    # ------------------------------------------------------------------
    # 3. PUBLIC HANDLERS
    # ------------------------------------------------------------------
    async def handle_search_inventory_by_group(
        self,
        thread_id: str,
        run_id: str,
        assistant_id: str,
        arguments_dict: Dict[str, Any],
        tool_call_id: Optional[str] = None,
        decision: Optional[Dict] = None,
        user_id: Optional[str] = None,  # ← ADDED
    ) -> AsyncGenerator[str, None]:
        """Handler for 'search_inventory_by_group'."""
        async for event in self._execute_engineer_tool_logic(
            tool_name="search_inventory_by_group",
            required_schema={"group": str},
            thread_id=thread_id,
            run_id=run_id,
            assistant_id=assistant_id,
            arguments_dict=arguments_dict,
            tool_call_id=tool_call_id,
            decision=decision,
            user_id=user_id,  # ← ADDED
        ):
            yield event

    async def handle_get_device_info(
        self,
        thread_id: str,
        run_id: str,
        assistant_id: str,
        arguments_dict: Dict[str, Any],
        tool_call_id: Optional[str] = None,
        decision: Optional[Dict] = None,
        user_id: Optional[str] = None,  # ← ADDED
    ) -> AsyncGenerator[str, None]:
        """Handler for 'get_device_info'."""
        async for event in self._execute_engineer_tool_logic(
            tool_name="get_device_info",
            required_schema={"hostname": str},
            thread_id=thread_id,
            run_id=run_id,
            assistant_id=assistant_id,
            arguments_dict=arguments_dict,
            tool_call_id=tool_call_id,
            decision=decision,
            user_id=user_id,  # ← ADDED
        ):
            yield event
