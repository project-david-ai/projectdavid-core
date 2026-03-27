# src/api/entities_api/orchestration/mixins/tool_routing_mixin.py

from __future__ import annotations

import json
import re
import uuid
from typing import AsyncGenerator, Dict, List, Optional, Union

from src.api.entities_api.services.logging_service import LoggingUtility

LOG = LoggingUtility()


class ToolRoutingMixin:
    """
    High-level routing and dispatch of tool call batches.
    Level 3 refactor: batch extraction, plan isolation, ID propagation.

    Owns:
        _tool_response           — bool flag, True when tool calls are pending
        _function_calls          — List[Dict] of parsed tool calls for current turn
        _tools_called            — List[str] of tool names dispatched this turn
        FC_REGEX                 — compiled <fc>...</fc> tag parser
        set/get_tool_response_state()
        set/get_function_call_state()
        reset/get_tools_called()
        parse_and_set_function_calls() — L3 parser, primary + loose fallback
        process_tool_calls()     — async generator, batch dispatcher

    Requires on self:
        self.ensure_valid_json()              — JsonUtilsMixin (explicit
                                               isinstance guard enforced)
        self.extract_function_calls_within_body_of_text() — JsonUtilsMixin
        self._batfish_owner_user_id           — set in DelegationMixin.__init__
                                               or QwenBaseWorker.stream()
        self.handle_code_interpreter_action() — CodeInterpreterMixin
        self.handle_shell_action()            — ShellExecutionMixin
        self.handle_file_search()             — FileSearchMixin
        self.handle_read_web_page()
        self.handle_scroll_web_page()
        self.handle_search_web_page()
        self.handle_perform_web_search()      — WebSearchMixin
        self.handle_delegate_research_task()
        self.handle_delegate_engineer_task()  — DelegationMixin
        self.handle_read_scratchpad()
        self.handle_update_scratchpad()
        self.handle_append_scratchpad()       — ScratchpadMixin
        self._handover_to_consumer()          — ConsumerToolHandlersMixin

    Contract:
        parse_and_set_function_calls() raises TypeError if JsonUtilsMixin is
        not in the MRO — this is intentional and is the gold-standard pattern
        for explicit cross-mixin dependency enforcement in this codebase.
        All other handler dependencies are resolved implicitly via _ProviderMixins
        composition. Turn state (_tool_response, _function_calls, _tools_called)
        must be reset at the start of each orchestration turn via
        OrchestratorCore.process_conversation() — do not reset inside this mixin.
    """

    FC_REGEX = re.compile(r"<fc>\s*(?P<payload>\{.*?\})\s*</fc>", re.DOTALL | re.I)

    _tool_response: bool = False
    _function_calls: List[Dict] = []
    _tools_called: List[str] = []

    # -----------------------------------------------------
    # State Management
    # -----------------------------------------------------
    def set_tool_response_state(self, value: bool) -> None:
        LOG.debug("TOOL-ROUTER ▸ set_tool_response_state(%s)", value)
        self._tool_response = value

    def get_tool_response_state(self) -> bool:
        return self._tool_response

    def set_function_call_state(
        self, value: Optional[Union[Dict, List[Dict]]] = None
    ) -> None:
        if value is None:
            self._function_calls = []
        elif isinstance(value, dict):
            self._function_calls = [value]
        else:
            self._function_calls = value

    def get_function_call_state(self) -> List[Dict]:
        return self._function_calls

    def reset_tools_called(self) -> None:
        self._tools_called = []

    def get_tools_called(self) -> List[str]:
        return list(self._tools_called)

    # -----------------------------------------------------
    # Helper Methods
    # -----------------------------------------------------
    def _normalize_arguments(self, payload: Dict) -> Dict:
        args = payload.get("arguments")
        if isinstance(args, str):
            try:
                clean_args = args.strip()
                if clean_args.startswith("```"):
                    clean_args = clean_args.strip("`").replace("json", "").strip()
                payload["arguments"] = json.loads(clean_args)
            except Exception:
                LOG.warning("TOOL-ROUTER ▸ failed to parse string arguments")
        return payload

    # -----------------------------------------------------
    # The Pedantic L3 Parser
    # -----------------------------------------------------
    def parse_and_set_function_calls(
        self, accumulated_content: str, assistant_reply: str
    ) -> List[Dict]:
        from src.api.entities_api.orchestration.mixins.json_utils_mixin import (
            JsonUtilsMixin,
        )

        if not isinstance(self, JsonUtilsMixin):
            raise TypeError("ToolRoutingMixin must be mixed with JsonUtilsMixin")

        body_to_scan = re.sub(
            r"<plan>.*?</plan>", "", accumulated_content, flags=re.DOTALL
        )

        matches = self.FC_REGEX.finditer(body_to_scan)
        results = []

        for m in matches:
            raw_payload = m.group("payload")
            parsed = self.ensure_valid_json(raw_payload)
            if parsed:
                if not parsed.get("id"):
                    parsed["id"] = f"call_{uuid.uuid4().hex[:8]}"
                normalized = self._normalize_arguments(parsed)
                results.append(normalized)

        if results:
            LOG.info(f"L3-PARSER ▸ Detected batch of {len(results)} tool(s).")
            self.set_tool_response_state(True)
            self.set_function_call_state(results)
            return results

        loose = self.extract_function_calls_within_body_of_text(assistant_reply)
        if loose:
            normalized_list = []
            for item in loose:
                if not item.get("id"):
                    item["id"] = f"call_{uuid.uuid4().hex[:8]}"
                normalized_list.append(self._normalize_arguments(item))

            self.set_tool_response_state(True)
            self.set_function_call_state(normalized_list)
            return normalized_list

        LOG.debug("L3-PARSER ✗ nothing found")
        return []

    # -----------------------------------------------------
    # Tool Processing & Dispatching (Level 3 Batch Enabled)
    # -----------------------------------------------------
    async def process_tool_calls(
        self,
        thread_id: str,
        run_id: str,
        assistant_id: str,
        scratch_pad_thread: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        *,
        model: str | None = None,
        api_key: str | None = None,
        decision: Optional[Dict] = None,
    ) -> AsyncGenerator:
        """
        Orchestrates the execution of a detected batch of tool calls.
        Level 3: Iterates through the batch, propagating IDs for history linking.
        """
        LOG.info("TOOL-ROUTER ▸ The scratchpad id: %s", scratch_pad_thread)

        owner_id = self._batfish_owner_user_id
        LOG.info("TOOL-ROUTER ▸ The Real OwnerID: %s", owner_id)

        batch = self.get_function_call_state()
        if not batch:
            return

        LOG.info("TOOL-ROUTER ▸ Dispatching Turn Batch (%s total)", len(batch))

        for fc in batch:
            name = fc.get("name")
            args = fc.get("arguments")

            current_call_id = tool_call_id or fc.get("id")

            if not name and decision:
                inferred_name = (
                    decision.get("tool")
                    or decision.get("function")
                    or decision.get("name")
                )
                if inferred_name:
                    name = inferred_name
                    if args is None:
                        args = fc

            if not name:
                LOG.error(
                    "TOOL-ROUTER ▸ Failed to resolve tool name for item in batch."
                )
                continue

            LOG.info(f"TOOL-ROUTER ▸ scratchpad thread id: {scratch_pad_thread}")
            LOG.info("TOOL-ROUTER ▶ dispatching: %s (ID: %s)", name, current_call_id)
            self._tools_called.append(name)

            # ---------------------------------------------------------
            # 1. PLATFORM TOOLS (Explicit Routing)
            # ---------------------------------------------------------
            if name == "code_interpreter":
                async for chunk in self.handle_code_interpreter_action(
                    thread_id=thread_id,
                    run_id=run_id,
                    assistant_id=assistant_id,
                    arguments_dict=args,
                    tool_call_id=current_call_id,
                    decision=decision,
                ):
                    yield chunk

            elif name == "computer":
                # handle_shell_action is an async generator — iterate, don't await.
                async for chunk in self.handle_shell_action(
                    thread_id=thread_id,
                    run_id=run_id,
                    assistant_id=assistant_id,
                    arguments_dict=args,
                    tool_call_id=current_call_id,
                    decision=decision,
                ):
                    yield chunk

            elif name == "file_search":
                await self.handle_file_search(
                    thread_id=thread_id,
                    run_id=run_id,
                    assistant_id=assistant_id,
                    arguments_dict=args,
                    tool_call_id=current_call_id,
                    decision=decision,
                )

            elif name == "read_web_page":
                async for chunk in self.handle_read_web_page(
                    thread_id=thread_id,
                    run_id=run_id,
                    assistant_id=assistant_id,
                    arguments_dict=args,
                    tool_call_id=current_call_id,
                    decision=decision,
                ):
                    yield chunk

            elif name == "scroll_web_page":
                async for chunk in self.handle_scroll_web_page(
                    thread_id=thread_id,
                    run_id=run_id,
                    assistant_id=assistant_id,
                    arguments_dict=args,
                    tool_call_id=current_call_id,
                    decision=decision,
                ):
                    yield chunk

            elif name == "search_web_page":
                async for chunk in self.handle_search_web_page(
                    thread_id=thread_id,
                    run_id=run_id,
                    assistant_id=assistant_id,
                    arguments_dict=args,
                    tool_call_id=current_call_id,
                    decision=decision,
                ):
                    yield chunk

            elif name == "perform_web_search":
                async for chunk in self.handle_perform_web_search(
                    thread_id=thread_id,
                    run_id=run_id,
                    assistant_id=assistant_id,
                    arguments_dict=args,
                    tool_call_id=current_call_id,
                    decision=decision,
                ):
                    yield chunk

            # ---------------------------------------------------------
            # DELEGATION / DEEP RESEARCH / MEMORY TOOLS
            # ---------------------------------------------------------
            elif name == "delegate_research_task":
                async for chunk in self.handle_delegate_research_task(
                    thread_id=thread_id,
                    run_id=run_id,
                    assistant_id=assistant_id,
                    arguments_dict=args,
                    tool_call_id=current_call_id,
                    decision=decision,
                ):
                    yield chunk

            elif name == "delegate_engineer_task":
                async for chunk in self.handle_delegate_engineer_task(
                    thread_id=thread_id,
                    run_id=run_id,
                    assistant_id=assistant_id,
                    arguments_dict=args,
                    tool_call_id=current_call_id,
                    decision=decision,
                ):
                    yield chunk

            elif name == "read_scratchpad":
                async for chunk in self.handle_read_scratchpad(
                    thread_id=thread_id,
                    scratch_pad_thread=scratch_pad_thread,
                    run_id=run_id,
                    assistant_id=assistant_id,
                    arguments_dict=args,
                    tool_call_id=current_call_id,
                    decision=decision,
                ):
                    yield chunk

            elif name == "update_scratchpad":
                async for chunk in self.handle_update_scratchpad(
                    thread_id=thread_id,
                    run_id=run_id,
                    assistant_id=assistant_id,
                    arguments_dict=args,
                    tool_call_id=current_call_id,
                    decision=decision,
                ):
                    yield chunk

            elif name == "append_scratchpad":
                async for chunk in self.handle_append_scratchpad(
                    thread_id=thread_id,
                    scratch_pad_thread=scratch_pad_thread,
                    run_id=run_id,
                    assistant_id=assistant_id,
                    arguments_dict=args,
                    tool_call_id=current_call_id,
                    decision=decision,
                ):
                    yield chunk

            # ---------------------------------------------------------
            # 3. CONSUMER TOOLS (Handover to SDK)
            # ---------------------------------------------------------
            else:
                async for chunk in self._handover_to_consumer(
                    thread_id=thread_id,
                    assistant_id=assistant_id,
                    content=fc,
                    run_id=run_id,
                    tool_call_id=current_call_id,
                    api_key=api_key,
                    decision=decision,
                ):
                    yield chunk

        LOG.info("TOOL-ROUTER ▸ Batch dispatch Turn complete.")
