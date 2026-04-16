# src/api/entities_api/orchestration/mixins/tool_routing_mixin.py

from __future__ import annotations

from typing import AsyncGenerator, Dict, List, Optional, Union

from src.api.entities_api.services.logging_service import LoggingUtility

LOG = LoggingUtility()


class ToolRoutingMixin:
    """
    High-level routing and dispatch of tool call batches.

    Owns:
        _tool_response           — bool flag, True when tool calls are pending
        _function_calls          — List[Dict] of parsed tool calls for current turn
        _tools_called            — List[str] of tool names dispatched this turn
        set/get_tool_response_state()
        set/get_function_call_state()
        reset/get_tools_called()
        parse_and_set_function_calls() — Rust-accelerated fc_parser, primary + loose fallback
        process_tool_calls()     — async generator, batch dispatcher

    Requires on self:
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

    Parser:
        parse_and_set_function_calls() delegates to the fc_parser Rust extension
        (rust/fc_parser). Regex compilation, JSON repair, argument normalisation,
        plan-block stripping, and ID generation all execute in compiled Rust with
        no GIL contention. Build with: cd rust/fc_parser && maturin develop --release
    """

    _tool_response: bool = False
    _function_calls: List[Dict] = []
    _tools_called: List[str] = []

    # ------------------------------------------------------------------
    # State Management
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Parser — Rust-accelerated
    # ------------------------------------------------------------------

    def parse_and_set_function_calls(
        self, accumulated_content: str, assistant_reply: str
    ) -> List[Dict]:
        from src.api.entities_api.orchestration.mixins.json_utils_mixin import (
            JsonUtilsMixin,
        )

        if not isinstance(self, JsonUtilsMixin):
            raise TypeError("ToolRoutingMixin must be mixed with JsonUtilsMixin")

        from fc_parser import parse_function_calls as _rust_parse

        results = _rust_parse(accumulated_content, assistant_reply)

        if results:
            LOG.info("L3-PARSER (Rust) ▸ Detected batch of %d tool(s).", len(results))
            self.set_tool_response_state(True)
            self.set_function_call_state(results)
            return results

        LOG.debug("L3-PARSER (Rust) ✗ nothing found")
        return []

    # ------------------------------------------------------------------
    # Tool Processing & Dispatching (Level 3 Batch Enabled)
    # ------------------------------------------------------------------

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
        LOG.info("TOOL-ROUTER ▸ The Real OwnerID: %s", self._batfish_owner_user_id)

        batch = self.get_function_call_state()
        if not batch:
            return

        LOG.info("TOOL-ROUTER ▸ Dispatching Turn Batch (%s total)", len(batch))

        for fc in batch:
            name = fc.get("name")
            args = fc.get("arguments")
            current_call_id = tool_call_id or fc.get("id")

            if not name and decision:
                name = (
                    decision.get("tool")
                    or decision.get("function")
                    or decision.get("name")
                )
                if name and args is None:
                    args = fc

            if not name:
                LOG.error(
                    "TOOL-ROUTER ▸ Failed to resolve tool name for item in batch."
                )
                continue

            LOG.info("TOOL-ROUTER ▸ scratchpad thread id: %s", scratch_pad_thread)
            LOG.info("TOOL-ROUTER ▶ dispatching: %s (ID: %s)", name, current_call_id)
            self._tools_called.append(name)

            # ----------------------------------------------------------
            # Platform tools — explicit routing
            # ----------------------------------------------------------
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

            # ----------------------------------------------------------
            # Delegation / deep research / memory tools
            # ----------------------------------------------------------
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

            # ----------------------------------------------------------
            # Consumer tools — handover to SDK
            # ----------------------------------------------------------
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
