# src/api/entities_api/orchestration/mixins/native_exec_mixin.py
from __future__ import annotations

from src.api.entities_api.services.native_execution_service import NativeExecutionService


class NativeExecMixin:
    """
    Provides a lazy singleton NativeExecutionService on self._native_exec.

    Owns:
        _native_exec_svc         — backing attribute (private, no name-mangling)
        _native_exec             — property, initialises on first access

    Resolution:
        Single instantiation per object lifetime. NativeExecutionService
        manages its own DB and Redis connections internally — no constructor
        arguments are required from the caller.

    Used by (direct _native_exec calls observed across codebase):
        ContextMixin             — get_raw_messages, hydrate_messages
        ConsumerToolHandlersMixin — submit_tool_output, update_action_status,
                                   create_action, update_run_status
        DelegationMixin          — retrieve_run, create_action,
                                   update_action_status, update_run_fields
        CodeInterpreterMixin     — create_action, submit_tool_output,
                                   update_action_status
        StreamingMixin           — retrieve_run (cancellation monitor)
        OrchestratorCore         — update_run_fields (lifecycle stamps)

    Contract:
        No __init__ participation required. The getattr guard on _native_exec_svc
        means this works correctly even when super().__init__() is not called
        through the full MRO. Do NOT redefine _native_exec on any subclass.
    """

    @property
    def _native_exec(self) -> NativeExecutionService:
        if getattr(self, "_native_exec_svc", None) is None:
            self._native_exec_svc = NativeExecutionService()
        return self._native_exec_svc
