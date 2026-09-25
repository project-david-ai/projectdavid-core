"""Minimal internal tool contracts at Project David's execution boundary.

These types describe behavior that Core already supports. They do not add a
transport, registry, namespace, policy model, or rich result representation.
Those concerns belong to later MCP milestones.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ToolDefinition:
    """Provider-independent view of the function schemas Core exposes today."""

    name: str
    description: str
    input_schema: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_function_tool(cls, payload: Mapping[str, Any]) -> ToolDefinition:
        """Map the existing OpenAI-compatible function-tool dictionary."""
        function = payload.get("function")
        if payload.get("type") != "function" or not isinstance(function, Mapping):
            raise ValueError("Expected a function tool definition")

        name = function.get("name")
        description = function.get("description", "")
        input_schema = function.get("parameters", {})
        if not isinstance(name, str) or not name:
            raise ValueError("Function tool definition requires a name")
        if not isinstance(description, str):
            raise TypeError("Function tool description must be a string")
        if not isinstance(input_schema, Mapping):
            raise TypeError("Function tool parameters must be an object")

        return cls(
            name=name,
            description=description,
            input_schema=deepcopy(dict(input_schema)),
        )

    def to_function_tool(self) -> dict[str, Any]:
        """Return the provider-facing dictionary used by current workers."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": deepcopy(dict(self.input_schema)),
            },
        }


@dataclass(frozen=True)
class ToolCallEnvelope:
    """A current flat tool call plus its existing Run/Thread context."""

    name: str
    arguments: Mapping[str, Any]
    run_id: str
    thread_id: str
    assistant_id: str
    tool_call_id: str | None = None

    @classmethod
    def from_legacy_call(
        cls,
        payload: Mapping[str, Any],
        *,
        run_id: str,
        thread_id: str,
        assistant_id: str,
        tool_call_id: str | None = None,
    ) -> ToolCallEnvelope:
        """Map the flat dictionaries consumed by ``ToolRoutingMixin``."""
        name = payload.get("name") or payload.get("tool_name")
        arguments = payload.get("arguments") or payload.get("args") or {}
        if not isinstance(name, str) or not name:
            raise ValueError("Legacy tool call requires a name")
        if not isinstance(arguments, Mapping):
            raise TypeError("Legacy tool call arguments must be an object")

        return cls(
            name=name,
            arguments=deepcopy(dict(arguments)),
            run_id=run_id,
            thread_id=thread_id,
            assistant_id=assistant_id,
            tool_call_id=tool_call_id or payload.get("id"),
        )

    def to_legacy_call(self) -> dict[str, Any]:
        """Return the flat call dictionary expected by current dispatch code."""
        payload: dict[str, Any] = {
            "name": self.name,
            "arguments": deepcopy(dict(self.arguments)),
        }
        if self.tool_call_id is not None:
            payload["id"] = self.tool_call_id
        return payload


@dataclass(frozen=True)
class ToolResultEnvelope:
    """The string result contract Core persists today."""

    content: str
    is_error: bool = False

    @classmethod
    def from_legacy_result(
        cls, content: str, *, is_error: bool = False
    ) -> ToolResultEnvelope:
        if not isinstance(content, str):
            raise TypeError("Legacy tool result content must be a string")
        return cls(content=content, is_error=is_error)

    def to_legacy_result(self) -> dict[str, Any]:
        return {"content": self.content, "is_error": self.is_error}


@runtime_checkable
class ToolExecutor(Protocol):
    """Execution adapter contract; lifecycle ownership remains in Core."""

    async def execute(self, call: ToolCallEnvelope) -> ToolResultEnvelope: ...


__all__ = [
    "ToolCallEnvelope",
    "ToolDefinition",
    "ToolExecutor",
    "ToolResultEnvelope",
]
