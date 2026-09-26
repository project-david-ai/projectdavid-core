"""Helpers for keeping model-facing tools and MCP provenance coherent."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def function_tool_name(tool: object) -> str | None:
    """Return an OpenAI-style function tool name, if present."""

    if not isinstance(tool, dict):
        return None

    if tool.get("type") != "function":
        return None

    function = tool.get("function")
    if not isinstance(function, dict):
        return None

    name = function.get("name")

    if isinstance(name, str) and name:
        return name

    return None


def merge_user_tools_preserving_managed(
    current_tools: list[Any] | None,
    incoming_tools: list[Any] | None,
    managed_names: set[str],
) -> list[Any]:
    """Replace ordinary tools while preserving MCP-managed function specs.

    A generic Assistant update may round-trip an unchanged MCP tool, but it
    may not alter or remove MCP-managed capabilities. Detachment belongs to
    the MCP management API so provenance and model-facing capabilities stay
    synchronized.
    """

    current = deepcopy(list(current_tools or []))
    incoming = deepcopy(list(incoming_tools or []))

    managed_by_name: dict[str, Any] = {}
    managed_order: list[str] = []

    for tool in current:
        name = function_tool_name(tool)

        if name not in managed_names:
            continue

        if name in managed_by_name:
            raise ValueError(f"duplicate MCP-managed tool configuration: {name}")

        managed_by_name[name] = tool
        managed_order.append(name)

    missing = managed_names - set(managed_by_name)

    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(
            f"MCP provenance is missing model-facing tool configuration: {names}"
        )

    ordinary: list[Any] = []

    for tool in incoming:
        name = function_tool_name(tool)

        if name not in managed_names:
            ordinary.append(tool)
            continue

        if managed_by_name[name] != tool:
            raise ValueError(
                f"MCP-managed tool '{name}' cannot be modified through "
                "the generic assistant tools field"
            )

        # Exact round-trip of the managed spec is accepted, but the canonical
        # stored copy below remains authoritative.

    ordinary.extend(deepcopy(managed_by_name[name]) for name in managed_order)

    return ordinary


def upsert_function_tool(
    tools: list[Any] | None,
    function_tool: dict[str, Any],
) -> list[Any]:
    """Insert or replace one model-facing function definition by name."""

    name = function_tool_name(function_tool)

    if name is None:
        raise ValueError("managed MCP tool must be a function tool")

    updated = deepcopy(list(tools or []))
    replacement = deepcopy(function_tool)

    indexes = [
        index for index, tool in enumerate(updated) if function_tool_name(tool) == name
    ]

    if len(indexes) > 1:
        raise ValueError(f"duplicate model-facing function tool configuration: {name}")

    if indexes:
        updated[indexes[0]] = replacement
    else:
        updated.append(replacement)

    return updated


def remove_function_tools(
    tools: list[Any] | None,
    names: set[str],
) -> list[Any]:
    """Remove model-facing function definitions matching *names*."""

    return [
        deepcopy(tool)
        for tool in list(tools or [])
        if function_tool_name(tool) not in names
    ]


__all__ = [
    "function_tool_name",
    "merge_user_tools_preserving_managed",
    "remove_function_tools",
    "upsert_function_tool",
]
