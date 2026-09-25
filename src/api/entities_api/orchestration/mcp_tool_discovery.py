"""Adapt MCP tool discovery into Project David's internal Tool ABI.

This module owns MCP discovery identity and provider-safe naming only.
It does not execute tools, mutate orchestration state, persist server
registrations, or apply assistant/run policy.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from urllib.parse import quote

from mcp.types import ListToolsResult, Tool

from .tool_abi import ToolDefinition

_PROVIDER_NAME_MAX_LENGTH = 64
_PROVIDER_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9_-]+")


@dataclass(frozen=True)
class McpDiscoveredTool:
    """One remote MCP tool mapped into Project David's provider-facing ABI."""

    server_id: str
    remote_name: str
    canonical_id: str
    provider_name: str
    definition: ToolDefinition


@dataclass(frozen=True)
class McpToolDiscoveryPage:
    """One adapted page from an MCP ``tools/list`` response."""

    tools: tuple[McpDiscoveredTool, ...]
    next_cursor: str | None = None


@dataclass(frozen=True)
class _PendingTool:
    server_id: str
    remote_name: str
    canonical_id: str
    provider_base: str
    description: str
    input_schema: dict[str, object]


def _validate_server_id(server_id: str) -> str:
    if not isinstance(server_id, str) or not server_id.strip():
        raise ValueError("MCP server_id must be a non-empty string")
    return server_id


def _canonical_component(value: str) -> str:
    """Escape separators while leaving ordinary identifier characters readable."""

    return quote(value, safe="-._~")


def _canonical_id(server_id: str, remote_name: str) -> str:
    return (
        f"mcp:{_canonical_component(server_id)}:" f"{_canonical_component(remote_name)}"
    )


def _provider_component(value: str, *, fallback: str) -> str:
    component = _PROVIDER_UNSAFE_CHARS.sub("_", value).strip("_")
    return component or fallback


def _provider_base(server_id: str, remote_name: str) -> str:
    server = _provider_component(server_id, fallback="mcp")
    tool = _provider_component(remote_name, fallback="tool")
    return f"{server}__{tool}"


def _hashed_provider_name(
    provider_base: str,
    canonical_id: str,
    *,
    digest_length: int = 10,
) -> str:
    digest = hashlib.sha256(canonical_id.encode("utf-8")).hexdigest()[:digest_length]
    suffix = f"__{digest}"

    prefix_length = _PROVIDER_NAME_MAX_LENGTH - len(suffix)
    prefix = provider_base[:prefix_length].rstrip("_-")

    if not prefix:
        prefix = "mcp"

    return f"{prefix}{suffix}"


def _pending_tool(server_id: str, tool: Tool) -> _PendingTool:
    remote_name = tool.name
    if not isinstance(remote_name, str) or not remote_name:
        raise ValueError("MCP tool name must be a non-empty string")

    description = tool.description or tool.title or ""

    return _PendingTool(
        server_id=server_id,
        remote_name=remote_name,
        canonical_id=_canonical_id(server_id, remote_name),
        provider_base=_provider_base(server_id, remote_name),
        description=description,
        input_schema=deepcopy(tool.input_schema),
    )


def adapt_mcp_tools(
    server_id: str,
    tools: Iterable[Tool],
    *,
    reserved_provider_names: Iterable[str] = (),
) -> tuple[McpDiscoveredTool, ...]:
    """Map MCP tools into stable identities and provider-safe definitions.

    Provider aliases retain the readable ``server__tool`` shape whenever it is
    safe and unambiguous. A deterministic canonical-id hash is appended when
    normalization would collide, when an alias is already reserved, or when
    the readable alias exceeds the provider-facing 64-character limit.
    """

    server_id = _validate_server_id(server_id)
    pending = tuple(_pending_tool(server_id, tool) for tool in tools)

    remote_names = [tool.remote_name for tool in pending]
    if len(remote_names) != len(set(remote_names)):
        raise ValueError("MCP tools/list contains duplicate tool names")

    base_counts = Counter(tool.provider_base for tool in pending)
    reserved = set(reserved_provider_names)
    assigned = set(reserved)

    discovered: list[McpDiscoveredTool] = []

    for tool in pending:
        requires_hash = (
            len(tool.provider_base) > _PROVIDER_NAME_MAX_LENGTH
            or base_counts[tool.provider_base] > 1
            or tool.provider_base in reserved
        )

        if requires_hash:
            provider_name = _hashed_provider_name(
                tool.provider_base,
                tool.canonical_id,
            )
        else:
            provider_name = tool.provider_base

        # A digest collision is extraordinarily unlikely, but identity
        # correctness must not depend on probability.
        digest_length = 10
        while provider_name in assigned:
            digest_length += 2
            if digest_length > 64:
                raise ValueError("Unable to allocate unique MCP provider name")
            provider_name = _hashed_provider_name(
                tool.provider_base,
                tool.canonical_id,
                digest_length=digest_length,
            )

        assigned.add(provider_name)

        definition = ToolDefinition(
            name=provider_name,
            description=tool.description,
            input_schema=deepcopy(tool.input_schema),
        )

        discovered.append(
            McpDiscoveredTool(
                server_id=tool.server_id,
                remote_name=tool.remote_name,
                canonical_id=tool.canonical_id,
                provider_name=provider_name,
                definition=definition,
            )
        )

    return tuple(discovered)


def adapt_mcp_list_tools_result(
    server_id: str,
    result: ListToolsResult,
    *,
    reserved_provider_names: Iterable[str] = (),
) -> McpToolDiscoveryPage:
    """Adapt one raw MCP ``tools/list`` result without losing pagination."""

    return McpToolDiscoveryPage(
        tools=adapt_mcp_tools(
            server_id,
            result.tools,
            reserved_provider_names=reserved_provider_names,
        ),
        next_cursor=result.next_cursor,
    )


__all__ = [
    "McpDiscoveredTool",
    "McpToolDiscoveryPage",
    "adapt_mcp_list_tools_result",
    "adapt_mcp_tools",
]
