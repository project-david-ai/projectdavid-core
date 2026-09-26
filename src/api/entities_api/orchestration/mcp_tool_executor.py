"""Execute discovered MCP tools through Project David's Tool ABI.

The MCP adapter preserves rich protocol results inside ToolResultEnvelope while
also providing the legacy string projection required by Core's current durable
tool-output persistence path.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from typing import Any

from mcp.types import CallToolResult, TextContent

from .mcp_remote_client import RemoteMcpClient
from .mcp_tool_discovery import McpDiscoveredTool
from .tool_abi import ToolCallEnvelope, ToolResultEnvelope


class McpToolExecutor:
    """Execute one discovered MCP tool using a fresh remote client session."""

    def __init__(
        self,
        tool: McpDiscoveredTool,
        client_factory: Callable[[], RemoteMcpClient],
    ) -> None:
        self._tool = tool
        self._client_factory = client_factory

    @property
    def tool(self) -> McpDiscoveredTool:
        return self._tool

    @property
    def provider_name(self) -> str:
        return self._tool.provider_name

    @staticmethod
    def _serialize_content_blocks(
        result: CallToolResult,
    ) -> tuple[dict[str, Any], ...]:
        """Convert MCP SDK content models into transport-neutral dictionaries."""

        return tuple(
            block.model_dump(
                mode="json",
                by_alias=True,
                exclude_none=True,
            )
            for block in result.content
        )

    @classmethod
    def _legacy_content(
        cls,
        result: CallToolResult,
        content_blocks: tuple[dict[str, Any], ...],
    ) -> str:
        """Produce the string projection consumed by today's persistence path."""

        text_parts = [
            block.text for block in result.content if isinstance(block, TextContent)
        ]

        if text_parts:
            return "\n".join(text_parts)

        if result.structured_content is not None:
            return json.dumps(
                result.structured_content,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )

        if content_blocks:
            return json.dumps(
                content_blocks,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )

        return ""

    @classmethod
    def _adapt_result(
        cls,
        result: CallToolResult,
    ) -> ToolResultEnvelope:
        """Preserve the rich MCP result while retaining legacy compatibility."""

        content_blocks = cls._serialize_content_blocks(result)

        metadata: dict[str, Any] = {}

        if result.meta is not None:
            metadata["mcp_meta"] = deepcopy(result.meta)

        if result.result_type is not None:
            metadata["mcp_result_type"] = result.result_type

        return ToolResultEnvelope(
            content=cls._legacy_content(result, content_blocks),
            structured_content=deepcopy(result.structured_content),
            content_blocks=content_blocks,
            metadata=metadata,
            is_error=result.is_error,
        )

    async def execute(self, call: ToolCallEnvelope) -> ToolResultEnvelope:
        """Execute the mapped remote MCP tool."""

        if call.name != self.provider_name:
            raise ValueError(
                "MCP executor received a call for a different provider alias"
            )

        try:
            async with self._client_factory() as client:
                result = await client.call_tool(
                    self._tool.remote_name,
                    dict(call.arguments),
                )
        # Remote MCP execution is a fault boundary: transport, protocol, and
        # server failures must become tool failures rather than escape into the
        # Project David orchestration loop.
        except Exception as exc:  # noqa: BLE001
            return ToolResultEnvelope(
                content=(
                    f"ERROR: MCP tool '{self.provider_name}' failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
                is_error=True,
            )

        return self._adapt_result(result)


__all__ = ["McpToolExecutor"]
