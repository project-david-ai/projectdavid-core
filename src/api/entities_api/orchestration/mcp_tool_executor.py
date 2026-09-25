"""Execute discovered MCP tools through Project David's Tool ABI.

This adapter owns the remote MCP tools/call boundary only. Durable Run/Action
lifecycle, persistence, routing policy, and assistant capability policy remain
owned by Project David's orchestration layer.
"""

from __future__ import annotations

import json
from collections.abc import Callable

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
    def _legacy_content(result: object) -> str:
        """Flatten an MCP result into Core's current string result contract.

        Rich structured/content-block preservation belongs to MCP-4. MCP-3
        prefers textual MCP content and uses structured JSON only when no text
        representation is available.
        """

        content_blocks = getattr(result, "content", None) or []

        text_parts = [
            text
            for block in content_blocks
            if isinstance((text := getattr(block, "text", None)), str)
        ]

        if text_parts:
            return "\n".join(text_parts)

        structured = getattr(result, "structured_content", None)
        if structured is not None:
            return json.dumps(
                structured,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )

        if content_blocks:
            serializable = []

            for block in content_blocks:
                model_dump = getattr(block, "model_dump", None)

                if callable(model_dump):
                    serializable.append(
                        model_dump(
                            mode="json",
                            by_alias=True,
                            exclude_none=True,
                        )
                    )
                else:
                    serializable.append(str(block))

            return json.dumps(
                serializable,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )

        return ""

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

        return ToolResultEnvelope(
            content=self._legacy_content(result),
            is_error=bool(getattr(result, "is_error", False)),
        )


__all__ = ["McpToolExecutor"]
