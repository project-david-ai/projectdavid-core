# src/api/entities_api/orchestration/mcp_tool_executor.py
"""Execute discovered MCP tools through Project David's Tool ABI.

The MCP adapter preserves rich protocol results inside ToolResultEnvelope while
also providing the legacy string projection required by Core's current durable
tool-output persistence path.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from typing import Any, AsyncContextManager

from mcp.types import CallToolResult, TextContent

from .mcp_remote_client import RemoteMcpClient
from .mcp_tool_discovery import McpDiscoveredTool
from .tool_abi import ToolCallEnvelope, ToolResultEnvelope


class McpToolExecutor:
    """Execute one discovered MCP tool using a fresh remote client session."""

    def __init__(
        self,
        tool: McpDiscoveredTool,
        client_factory: Callable[[], AsyncContextManager[RemoteMcpClient]],
    ) -> None:
        self._tool = tool
        self._client_factory = client_factory

    @property
    def tool(self) -> McpDiscoveredTool:
        return self._tool

    @property
    def provider_name(self) -> str:
        return self._tool.provider_name

    @property
    def server_id(self) -> str:
        """Durable MCP registration identity for this tool."""
        return self._tool.server_id

    def client_context(self) -> AsyncContextManager[RemoteMcpClient]:
        """Create the registered remote-client context for runtime ownership."""
        return self._client_factory()

    @staticmethod
    def _is_timeout_error(exc: BaseException) -> bool:
        """Recognise timeout failures across SDK/transport exception wrappers."""

        current: BaseException | None = exc
        seen: set[int] = set()

        while current is not None and id(current) not in seen:
            seen.add(id(current))

            name = type(current).__name__.lower()

            if isinstance(current, TimeoutError) or "timeout" in name:
                return True

            current = (
                current.__cause__
                if current.__cause__ is not None
                else current.__context__
            )

        return False

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

    def failure_result(
        self,
        exc: Exception,
    ) -> ToolResultEnvelope:
        """Map an MCP transport/protocol failure onto the tool ABI."""

        if self._is_timeout_error(exc):
            return ToolResultEnvelope(
                content=(f"ERROR: MCP tool '{self.provider_name}' timed out"),
                metadata={
                    "mcp_timeout": True,
                    "exception_type": type(exc).__name__,
                },
                is_error=True,
            )

        return ToolResultEnvelope(
            content=(
                f"ERROR: MCP tool '{self.provider_name}' failed: "
                f"{type(exc).__name__}: {exc}"
            ),
            is_error=True,
        )

    async def execute_connected(
        self,
        client: RemoteMcpClient,
        call: ToolCallEnvelope,
        *,
        progress_callback: Any | None = None,
    ) -> ToolResultEnvelope:
        """Execute through an already-active Project David-owned MCP session."""

        if call.name != self.provider_name:
            raise ValueError(
                "MCP executor received a call for a different provider alias"
            )

        try:
            result = await client.call_tool(
                self._tool.remote_name,
                dict(call.arguments),
                progress_callback=progress_callback,
            )
        except Exception as exc:  # noqa: BLE001
            return self.failure_result(exc)

        return self._adapt_result(result)

    async def execute(
        self,
        call: ToolCallEnvelope,
        *,
        progress_callback: Any | None = None,
    ) -> ToolResultEnvelope:
        """Execute using a short-lived MCP session.

        This remains the standalone/backwards-compatible execution path.
        Runtime orchestration may instead provide a run-affine connected client
        through ``execute_connected``.
        """

        if call.name != self.provider_name:
            raise ValueError(
                "MCP executor received a call for a different provider alias"
            )

        try:
            async with self.client_context() as client:
                return await self.execute_connected(
                    client,
                    call,
                    progress_callback=progress_callback,
                )
        except Exception as exc:  # noqa: BLE001
            return self.failure_result(exc)


__all__ = ["McpToolExecutor"]
