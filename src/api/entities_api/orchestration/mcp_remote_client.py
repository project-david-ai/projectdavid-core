"""Remote MCP client transport built on the official MCP Python SDK.

This module deliberately stops at the protocol boundary. Tool discovery mapping,
tool routing, persistence, and policy enforcement belong to later MCP milestones.
"""

from __future__ import annotations

import math
from types import TracebackType
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import (
    CallToolResult,
    Implementation,
    ListToolsResult,
    ServerCapabilities,
)
from typing_extensions import Self

if TYPE_CHECKING:
    from httpx2 import AsyncClient


class McpClientStateError(RuntimeError):
    """Raised when an MCP operation is attempted outside an active connection."""


class RemoteMcpClient:
    """Single-use client for a remote MCP Streamable HTTP endpoint.

    Authentication, headers, and TLS customization can be supplied through a
    caller-owned ``httpx2.AsyncClient``. The official SDK does not close a client
    supplied this way, and this wrapper preserves that ownership boundary. It
    never accepts or persists raw credentials itself.

    The values returned by ``list_tools`` and ``call_tool`` are the official MCP
    SDK result types. Adapting them to Project David's tool ABI is MCP-2/MCP-4
    work, not a transport concern.
    """

    def __init__(
        self,
        url: str,
        *,
        http_client: AsyncClient | None = None,
        read_timeout_seconds: float | None = 30.0,
        terminate_on_close: bool = True,
    ) -> None:
        self._url = self._validate_url(url)
        self._read_timeout_seconds = self._validate_timeout(read_timeout_seconds)
        transport = streamable_http_client(
            self._url,
            http_client=http_client,
            terminate_on_close=terminate_on_close,
        )
        self._client = Client(
            transport,
            read_timeout_seconds=self._read_timeout_seconds,
        )
        self._connected = False
        self._entered = False

    @staticmethod
    def _validate_url(url: str) -> str:
        try:
            parsed = urlsplit(url)
            parsed_port = parsed.port
        except ValueError as exc:
            raise ValueError("MCP endpoint URL is invalid") from exc

        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError(
                "remote MCP endpoint must be an absolute HTTP or HTTPS URL"
            )
        if parsed.username is not None or parsed.password is not None:
            raise ValueError(
                "MCP endpoint credentials must be supplied by an external HTTP client"
            )
        if parsed_port is not None and not 1 <= parsed_port <= 65535:
            raise ValueError("MCP endpoint URL contains an invalid port")
        return url

    @staticmethod
    def _validate_timeout(timeout: float | None) -> float | None:
        if timeout is None:
            return None
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("read_timeout_seconds must be positive and finite")
        return timeout

    @property
    def connected(self) -> bool:
        """Whether the official SDK handshake has completed successfully."""

        return self._connected

    def _active_client(self) -> Client:
        if not self._connected:
            raise McpClientStateError("MCP client is not connected")
        return self._client

    @property
    def protocol_version(self) -> str:
        """Negotiated MCP protocol version for the active connection."""

        return self._active_client().protocol_version

    @property
    def server_info(self) -> Implementation | None:
        """Server identity advertised during the MCP handshake, when present."""

        return self._active_client().server_info

    @property
    def server_capabilities(self) -> ServerCapabilities:
        """Capabilities advertised by the connected MCP server."""

        return self._active_client().server_capabilities

    @property
    def instructions(self) -> str | None:
        """Optional instructions advertised by the connected MCP server."""

        return self._active_client().instructions

    async def __aenter__(self) -> Self:
        if self._entered:
            raise McpClientStateError("MCP client connections cannot be reused")
        self._entered = True
        await self._client.__aenter__()
        self._connected = True
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            await self._client.__aexit__(exc_type, exc_value, traceback)
        finally:
            self._connected = False

    async def list_tools(self, *, cursor: str | None = None) -> ListToolsResult:
        """Return one raw ``tools/list`` page from the connected MCP server."""

        return await self._active_client().list_tools(cursor=cursor)

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> CallToolResult:
        """Return the raw result of a ``tools/call`` request."""

        return await self._active_client().call_tool(name, arguments)
