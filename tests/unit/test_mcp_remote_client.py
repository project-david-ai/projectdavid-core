import asyncio
import contextlib
import socket
from collections.abc import AsyncIterator
from typing import Any

import httpx2
import pytest
import uvicorn
from entities_api.orchestration.mcp_remote_client import (
    McpClientStateError,
    RemoteMcpClient,
)
from mcp.server import MCPServer
from mcp.types import CallToolResult, ListToolsResult, TextContent
from typing_extensions import Self


class _HeaderRecorder:
    def __init__(self, app: Any) -> None:
        self.app = app
        self.authorization_headers: list[bytes | None] = []

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            headers = dict(scope["headers"])
            self.authorization_headers.append(headers.get(b"authorization"))
        await self.app(scope, receive, send)


@contextlib.asynccontextmanager
async def _serve(app: Any) -> AsyncIterator[str]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    sock.setblocking(False)
    port = sock.getsockname()[1]

    server = uvicorn.Server(uvicorn.Config(app, log_level="critical", lifespan="on"))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        for _ in range(500):
            if server.started:
                break
            if task.done():
                await task
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("test MCP server did not start")

        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, timeout=5)
        sock.close()


def _test_server() -> MCPServer:
    server = MCPServer(name="transport-test", version="1.0.0")

    @server.tool()
    async def echo(message: str) -> str:
        """Return the supplied message."""
        return message

    return server


def test_remote_streamable_http_connection_lists_and_calls_tools() -> None:
    async def exercise() -> None:
        server = _test_server()
        app = server.streamable_http_app(json_response=True, stateless_http=True)

        async with _serve(app) as url:
            client = RemoteMcpClient(url, read_timeout_seconds=2)
            assert client.connected is False

            async with client:
                assert client.connected is True
                assert client.protocol_version
                assert client.server_info is not None
                assert client.server_info.name == "transport-test"

                listing = await client.list_tools()
                assert isinstance(listing, ListToolsResult)
                assert [tool.name for tool in listing.tools] == ["echo"]

                result = await client.call_tool("echo", {"message": "hello"})
                assert isinstance(result, CallToolResult)
                assert result.is_error is False
                assert result.content == [TextContent(type="text", text="hello")]

            assert client.connected is False

    asyncio.run(exercise())


def test_external_http_client_supplies_auth_and_remains_caller_owned() -> None:
    async def exercise() -> None:
        server = _test_server()
        app = _HeaderRecorder(
            server.streamable_http_app(json_response=True, stateless_http=True)
        )

        async with _serve(app) as url:
            http_client = httpx2.AsyncClient(
                headers={"Authorization": "Bearer externally-managed"}
            )
            try:
                async with RemoteMcpClient(
                    url,
                    http_client=http_client,
                    read_timeout_seconds=2,
                ) as client:
                    await client.list_tools()

                assert http_client.is_closed is False
                assert app.authorization_headers
                assert set(app.authorization_headers) == {b"Bearer externally-managed"}
            finally:
                await http_client.aclose()

    asyncio.run(exercise())


def test_operations_require_an_active_connection() -> None:
    client = RemoteMcpClient("https://mcp.example.test/mcp")

    with pytest.raises(McpClientStateError, match="not connected"):
        asyncio.run(client.list_tools())


@pytest.mark.parametrize(
    "url",
    [
        "mcp.example.test/mcp",
        "ftp://mcp.example.test/mcp",
        "http:///mcp",
        "https://user:secret@mcp.example.test/mcp",
    ],
)
def test_remote_client_rejects_non_http_or_inline_credential_urls(url: str) -> None:
    with pytest.raises(ValueError):
        RemoteMcpClient(url)


def test_sdk_timeout_and_caller_cancellation_are_not_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, Any]] = []
    cancelled = asyncio.Event()

    class FakeSdkClient:
        def __init__(
            self,
            transport: Any,
            *,
            read_timeout_seconds: float | None,
        ) -> None:
            calls.append(("init-timeout", read_timeout_seconds))

        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def list_tools(self, *, cursor: str | None = None) -> None:
            calls.append(("list-cursor", cursor))
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

    monkeypatch.setattr(
        "entities_api.orchestration.mcp_remote_client.Client", FakeSdkClient
    )

    async def exercise() -> None:
        client = RemoteMcpClient(
            "https://mcp.example.test/mcp", read_timeout_seconds=1.25
        )
        async with client:
            task = asyncio.create_task(client.list_tools(cursor="next"))
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.wait_for(cancelled.wait(), timeout=1)

    asyncio.run(exercise())
    assert calls == [("init-timeout", 1.25), ("list-cursor", "next")]
