import asyncio
from unittest.mock import AsyncMock

import httpx2
import pytest

from src.api.entities_api.orchestration import mcp_outbound_transport as transport


def test_pins_all_requests_preserves_tls_and_original_url(monkeypatch):
    async def exercise():
        resolve = AsyncMock(return_value="8.8.8.8")
        monkeypatch.setattr(transport, "resolve_mcp_destination", resolve)
        seen = []

        async def send(self, request):
            seen.append(request)
            return httpx2.Response(200, content=b"ok")

        monkeypatch.setattr(httpx2.AsyncHTTPTransport, "handle_async_request", send)
        guarded = transport.McpOutboundTransport("https://example.com/mcp")
        async with httpx2.AsyncClient(
            transport=guarded, trust_env=False, headers={"Authorization": "Bearer test"}
        ) as client:
            for method in ("POST", "GET", "DELETE"):
                result = await client.request(
                    method, "https://example.com/mcp?cursor=1"
                )
                assert str(result.request.url) == "https://example.com/mcp?cursor=1"
            with pytest.raises(transport.OutboundPolicyError):
                await client.get("https://different.example/mcp")
            with pytest.raises(transport.OutboundPolicyError):
                await client.get("http://example.com/mcp")
        resolve.assert_awaited_once_with(("https", "example.com", 443))
        assert [r.method for r in seen] == ["POST", "GET", "DELETE"]
        for request in seen:
            assert str(request.url) == "https://8.8.8.8/mcp?cursor=1"
            assert request.headers["host"] == "example.com"
            assert request.extensions["sni_hostname"] == "example.com"
            assert request.headers["authorization"] == "Bearer test"

    asyncio.run(exercise())


def test_denied_resolution_never_reaches_transport(monkeypatch):
    async def exercise():
        monkeypatch.setattr(
            transport,
            "resolve_mcp_destination",
            AsyncMock(side_effect=transport.OutboundPolicyError("blocked")),
        )
        send = AsyncMock()
        monkeypatch.setattr(httpx2.AsyncHTTPTransport, "handle_async_request", send)
        async with httpx2.AsyncClient(
            transport=transport.McpOutboundTransport("https://example.com/mcp"),
            trust_env=False,
        ) as client:
            with pytest.raises(transport.OutboundPolicyError):
                await client.post("https://example.com/mcp")
        send.assert_not_awaited()

    asyncio.run(exercise())
