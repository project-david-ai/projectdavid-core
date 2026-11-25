"""Origin-bound HTTP transport for tenant-managed remote MCP connections."""

from __future__ import annotations

import asyncio

import httpx2

from src.api.entities_api.services.outbound_network_policy import (
    OutboundPolicyError,
    origin,
    resolve_mcp_destination,
)


class McpOutboundTransport(httpx2.AsyncHTTPTransport):
    """Pin one authorized IP per MCP connection context, including redirects.

    Keep the SDK-facing URL unchanged. Only the HTTP transport sees the
    numeric destination. The original authority remains the Host and TLS name.
    Each registration context owns a separate pool.
    """

    def __init__(self, endpoint: str, *, resolution_timeout: float = 10.0) -> None:
        self._allowed_origin = origin(endpoint)
        self._resolution_timeout = resolution_timeout
        self._pinned_address: str | None = None
        self._pin_lock = asyncio.Lock()
        super().__init__(verify=True, trust_env=False, retries=0)

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        if origin(str(request.url)) != self._allowed_origin:
            raise OutboundPolicyError("MCP cross-origin request blocked")
        async with self._pin_lock:
            if self._pinned_address is None:
                self._pinned_address = await asyncio.wait_for(
                    resolve_mcp_destination(self._allowed_origin),
                    timeout=self._resolution_timeout,
                )
            address = self._pinned_address
        scheme, host, port = self._allowed_origin
        authority = f"[{host}]" if ":" in host else host
        if port != (443 if scheme == "https" else 80):
            authority = f"{authority}:{port}"
        headers = httpx2.Headers(request.headers)
        headers["Host"] = authority
        pinned = httpx2.Request(
            method=request.method,
            url=request.url.copy_with(host=address),
            headers=headers,
            stream=request.stream,
            extensions={**request.extensions, "sni_hostname": host},
        )
        return await super().handle_async_request(pinned)
