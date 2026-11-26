import asyncio
import base64
import io
import socket
from unittest.mock import AsyncMock

import httpx
import pytest
from PIL import Image

from src.api.entities_api.services import favicon_resolver as resolver


def _png(size=(64, 64)):
    buffer = io.BytesIO()
    Image.new("RGBA", size, (10, 20, 30, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/",
        "https://localhost/",
        "https://redis/",
        "https://127.0.0.1/",
        "https://[::1]/",
        "https://127.1/",
        "https://0x7f000001/",
        "https://example.com:8443/",
        "https://user:password@example.com/",
        "https://example.com\\@internal/",
        "https://example.com./",
        "https://%65xample.com/",
    ],
)
def test_reject_unsafe_page_url(url):
    assert resolver.favicon_candidate(url) is None


@pytest.mark.parametrize(
    "candidate",
    [
        "https://other.example.com/icon.png",
        "//redis/icon.png",
        "https://example.com:8443/icon.png",
        "https://user@example.com/icon.png",
        "https://example.com/icon.svg",
        "/icon.png?secret=123",
        "/icon.png#fragment",
        "data:image/png;base64,AA==",
        "/bad\\icon.png",
    ],
)
def test_untrusted_icon_falls_back_to_origin(candidate):
    assert str(resolver.favicon_candidate("https://example.com/docs/", candidate)) == (
        "https://example.com/favicon.ico"
    )


def test_same_origin_relative_icon():
    assert str(resolver.favicon_candidate("https://example.com/docs/", "icon.png")) == (
        "https://example.com/docs/icon.png"
    )


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.1",
        "172.16.0.1",
        "192.168.1.1",
        "169.254.169.254",
        "100.64.0.1",
        "0.0.0.0",
        "224.0.0.1",
        "240.0.0.1",
        "::1",
        "fc00::1",
        "fe80::1",
        "::ffff:8.8.8.8",
        "64:ff9b::a00:1",
        "2002:0a00:0001::1",
        "2001:0000::1",
        "ff02::1",
        "2606:4700::1%eth0",
    ],
)
def test_reject_nonpublic_and_transition_addresses(address):
    assert not resolver._public_address(address)


@pytest.mark.parametrize("address", ["8.8.8.8", "2606:4700:4700::1111"])
def test_public_address(address):
    assert resolver._public_address(address)


def test_mixed_dns_answers_fail_closed(monkeypatch):
    async def run():
        loop = asyncio.get_running_loop()
        lookup = AsyncMock(
            return_value=[
                (
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                    socket.IPPROTO_TCP,
                    "",
                    ("8.8.8.8", 443),
                ),
                (
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                    socket.IPPROTO_TCP,
                    "",
                    ("10.0.0.1", 443),
                ),
            ]
        )
        monkeypatch.setattr(loop, "getaddrinfo", lookup)
        assert await resolver._resolve_public("example.com") is None
        lookup.assert_awaited_once()

    asyncio.run(run())


def test_normalize_to_inert_png():
    value = resolver._normalize_image(_png())
    assert value.startswith(resolver.DATA_PREFIX)
    with Image.open(
        io.BytesIO(base64.b64decode(value[len(resolver.DATA_PREFIX) :]))
    ) as icon:
        assert icon.format == "PNG"
        assert icon.size == (32, 32)
        assert icon.mode == "RGBA"
        assert not icon.info


@pytest.mark.parametrize(
    "body",
    [b"<svg/>", b"garbage", b"", b"x" * (resolver.MAX_DOWNLOAD + 1), _png((513, 1))],
)
def test_reject_invalid_or_oversized_image(body):
    assert resolver._normalize_image(body) is None


def test_disabled_never_resolves_dns(monkeypatch):
    monkeypatch.delenv("WEB_FAVICONS_ENABLED", raising=False)
    fetch = AsyncMock()
    monkeypatch.setattr(resolver, "_fetch_icon", fetch)
    assert asyncio.run(resolver.resolve_favicon("https://example.com")) is None
    fetch.assert_not_awaited()


def _mock_client(monkeypatch, handler):
    original = httpx.AsyncClient

    def client(**kwargs):
        assert kwargs["trust_env"] is False
        assert kwargs["follow_redirects"] is False
        assert kwargs["verify"] is True
        return original(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(resolver.httpx, "AsyncClient", client)
    monkeypatch.setattr(resolver, "_resolve_public", AsyncMock(return_value="8.8.8.8"))
    monkeypatch.setenv("WEB_FAVICONS_ENABLED", "true")


def test_pinned_request_preserves_tls_name(monkeypatch):
    class RawBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield _png()

    def handler(request):
        assert str(request.url) == "https://8.8.8.8/favicon.ico"
        assert request.headers["host"] == "example.com"
        assert request.extensions["sni_hostname"] == "example.com"
        assert "authorization" not in request.headers
        assert "cookie" not in request.headers
        return httpx.Response(200, stream=RawBody())

    _mock_client(monkeypatch, handler)
    value = asyncio.run(
        resolver.resolve_favicon("https://example.com/private?token=123")
    )
    assert value and value.startswith(resolver.DATA_PREFIX)


@pytest.mark.parametrize(
    "status,headers",
    [
        (302, {"location": "http://redis/"}),
        (404, {}),
        (200, {"content-length": str(resolver.MAX_DOWNLOAD + 1)}),
        (200, {"content-encoding": "gzip"}),
    ],
)
def test_no_redirect_or_invalid_response(monkeypatch, status, headers):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(status, headers=headers, content=b"")

    _mock_client(monkeypatch, handler)
    assert asyncio.run(resolver.resolve_favicon("https://example.com")) is None
    assert len(requests) == 1


def test_streaming_download_limit(monkeypatch):
    class HugeBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            for _ in range(33):
                yield b"x" * 8192

    _mock_client(monkeypatch, lambda request: httpx.Response(200, stream=HugeBody()))
    assert asyncio.run(resolver.resolve_favicon("https://example.com")) is None


def test_network_deadline(monkeypatch):
    monkeypatch.setenv("WEB_FAVICONS_ENABLED", "true")
    monkeypatch.setattr(resolver, "DEADLINE_SECONDS", 0.01)

    async def stalled(url):
        await asyncio.sleep(10)

    monkeypatch.setattr(resolver, "_fetch_icon", stalled)
    assert asyncio.run(resolver.resolve_favicon("https://example.com")) is None


def test_network_failure_is_optional(monkeypatch):
    monkeypatch.setenv("WEB_FAVICONS_ENABLED", "true")
    monkeypatch.setattr(
        resolver, "_fetch_icon", AsyncMock(side_effect=OSError("failed"))
    )
    assert asyncio.run(resolver.resolve_favicon("https://example.com")) is None
