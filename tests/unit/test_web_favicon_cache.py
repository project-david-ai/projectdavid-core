import asyncio
import base64
import io
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from src.api.entities_api.cache.web_cache import REDIS_WEB_TTL, WebSessionCache
from src.api.entities_api.services import web_reader

URL = "https://docs.python.org/3/"
OTHER_URL = "https://docs.python.org/other/"


class MemoryRedis:
    def __init__(self):
        self.data = {}
        self.expiration = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value, *, ex):
        self.data[key] = value.encode("utf-8")
        self.expiration[key] = ex

    def delete(self, key):
        self.data.pop(key, None)
        self.expiration.pop(key, None)


@pytest.fixture
def icon():
    output = io.BytesIO()
    Image.new("RGBA", (32, 32), (1, 2, 3, 255)).save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode(
        "ascii"
    )


@pytest.fixture
def cache():
    return WebSessionCache(MemoryRedis())


@pytest.fixture
def reader(cache, monkeypatch):
    monkeypatch.setenv("WEB_FAVICONS_ENABLED", "true")
    return web_reader.UniversalWebReader(cache)


def test_cache_round_trip_and_ttl(cache, icon):
    async def scenario():
        await cache.save_favicon(URL, icon)
        assert await cache.get_favicon(URL) == icon
        assert await cache.get_favicon(OTHER_URL) is None
        assert cache.redis.expiration[cache._favicon_key(URL)] == REDIS_WEB_TTL
        assert await cache.get_session(URL) is None

    asyncio.run(scenario())


def test_page_content_contract_unchanged(cache, icon):
    async def scenario():
        await cache.save_session(URL, "hello world", ["hello world"], "Test")
        before = await cache.get_page_view(URL, 0)
        search_before = await cache.search_session(URL, "hello")
        await cache.save_favicon(URL, icon)
        assert await cache.get_page_view(URL, 0) == before
        assert await cache.search_session(URL, "hello") == search_before
        assert "favicon" not in await cache.get_session(URL)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "bad",
    [
        "https://example.com/icon.png",
        "data:image/svg+xml;base64,AA==",
        "data:image/png;base64,AA==",
        "data:image/png;base64," + "A" * 8192,
    ],
)
def test_invalid_cache_payload_rejected(cache, bad):
    async def scenario():
        await cache.save_favicon(URL, bad)
        assert await cache.get_favicon(URL) is None
        cache.redis.data[cache._favicon_key(URL)] = bad.encode("ascii")
        assert await cache.get_favicon(URL) is None

    asyncio.run(scenario())


def test_reader_acquires_and_reuses_icon(reader, monkeypatch, icon):
    fetch = AsyncMock(return_value=icon)
    monkeypatch.setattr(web_reader, "resolve_favicon", fetch)

    async def scenario():
        await reader._cache_page_favicon(URL, URL, "/favicon.ico")
        await reader._cache_page_favicon(URL, URL, "/favicon.ico")
        assert await reader.get_favicon(URL) == icon
        fetch.assert_awaited_once_with(URL, "/favicon.ico")

    asyncio.run(scenario())


def test_disabled_neither_reads_nor_fetches(reader, monkeypatch, icon):
    fetch = AsyncMock(return_value=icon)
    monkeypatch.setattr(web_reader, "resolve_favicon", fetch)
    monkeypatch.setenv("WEB_FAVICONS_ENABLED", "false")

    async def scenario():
        await reader.cache.save_favicon(URL, icon)
        await reader._cache_page_favicon(URL, URL, "/favicon.ico")
        assert await reader.get_favicon(URL) is None
        fetch.assert_not_awaited()

    asyncio.run(scenario())


def test_cross_origin_redirect_clears_cached_icon(reader, monkeypatch, icon):
    fetch = AsyncMock(return_value=icon)
    monkeypatch.setattr(web_reader, "resolve_favicon", fetch)

    async def scenario():
        await reader.cache.save_favicon(URL, icon)
        await reader._cache_page_favicon(URL, "https://example.org/", "/favicon.ico")
        assert await reader.get_favicon(URL) is None
        fetch.assert_not_awaited()

    asyncio.run(scenario())


def test_cache_failure_is_nonfatal(reader, monkeypatch, icon):
    monkeypatch.setattr(
        reader.cache, "get_favicon", AsyncMock(side_effect=RuntimeError("redis"))
    )
    monkeypatch.setattr(
        reader.cache, "save_favicon", AsyncMock(side_effect=RuntimeError("redis"))
    )
    monkeypatch.setattr(web_reader, "resolve_favicon", AsyncMock(return_value=icon))

    async def scenario():
        await reader._cache_page_favicon(URL, URL, None)
        assert await reader.get_favicon(URL) is None

    asyncio.run(scenario())


def test_cached_read_does_not_fetch(reader, monkeypatch, icon):
    fetch = AsyncMock(return_value=icon)
    scrape = AsyncMock(side_effect=AssertionError("Unexpected browser call"))
    monkeypatch.setattr(web_reader, "resolve_favicon", fetch)
    monkeypatch.setattr(reader, "_fetch_via_browserless", scrape)

    async def scenario():
        await reader.cache.save_session(URL, "hello world", ["hello world"], "Test")
        assert await reader.read(URL) == await reader.cache.get_page_view(URL, 0)
        assert await reader.get_favicon(URL) is None
        fetch.assert_not_awaited()
        scrape.assert_not_awaited()

    asyncio.run(scenario())
