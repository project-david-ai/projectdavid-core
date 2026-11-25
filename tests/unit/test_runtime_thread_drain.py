"""Thread ownership survives HTTP exit and bridge closure."""

import asyncio
from threading import Event

import pytest

from src.api.entities_api.clients.async_to_sync import async_to_sync_stream
from src.api.entities_api.services import request_activity as activity
from src.api.entities_api.services.request_activity import (
    RequestActivityMiddleware,
    RequestActivityRegistry,
    start_tracked_thread,
    track_authenticated_request,
)


async def _receive():
    return {"type": "http.disconnect"}


async def _send(message):
    pass


def _request(app, registry):
    asyncio.run(
        RequestActivityMiddleware(app, registry)({"type": "http"}, _receive, _send)
    )


def test_thread_outlives_http_request():
    registry = RequestActivityRegistry()
    release = Event()
    threads = []

    async def app(scope, receive, send):
        track_authenticated_request("owner")
        threads.append(start_tracked_thread(lambda: release.wait(5)))
        assert registry.active_count("owner") == 2

    try:
        _request(app, registry)
        assert registry.active_count("owner") == 1
        assert registry.active_count("other") == 0
    finally:
        release.set()
        for thread in threads:
            thread.join(5)
    assert not threads[0].is_alive()
    assert registry.active_count("owner") == 0


def test_failed_thread_start_releases_reference(monkeypatch):
    registry = RequestActivityRegistry()

    class FailedThread:
        def __init__(self, **kwargs):
            pass

        def start(self):
            raise RuntimeError("cannot start thread")

    monkeypatch.setattr(activity, "Thread", FailedThread)

    async def app(scope, receive, send):
        track_authenticated_request("owner")
        with pytest.raises(RuntimeError, match="cannot start thread"):
            start_tracked_thread(lambda: None)
        assert registry.active_count("owner") == 1

    _request(app, registry)
    assert registry.active_count("owner") == 0


def test_bridge_preserves_order_and_releases_on_exhaustion():
    registry = RequestActivityRegistry()

    async def source():
        for value in range(150):
            yield value

    async def app(scope, receive, send):
        track_authenticated_request("owner")
        assert list(async_to_sync_stream(source())) == list(range(150))
        assert registry.active_count("owner") == 1

    _request(app, registry)
    assert registry.active_count("owner") == 0


def test_bridge_propagates_producer_error_after_buffered_items():
    registry = RequestActivityRegistry()

    async def source():
        yield 1
        raise ValueError("source failed")

    async def app(scope, receive, send):
        track_authenticated_request("owner")
        stream = async_to_sync_stream(source())
        assert next(stream) == 1
        with pytest.raises(ValueError, match="source failed"):
            next(stream)
        assert registry.active_count("owner") == 1

    _request(app, registry)
    assert registry.active_count("owner") == 0


@pytest.mark.parametrize("backpressure", [True, False])
def test_early_close_retains_producer_until_cleanup(monkeypatch, backpressure):
    from src.api.entities_api.clients import async_to_sync as bridge

    registry = RequestActivityRegistry()
    entered = Event()
    cleaning = Event()
    release = Event()
    threads = []
    original = bridge.start_tracked_thread

    def capture(target):
        thread = original(target)
        threads.append(thread)
        return thread

    monkeypatch.setattr(bridge, "start_tracked_thread", capture)

    async def source():
        try:
            yield 0
            if backpressure:
                for value in range(1, 500):
                    if value == 101:
                        entered.set()
                    yield value
            else:
                entered.set()
                await asyncio.Event().wait()
        finally:
            cleaning.set()
            # Deliberately hold cleanup to prove the producer stays counted.
            await asyncio.to_thread(release.wait, 5)

    async def app(scope, receive, send):
        track_authenticated_request("owner")
        stream = async_to_sync_stream(source())
        try:
            assert next(stream) == 0
            assert entered.wait(5)
        finally:
            stream.close()
        assert cleaning.wait(5)
        assert registry.active_count("owner") == 2

    try:
        _request(app, registry)
        assert registry.active_count("owner") == 1
    finally:
        release.set()
        for thread in threads:
            thread.join(5)
    assert not threads[0].is_alive()
    assert registry.active_count("owner") == 0
