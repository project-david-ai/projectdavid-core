"""Offloaded work must remain counted after its HTTP waiter exits."""

import asyncio
from threading import Event

import pytest

from src.api.entities_api.services.request_activity import (
    RequestActivityMiddleware,
    RequestActivityRegistry,
    run_tracked_sync,
    track_authenticated_request,
)


async def _receive():
    return {"type": "http.disconnect"}


async def _send(message):
    pass


def test_cancelled_waiter_does_not_release_running_thread():
    registry = RequestActivityRegistry()
    release_worker = Event()

    async def scenario():
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()

        def worker():
            loop.call_soon_threadsafe(entered.set)
            assert release_worker.wait(5)

        async def app(scope, receive, send):
            track_authenticated_request("owner")
            await run_tracked_sync(worker)

        task = asyncio.create_task(
            RequestActivityMiddleware(app, registry)({"type": "http"}, _receive, _send)
        )
        try:
            await asyncio.wait_for(entered.wait(), 5)
            assert registry.active_count("owner") == 2
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert registry.active_count("owner") == 1
            assert registry.active_count("unrelated") == 0
        finally:
            release_worker.set()

    asyncio.run(scenario())
    # asyncio.run waits for the default executor to shut down.
    assert registry.active_count("owner") == 0


@pytest.mark.parametrize("fails", [False, True])
def test_offload_releases_reference_and_preserves_result(fails):
    registry = RequestActivityRegistry()

    def worker(value, *, suffix):
        assert registry.active_count("owner") == 2
        assert registry.active_count("other") == 2
        if fails:
            raise ValueError("worker failed")
        return value + suffix

    async def app(scope, receive, send):
        track_authenticated_request("owner")
        track_authenticated_request("other")
        if fails:
            with pytest.raises(ValueError, match="worker failed"):
                await run_tracked_sync(worker, "a", suffix="b")
        else:
            assert await run_tracked_sync(worker, "a", suffix="b") == "ab"
        assert registry.active_count("owner") == 1
        assert registry.active_count("other") == 1

    asyncio.run(
        RequestActivityMiddleware(app, registry)({"type": "http"}, _receive, _send)
    )
    assert registry.active_count("owner") == 0
    assert registry.active_count("other") == 0


def test_closed_request_cannot_submit_new_sync_work():
    registry = RequestActivityRegistry()
    called = []

    async def scenario():
        proceed = asyncio.Event()
        tasks = []

        async def child():
            await proceed.wait()
            with pytest.raises(RuntimeError, match="already closed"):
                await run_tracked_sync(called.append, "unexpected")

        async def app(scope, receive, send):
            track_authenticated_request("owner")
            tasks.append(asyncio.create_task(child()))

        await RequestActivityMiddleware(app, registry)(
            {"type": "http"}, _receive, _send
        )
        proceed.set()
        await tasks[0]

    asyncio.run(scenario())
    assert called == []
    assert registry.active_count("owner") == 0


def test_unscoped_native_call_preserves_existing_contract():
    assert asyncio.run(run_tracked_sync(lambda value: value + 1, 2)) == 3
