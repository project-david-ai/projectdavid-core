"""HTTP activity accounting must survive response streaming and failures."""

import asyncio

import pytest

from src.api.entities_api.services.request_activity import (
    RequestActivityMiddleware,
    RequestActivityRegistry,
    track_authenticated_request,
)


async def _receive():
    return {"type": "http.disconnect"}


async def _send(message):
    pass


def test_registration_deduplicates_and_isolates_accounts():
    registry = RequestActivityRegistry()

    async def app(scope, receive, send):
        assert track_authenticated_request("owner")
        assert track_authenticated_request("owner")
        assert registry.active_count("owner") == 1
        assert registry.active_count("unrelated") == 0
        assert track_authenticated_request("unrelated")
        assert registry.active_count("unrelated") == 1

    asyncio.run(
        RequestActivityMiddleware(app, registry)({"type": "http"}, _receive, _send)
    )
    assert registry.active_count("owner") == 0
    assert registry.active_count("unrelated") == 0
    assert not track_authenticated_request("owner")


def test_counts_concurrent_requests_until_each_exits():
    registry = RequestActivityRegistry()

    async def scenario():
        entered = [asyncio.Event(), asyncio.Event()]
        release = [asyncio.Event(), asyncio.Event()]

        async def app(scope, receive, send):
            index = scope["index"]
            track_authenticated_request("owner")
            entered[index].set()
            await release[index].wait()

        middleware = RequestActivityMiddleware(app, registry)
        tasks = [
            asyncio.create_task(
                middleware({"type": "http", "index": i}, _receive, _send)
            )
            for i in range(2)
        ]
        await asyncio.gather(*(event.wait() for event in entered))
        assert registry.active_count("owner") == 2
        release[0].set()
        await tasks[0]
        assert registry.active_count("owner") == 1
        release[1].set()
        await tasks[1]
        assert registry.active_count("owner") == 0

    asyncio.run(scenario())


def test_stream_remains_counted_after_response_headers():
    registry = RequestActivityRegistry()
    observed = []

    async def send(message):
        observed.append((message["type"], registry.active_count("owner")))

    async def app(scope, receive, send):
        track_authenticated_request("owner")
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"chunk", "more_body": True})
        await send({"type": "http.response.body", "body": b"", "more_body": False})
        assert registry.active_count("owner") == 1

    asyncio.run(
        RequestActivityMiddleware(app, registry)({"type": "http"}, _receive, send)
    )
    assert observed == [
        ("http.response.start", 1),
        ("http.response.body", 1),
        ("http.response.body", 1),
    ]
    assert registry.active_count("owner") == 0


@pytest.mark.parametrize("error", [RuntimeError, asyncio.CancelledError])
def test_release_on_exception_or_cancellation(error):
    registry = RequestActivityRegistry()

    async def app(scope, receive, send):
        track_authenticated_request("owner")
        raise error()

    with pytest.raises(error):
        asyncio.run(
            RequestActivityMiddleware(app, registry)({"type": "http"}, _receive, _send)
        )
    assert registry.active_count("owner") == 0


def test_inherited_closed_context_cannot_register_new_work():
    registry = RequestActivityRegistry()

    async def scenario():
        release = asyncio.Event()
        tasks = []

        async def child():
            await release.wait()
            with pytest.raises(RuntimeError, match="already closed"):
                track_authenticated_request("owner")

        async def app(scope, receive, send):
            track_authenticated_request("owner")
            tasks.append(asyncio.create_task(child()))

        await RequestActivityMiddleware(app, registry)(
            {"type": "http"}, _receive, _send
        )
        assert registry.active_count("owner") == 0
        release.set()
        await tasks[0]

    asyncio.run(scenario())


def test_non_http_scope_does_not_register():
    registry = RequestActivityRegistry()

    async def app(scope, receive, send):
        assert not track_authenticated_request("owner")

    asyncio.run(
        RequestActivityMiddleware(app, registry)({"type": "lifespan"}, _receive, _send)
    )
    assert registry.active_count("owner") == 0
