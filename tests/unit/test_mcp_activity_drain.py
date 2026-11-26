"""A run-affine MCP session remains active until its task has actually exited."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from src.api.entities_api.orchestration.mcp_run_session_manager import (
    McpRunSessionManager,
)
from src.api.entities_api.services.request_activity import (
    RequestActivityMiddleware,
    RequestActivityRegistry,
    create_tracked_task,
    track_authenticated_request,
)


async def _receive():
    return {"type": "http.disconnect"}


async def _send(message):
    pass


async def _request(registry, app):
    await RequestActivityMiddleware(app, registry)({"type": "http"}, _receive, _send)


class Executor:
    server_id = "server"

    def __init__(self, *, fail=False, hold_invocation=False, hold_teardown=False):
        self.fail = fail
        self.hold_invocation = hold_invocation
        self.hold_teardown = hold_teardown
        self.invoking = asyncio.Event()
        self.finish_invocation = asyncio.Event()
        self.tearing_down = asyncio.Event()
        self.finish_teardown = asyncio.Event()
        self.closed = False

    @asynccontextmanager
    async def client_context(self):
        try:
            yield object()
        finally:
            self.tearing_down.set()
            if self.hold_teardown:
                await self.finish_teardown.wait()
            self.closed = True

    async def execute_connected(self, client, call, *, progress_callback):
        self.invoking.set()
        if self.hold_invocation:
            await self.finish_invocation.wait()
        if self.fail:
            raise ValueError("tool failed")
        return "result"

    def failure_result(self, exc):
        return str(exc)


async def _execute(manager, executor):
    return await manager.execute(
        owner_scope="owner",
        executor=executor,
        call=SimpleNamespace(run_id="run"),
    )


def test_mcp_activity_survives_request_and_sdk_continuation():
    registry = RequestActivityRegistry()

    async def scenario():
        manager = McpRunSessionManager()
        executor = Executor()

        async def app(scope, receive, send):
            track_authenticated_request("owner")
            assert await _execute(manager, executor) == "result"
            assert registry.active_count("owner") == 2

        try:
            await _request(registry, app)
            assert registry.active_count("owner") == 1
            await _request(registry, app)
            assert registry.active_count("owner") == 1
            assert registry.active_count("unrelated") == 0
        finally:
            await manager.close_run("run", cancel=True)
        assert executor.closed
        assert registry.active_count("owner") == 0

    asyncio.run(scenario())


def test_cancelled_invocation_stays_counted_until_session_closes():
    registry = RequestActivityRegistry()

    async def scenario():
        manager = McpRunSessionManager()
        executor = Executor(hold_invocation=True)

        async def app(scope, receive, send):
            track_authenticated_request("owner")
            await _execute(manager, executor)

        request = asyncio.create_task(_request(registry, app))
        try:
            await asyncio.wait_for(executor.invoking.wait(), 3)
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
            assert registry.active_count("owner") == 1
            assert not executor.closed
        finally:
            await manager.close_all(cancel=True)
        assert executor.closed
        assert registry.active_count("owner") == 0

    asyncio.run(scenario())


def test_interrupted_close_retains_live_actor_for_retry():
    registry = RequestActivityRegistry()

    async def scenario():
        manager = McpRunSessionManager()
        executor = Executor(hold_teardown=True)

        async def app(scope, receive, send):
            track_authenticated_request("owner")
            await _execute(manager, executor)

        await _request(registry, app)
        closing = asyncio.create_task(manager.close_run("run", cancel=True))
        try:
            await asyncio.wait_for(executor.tearing_down.wait(), 3)
            closing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await closing
            assert registry.active_count("owner") == 1
            assert not executor.closed
            assert len(manager._actors) == 1
        finally:
            executor.finish_teardown.set()
            await manager.close_all(cancel=True)
        assert executor.closed
        assert registry.active_count("owner") == 0
        assert not manager._actors

    asyncio.run(scenario())


def test_failed_invocation_closes_actor_before_returning_failure():
    registry = RequestActivityRegistry()

    async def scenario():
        manager = McpRunSessionManager()
        executor = Executor(fail=True)

        async def app(scope, receive, send):
            track_authenticated_request("owner")
            assert await _execute(manager, executor) == "tool failed"
            assert executor.closed
            assert registry.active_count("owner") == 1
            assert not manager._actors

        await _request(registry, app)
        assert registry.active_count("owner") == 0

    asyncio.run(scenario())


def test_tracked_task_cancelled_before_start_releases_reference():
    registry = RequestActivityRegistry()

    async def work():
        pytest.fail("Task should not start")

    async def app(scope, receive, send):
        track_authenticated_request("owner")
        task = create_tracked_task(work)
        assert registry.active_count("owner") == 2
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert registry.active_count("owner") == 1

    asyncio.run(_request(registry, app))
    assert registry.active_count("owner") == 0
