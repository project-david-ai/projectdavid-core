"""Run-affine MCP connection ownership.

Stateful MCP servers such as Playwright require multiple tools/call requests to
share one live MCP protocol session.  Project David therefore keeps one live
remote MCP connection per tenant/run/registration execution scope.

The connection is owned by a dedicated asyncio task.  This is intentional:
async context managers used by the MCP SDK may own AnyIO task groups/cancel
scopes and must be entered and exited by the same task.
"""

from __future__ import annotations

import asyncio
import threading
import weakref
from dataclasses import dataclass
from typing import Any, AsyncContextManager

from src.api.entities_api.services.request_activity import create_tracked_task

from .mcp_remote_client import RemoteMcpClient
from .mcp_tool_executor import McpToolExecutor
from .tool_abi import ToolCallEnvelope, ToolResultEnvelope


@dataclass(frozen=True)
class McpRunSessionKey:
    owner_scope: str
    run_id: str
    server_id: str


@dataclass
class _Invocation:
    executor: McpToolExecutor
    call: ToolCallEnvelope
    progress_callback: Any | None
    future: asyncio.Future[ToolResultEnvelope]


class _McpSessionActor:
    """Own exactly one live RemoteMcpClient context."""

    def __init__(
        self,
        client_context: AsyncContextManager[RemoteMcpClient],
    ) -> None:
        self._client_context = client_context
        self._queue: asyncio.Queue[_Invocation | None] = asyncio.Queue()
        self._closing = False
        self._task = create_tracked_task(self._serve)
        self._task.add_done_callback(self._on_done)

    def _on_done(self, task: asyncio.Task[None]) -> None:
        # Cancellation before the first coroutine step skips _serve entirely.
        self._cancel_pending()

    @property
    def task(self) -> asyncio.Task[None]:
        return self._task

    async def _serve(self) -> None:
        try:
            async with self._client_context as client:
                while True:
                    invocation = await self._queue.get()

                    if invocation is None:
                        return

                    if invocation.future.cancelled():
                        continue

                    try:
                        result = await invocation.executor.execute_connected(
                            client,
                            invocation.call,
                            progress_callback=invocation.progress_callback,
                        )
                    except asyncio.CancelledError:
                        if not invocation.future.done():
                            invocation.future.cancel()
                        raise
                    except Exception as exc:  # noqa: BLE001
                        if not invocation.future.done():
                            invocation.future.set_exception(exc)
                    else:
                        if not invocation.future.done():
                            invocation.future.set_result(result)

        except asyncio.CancelledError:
            self._cancel_pending()
            raise

        except Exception as exc:  # noqa: BLE001
            self._fail_pending(exc)

    def _cancel_pending(self) -> None:
        while True:
            try:
                invocation = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return

            if invocation is not None and not invocation.future.done():
                invocation.future.cancel()

    def _fail_pending(self, exc: Exception) -> None:
        while True:
            try:
                invocation = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return

            if invocation is not None and not invocation.future.done():
                invocation.future.set_exception(exc)

    async def execute(
        self,
        *,
        executor: McpToolExecutor,
        call: ToolCallEnvelope,
        progress_callback: Any | None,
    ) -> ToolResultEnvelope:
        if self._closing or self._task.done():
            raise RuntimeError("MCP run session is closing or terminated")

        future: asyncio.Future[ToolResultEnvelope] = (
            asyncio.get_running_loop().create_future()
        )

        # Queue immediately: the actor opens its connection before consuming
        # invocations. Startup failure is delivered through _fail_pending.
        # No separate readiness future or extra scheduling handoff is needed.
        self._queue.put_nowait(
            _Invocation(
                executor=executor,
                call=call,
                progress_callback=progress_callback,
                future=future,
            )
        )

        return await future

    async def close(self, *, cancel: bool = False) -> None:
        if not self._task.done():
            if not self._closing:
                self._closing = True
                if cancel:
                    self._task.cancel()
                else:
                    self._queue.put_nowait(None)
            # A second close waits for the same shutdown, without injecting
            # another cancellation into an in-progress context teardown.
        try:
            await asyncio.shield(self._task)
        except asyncio.CancelledError:
            if not self._task.cancelled():
                raise


class McpRunSessionManager:
    """Own live MCP connections at tenant/run/registration scope."""

    def __init__(self) -> None:
        self._actors: dict[McpRunSessionKey, _McpSessionActor] = {}
        self._registry_lock = asyncio.Lock()

    async def _actor_for(
        self,
        *,
        key: McpRunSessionKey,
        executor: McpToolExecutor,
    ) -> _McpSessionActor:
        async with self._registry_lock:
            actor = self._actors.get(key)

            if actor is not None and not actor.task.done():
                return actor

            actor = _McpSessionActor(executor.client_context())
            self._actors[key] = actor
            return actor

    async def execute(
        self,
        *,
        owner_scope: str,
        executor: McpToolExecutor,
        call: ToolCallEnvelope,
        progress_callback: Any | None = None,
    ) -> ToolResultEnvelope:
        key = McpRunSessionKey(
            owner_scope=owner_scope,
            run_id=call.run_id,
            server_id=executor.server_id,
        )

        actor = await self._actor_for(
            key=key,
            executor=executor,
        )

        try:
            return await actor.execute(
                executor=executor,
                call=call,
                progress_callback=progress_callback,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            # A failed invocation must not orphan a still-live session.
            await self._close_selected([(key, actor)], cancel=True)
            return executor.failure_result(exc)

    async def _close_selected(
        self,
        selected: list[tuple[McpRunSessionKey, _McpSessionActor]],
        *,
        cancel: bool,
    ) -> None:
        for key, actor in selected:
            try:
                await actor.close(cancel=cancel)
            finally:
                # Preserve live actors so another close can retry. Never remove
                # a replacement actor created after this one terminated.
                async with self._registry_lock:
                    if actor.task.done() and self._actors.get(key) is actor:
                        self._actors.pop(key, None)

    async def close_run(
        self,
        run_id: str,
        *,
        cancel: bool = False,
    ) -> None:
        async with self._registry_lock:
            selected = [
                (key, actor)
                for key, actor in self._actors.items()
                if key.run_id == run_id
            ]
        await self._close_selected(selected, cancel=cancel)

    async def close_all(
        self,
        *,
        cancel: bool = False,
    ) -> None:
        async with self._registry_lock:
            selected = list(self._actors.items())
        await self._close_selected(selected, cancel=cancel)


_RUNTIME_MANAGERS_LOCK = threading.RLock()

# One manager per live asyncio event loop.
#
# This is deliberately NOT one manager per provider/orchestrator instance:
# inference continuation may construct a new handler for the same Project
# David Run. The session itself remains isolated by owner/run/registration
# inside McpRunSessionManager.
_RUNTIME_MANAGERS: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    McpRunSessionManager,
] = weakref.WeakKeyDictionary()


def get_mcp_run_session_manager() -> McpRunSessionManager:
    """Return the MCP runtime manager owned by the current application loop."""

    loop = asyncio.get_running_loop()

    with _RUNTIME_MANAGERS_LOCK:
        manager = _RUNTIME_MANAGERS.get(loop)

        if manager is None:
            manager = McpRunSessionManager()
            _RUNTIME_MANAGERS[loop] = manager

        return manager


__all__ = [
    "McpRunSessionKey",
    "McpRunSessionManager",
    "get_mcp_run_session_manager",
]
