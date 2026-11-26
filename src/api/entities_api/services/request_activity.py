"""Process-local request and offloaded work accounting; not a distributed barrier."""

import asyncio
from collections import Counter
from contextvars import ContextVar, copy_context
from functools import partial
from threading import Lock, Thread
from typing import Any, Callable, Coroutine, TypeVar

from starlette.types import ASGIApp, Receive, Scope, Send

_T = TypeVar("_T")


class RequestActivityRegistry:
    """Count authenticated HTTP requests and retained synchronous operations."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._counts: Counter[str] = Counter()

    def active_count(self, user_id: str) -> int:
        with self._lock:
            return self._counts[user_id]

    def _register(self, activity: "_RequestActivity", user_id: str) -> None:
        with self._lock:
            if activity.closed:
                raise RuntimeError("HTTP request activity has already closed.")
            if user_id not in activity.users:
                activity.users.add(user_id)
                self._counts[user_id] += 1

    def _retain(self, activity: "_RequestActivity") -> Callable[[], None]:
        """Acquire a work reference before submitting to an executor."""
        with self._lock:
            if activity.closed:
                raise RuntimeError("HTTP request activity has already closed.")
            users = tuple(activity.users)
            for user_id in users:
                self._counts[user_id] += 1

        released = False

        def release() -> None:
            nonlocal released
            with self._lock:
                if released:
                    return
                released = True
                for user_id in users:
                    self._counts[user_id] -= 1
                    if self._counts[user_id] == 0:
                        del self._counts[user_id]

        return release

    def _close(self, activity: "_RequestActivity") -> None:
        with self._lock:
            if activity.closed:
                return
            activity.closed = True
            for user_id in activity.users:
                self._counts[user_id] -= 1
                if self._counts[user_id] == 0:
                    del self._counts[user_id]
            activity.users.clear()


class _RequestActivity:
    def __init__(self, registry: RequestActivityRegistry) -> None:
        self.registry = registry
        self.users: set[str] = set()
        self.closed = False


request_activity = RequestActivityRegistry()
_current_request: ContextVar[_RequestActivity | None] = ContextVar(
    "projectdavid_request_activity", default=None
)


def track_authenticated_request(user_id: str) -> bool:
    """Register before the fresh erasure-state read; deduplicate repeated auth.

    False means there is no HTTP middleware scope (e.g. a direct service call).
    This registry does not count native workers, subprocesses or other replicas.
    """
    activity = _current_request.get()
    if activity is None:
        return False
    activity.registry._register(activity, user_id)
    return True


async def run_tracked_sync(func: Callable[..., _T], /, *args, **kwargs) -> _T:
    """Offload synchronous work without losing its activity on cancellation.

    The reference belongs to the submitted callable, not the awaiting task.
    Shielding the executor future prevents caller cancellation from cancelling
    queued work before its finally block can release the reference. The caller
    still receives cancellation promptly. A detached thread stays counted.
    Calls outside an HTTP activity scope retain ordinary to_thread semantics.
    This does not track subprocesses or tasks spawned by the callable itself.
    """
    activity = _current_request.get()
    if activity is None:
        return await asyncio.to_thread(func, *args, **kwargs)

    loop = asyncio.get_running_loop()
    context = copy_context()
    call = partial(func, *args, **kwargs)
    release = activity.registry._retain(activity)

    def invoke():
        try:
            return context.run(call)
        finally:
            release()

    try:
        future = loop.run_in_executor(None, invoke)
    except BaseException:
        release()
        raise

    # A cancelled waiter will no longer retrieve the result. Consume any
    # eventual exception to avoid an unobserved-future warning. Awaiters still
    # receive the original exception normally.
    def observe_result(done):
        if not done.cancelled():
            done.exception()

    future.add_done_callback(observe_result)
    return await asyncio.shield(future)


def create_tracked_task(
    factory: Callable[[], Coroutine[Any, Any, _T]],
) -> asyncio.Task[_T]:
    """Retain request activity until an independently owned task is done.

    The done callback also handles cancellation before the coroutine starts.
    The caller must own the task and arrange its shutdown. Unscoped callers
    keep their existing semantics; this is not a distributed worker registry.
    """
    activity = _current_request.get()
    release = activity.registry._retain(activity) if activity is not None else None
    coroutine = None
    try:
        coroutine = factory()
        task = asyncio.create_task(coroutine)
    except BaseException:
        if coroutine is not None:
            coroutine.close()
        if release is not None:
            release()
        raise

    if release is not None:
        task.add_done_callback(lambda _: release())
    return task


def start_tracked_thread(target: Callable[[], None], *, daemon: bool = True) -> Thread:
    """Retain the current account activity for the full thread target lifetime.

    Like an ordinary Thread, the target has no inherited request ContextVar.
    Its own synchronous work and joined children are covered by the outer
    reference. Targets must manage their own shutdown and must not abandon
    additional detached work.
    """
    activity = _current_request.get()
    release = activity.registry._retain(activity) if activity is not None else None

    def invoke() -> None:
        try:
            target()
        finally:
            if release is not None:
                release()

    try:
        thread = Thread(target=invoke, daemon=daemon)
        thread.start()
    except BaseException:
        if release is not None:
            release()
        raise
    return thread


class RequestActivityMiddleware:
    """Pure ASGI middleware: release only after the downstream call exits."""

    def __init__(
        self, app: ASGIApp, registry: RequestActivityRegistry = request_activity
    ) -> None:
        self.app = app
        self.registry = registry

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        activity = _RequestActivity(self.registry)
        token = _current_request.set(activity)
        try:
            await self.app(scope, receive, send)
        finally:
            self.registry._close(activity)
            _current_request.reset(token)
