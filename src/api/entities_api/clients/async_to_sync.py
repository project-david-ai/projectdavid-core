import asyncio
import queue
import threading
from typing import AsyncGenerator, Generator, TypeVar

from src.api.entities_api.services.request_activity import start_tracked_thread

T = TypeVar("T")


def async_to_sync_stream(agen: AsyncGenerator[T, None]) -> Generator[T, None, None]:
    """Bridge an async stream to a bounded synchronous iterator.

    Closing the consumer requests cancellation of the producer. Activity is
    retained until the producer's event loop and generator have shut down.
    """
    q: queue.Queue[T] = queue.Queue(maxsize=100)
    stopped = threading.Event()
    done = threading.Event()
    state_lock = threading.Lock()
    active_loop = None
    active_task = None
    failure: BaseException | None = None

    async def consume_stream():
        if stopped.is_set():
            return
        async for item in agen:
            # Bounded waits let early consumer closure break backpressure.
            while not stopped.is_set():
                try:
                    q.put(item, timeout=0.05)
                    break
                except queue.Full:
                    continue
            if stopped.is_set():
                break

    async def drive():
        nonlocal active_loop, active_task
        task = asyncio.create_task(consume_stream())
        with state_lock:
            active_loop = asyncio.get_running_loop()
            active_task = task
        try:
            if stopped.is_set():
                task.cancel()
            await task
        finally:
            # Cleanup belongs to the driver, not the cancelled consumer task.
            await agen.aclose()

    def producer():
        nonlocal failure, active_loop, active_task
        try:
            asyncio.run(drive())
        except asyncio.CancelledError as exc:
            if not stopped.is_set():
                failure = exc
        except BaseException as exc:
            failure = exc
        finally:
            with state_lock:
                active_loop = None
                active_task = None
            done.set()

    thread = start_tracked_thread(producer)
    try:
        while True:
            # The completion flag is set only after all queued items are written.
            if done.is_set():
                try:
                    item = q.get_nowait()
                except queue.Empty:
                    if failure is not None:
                        raise failure
                    break
            else:
                try:
                    item = q.get(timeout=0.05)
                except queue.Empty:
                    continue
            yield item
    finally:
        stopped.set()
        with state_lock:
            if active_loop is not None and active_task is not None:
                try:
                    active_loop.call_soon_threadsafe(active_task.cancel)
                except RuntimeError:
                    # The producer loop has already shut down.
                    pass
        # No blocking join on early close: the independent activity reference
        # remains held until the producer finishes. On natural exhaustion the
        # thread is already exiting and joining is safe.
        if done.is_set():
            thread.join()
