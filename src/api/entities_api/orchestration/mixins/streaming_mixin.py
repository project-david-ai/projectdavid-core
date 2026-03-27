"""
All generic streaming helpers shared by every provider:

• start_cancellation_listener — fire-and-forget thread
• _shunt_to_redis_stream      — mirror chunks for other workers
• _process_code_interpreter_chunks — line-wise splitter for ```python``` previews
• stream_function_call_output — injects reminders & SSE proxy
"""

from __future__ import annotations

import asyncio
import json
from threading import Event, Thread
from typing import Any, Callable, Generator, Optional

import redis as redis_py
from projectdavid_common.validation import StatusEnum

from src.api.entities_api.constants.assistant import (
    CODE_INTERPRETER_MESSAGE,
    DEFAULT_REMINDER_MESSAGE,
)
from src.api.entities_api.services.logging_service import LoggingUtility

LOG = LoggingUtility()


class StreamingMixin:
    """
    Shared streaming infrastructure used by all provider workers.

    Owns:
        start_cancellation_monitor() — spawns a daemon thread that polls
                                       run status and sets a stop Event
                                       when the run is cancelled or terminal
        check_cancellation_flag()    — returns self._cancelled (legacy sync flag)
        _shunt_to_redis_stream()     — async mirror of a chunk to a Redis Stream
                                       key with TTL management
        _process_code_interpreter_chunks() — line-wise buffer splitter for
                                             hot_code preview streaming
        stream_function_call_output() — sync generator that injects a reminder
                                        message, proxies the provider stream,
                                        and fire-and-forgets Redis shunting

    Requires on self:
        self.redis                   — sync redis.Redis instance, set in
                                       OrchestratorCore.__init__ before first use
        self._native_exec            — NativeExecMixin, used by cancellation
                                       monitor to poll run status
        self.project_david_client    — ServiceRegistryMixin, used by
                                       stream_function_call_output to inject
                                       reminder messages
        self.finalize_conversation() — ConsumerToolHandlersMixin, called at
                                       end of stream_function_call_output

    Contract:
        StreamingMixin is intentionally excluded from _ProviderMixins and
        composed directly on OrchestratorCore. This is deliberate — streaming
        infrastructure is orchestrator-level, not provider-level. Do not add
        StreamingMixin to _ProviderMixins. The cancellation monitor calls
        asyncio.run() inside a daemon thread — it must never be called from
        within an already-running event loop context.
    """

    redis: redis_py.Redis
    _cancelled: bool = False
    project_david_client: Any  # Type hint for IDE support

    def start_cancellation_monitor(self, run_id: str, interval: float = 1.0) -> Event:
        stop_event = Event()

        def monitor():
            terminal_statuses = {
                StatusEnum.completed.value,
                StatusEnum.failed.value,
                StatusEnum.cancelled.value,
                StatusEnum.expired.value,
                StatusEnum.pending_action.value,
                StatusEnum.deleted.value,
            }

            while not stop_event.is_set():
                try:
                    # ✅ Native DB call — no HTTP SDK
                    run = asyncio.run(self._native_exec.retrieve_run(run_id))

                    if run.status == StatusEnum.cancelled.value:
                        LOG.warning("Run %s was cancelled via API.", run_id)
                        stop_event.set()
                        break

                    if run.status in terminal_statuses:
                        LOG.debug(
                            "Run %s reached terminal status (%s). Stopping monitor.",
                            run_id,
                            run.status,
                        )
                        break

                except Exception as e:
                    LOG.warning("Cancellation monitor error for %s: %s", run_id, e)

                stop_event.wait(timeout=interval)

            LOG.debug("Cancellation monitor thread for %s has exited.", run_id)

        Thread(target=monitor, daemon=True).start()
        return stop_event

    def check_cancellation_flag(self) -> bool:
        return self._cancelled

    async def _shunt_to_redis_stream(
        self, redis, stream_key, chunk_dict, *, maxlen=1000, ttl_seconds=3600
    ):
        """
        Mirrors the chunk to Redis Stream.
        Fix: Serializes nested dicts/lists to JSON strings to prevent DataError.
        """
        try:
            if not callable(getattr(redis, "xadd", None)):
                LOG.debug("[Redis Shunt] async or stub Redis – skipping XADD")
                return

            # Ensure input is a dictionary
            if isinstance(chunk_dict, str):
                try:
                    chunk_dict = json.loads(chunk_dict)
                except json.JSONDecodeError:
                    chunk_dict = {"content": chunk_dict}

            # FIX: Redis XADD requires flat values (str, bytes, int, float).
            # We must serialize nested dicts/lists into JSON strings.
            redis_safe_chunk = {}
            for k, v in chunk_dict.items():
                if isinstance(v, (dict, list)):
                    redis_safe_chunk[k] = json.dumps(v)
                elif v is None:
                    redis_safe_chunk[k] = ""
                elif isinstance(v, bool):
                    redis_safe_chunk[k] = str(v)
                else:
                    redis_safe_chunk[k] = v

            loop = asyncio.get_running_loop()

            # Run sync redis ops in executor so we don't block event loop
            await loop.run_in_executor(
                None,
                lambda: redis.xadd(
                    stream_key, redis_safe_chunk, maxlen=maxlen, approximate=True
                ),
            )

            # Handle TTL
            ttl_flag = f"{stream_key}::ttl_set"
            exists = await loop.run_in_executor(None, lambda: redis.exists(ttl_flag))

            if not exists:
                await loop.run_in_executor(
                    None, lambda: redis.expire(stream_key, ttl_seconds)
                )
                await loop.run_in_executor(
                    None, lambda: redis.set(ttl_flag, "1", ex=ttl_seconds)
                )

        except Exception as exc:
            LOG.warning(
                "[Redis Shunt] failed (%s): %s",
                type(exc).__name__,
                exc,
                exc_info=True,
            )

    def _process_code_interpreter_chunks(self, content_chunk, code_buffer):
        """
        Process code chunks while in code mode.

        Appends the incoming content_chunk to code_buffer,
        then extracts a single line (if a newline exists) and handles buffer overflow.

        Returns:
            tuple: (results, updated code_buffer)
                - results: list of JSON strings representing code chunks.
                - updated code_buffer: the remaining buffer content.
        """
        self.code_mode = True
        results = []
        code_buffer += content_chunk
        if "\n" in code_buffer:
            newline_pos = code_buffer.find("\n") + 1
            line_chunk = code_buffer[:newline_pos]
            code_buffer = code_buffer[newline_pos:]
            results.append(json.dumps({"type": "hot_code", "content": line_chunk}))
        if len(code_buffer) > 100:
            results.append(json.dumps({"type": "hot_code", "content": code_buffer}))
            code_buffer = ""
        return (results, code_buffer)

    def stream_function_call_output(
        self,
        thread_id: str,
        run_id: str,
        assistant_id: str,
        model: str,
        *,
        stream: Callable[..., Generator[str, None, None]],
        name: Optional[str] = None,
        stream_reasoning: bool = False,
        api_key: Optional[str] = None,
    ):
        """
        Injects a short reminder (“You are now executing code …”) so the
        assistant does not hallucinate, then transparently proxies the
        underlying provider stream to the client **and** Redis.
        """
        reminder = (
            CODE_INTERPRETER_MESSAGE
            if name == "code_interpreter"
            else DEFAULT_REMINDER_MESSAGE
        )
        self.project_david_client.messages.create_message(
            thread_id=thread_id,
            assistant_id=assistant_id,
            content=reminder,
            role="user",
        )
        gen = stream(
            thread_id=thread_id,
            message_id=None,
            run_id=run_id,
            assistant_id=assistant_id,
            model=model,
            stream_reasoning=True,
            api_key=api_key,
        )
        redis_key = f"stream:{run_id}"
        assistant_reply = ""
        reasoning = ""

        # Get the current event loop to schedule the async redis shunt
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        for raw in gen:
            try:
                parsed = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                parsed = {"type": "content", "content": str(raw)}

            t = parsed.get("type")
            c = parsed.get("content", "")

            if t == "reasoning":
                reasoning += c
            elif t == "content":
                assistant_reply += c

            yield raw

            # Fire-and-forget the Redis shunt
            if loop and loop.is_running():
                loop.create_task(
                    self._shunt_to_redis_stream(self.redis, redis_key, parsed)
                )

        if assistant_reply:
            self.finalize_conversation(
                reasoning + assistant_reply, thread_id, assistant_id, run_id
            )
