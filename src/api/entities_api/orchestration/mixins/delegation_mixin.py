from __future__ import annotations

import asyncio
import json
import os
import threading
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Callable, Dict, Optional

from projectdavid.events import ScratchpadEvent
from projectdavid_common.utilities.logging_service import LoggingUtility
from projectdavid_common.validation import StatusEnum

from src.api.entities_api.services.native_execution_service import \
    NativeExecutionService
from src.api.entities_api.utils.assistant_manager import AssistantManager

LOG = LoggingUtility()

_TERMINAL_RUN_STATES = {"completed", "failed", "cancelled", "expired"}
_WORKER_RUN_TIMEOUT = 1200
_WORKER_POLL_INTERVAL = 2.0


class DelegationMixin:
    """
    Ephemeral worker lifecycle management for multi-agent delegation flows.
    Handles research delegation and engineer delegation via spawned ephemeral
    assistants running on isolated threads and runs.

    Owns:
        __init__()                        — cooperative, calls super().__init__().
                                            Only mixin in the stack with __init__
                                            participation. Initialises delegation
                                            state attributes.
        _stream_sync_generator()          — bridges a blocking sync generator into
                                            an async generator via a daemon thread
                                            and asyncio.Queue
        _wait_for_run_completion()        — async poller with configurable timeout
        _ephemeral_clean_up()             — deletes ephemeral assistant and
                                            optionally the ephemeral thread
        _capture_tool_outputs()           — async context manager that intercepts
                                            submit_tool_output calls into a dict
        create_ephemeral_worker_assistant()
        create_ephemeral_junior_engineer()
        create_ephemeral_thread()
        create_ephemeral_message()
        create_ephemeral_run()            — factory helpers, all require
                                            self._run_user_id to be set first
        _fetch_worker_final_report()      — fallback message retrieval, retained
                                            for other components
        handle_delegate_research_task()   — async generator, full research
                                            delegation lifecycle
        handle_delegate_engineer_task()   — async generator (implied by symmetry)

    Requires on self:
        self._native_exec                 — NativeExecMixin
        self._run_user_id                 — set in QwenBaseWorker.stream() before
                                            delegation is triggered
        self.submit_tool_output()         — ConsumerToolHandlersMixin
        self.project_david_client         — ServiceRegistryMixin, used for
                                            synchronous_inference_stream
        self._assistant_manager           — lazy property, self-initialising
                                            via AssistantManager()

    Instance state (set in __init__):
        self._delete_ephemeral_thread     — bool, controls thread cleanup
        self._delegation_model            — model string for delegated inference
        self._research_worker_thread      — ephemeral thread id for research flow
        self._run_user_id                 — owner resolved from run at stream time

    Contract:
        DelegationMixin is the only mixin in _ProviderMixins that calls
        super().__init__(). All other mixins are stateless at construction.
        Do not add __init__ to other mixins without updating the _ProviderMixins
        and OrchestratorCore contract comments.
        All factory methods guard on self._run_user_id via getattr and raise
        RuntimeError explicitly — this is the correct pattern for cross-mixin
        state access and should be followed in any new factory methods added here.
        The api_key scrub at the end of handle_delegate_research_task()
        (meta_data={"api_key": "***"}) is a security requirement — do not remove.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._delete_ephemeral_thread = False
        self._delegation_model = None
        self._research_worker_thread = None
        self._scratch_pad_thread = None
        self._run_user_id = None
        self._batfish_owner_user_id = None
        self._native_exec_svc: Optional[NativeExecutionService] = None

    @property
    def _assistant_manager(self) -> AssistantManager:
        if getattr(self, "_assistant_manager_svc", None) is None:
            self._assistant_manager_svc = AssistantManager()
        return self._assistant_manager_svc

    # ------------------------------------------------------------------
    # EMISSION HELPER
    # ------------------------------------------------------------------

    def _research_status(self, activity: str, state: str, run_id: str) -> str:
        return json.dumps(
            {
                "type": "research_status",
                "activity": activity,
                "state": state,
                "tool": "delegate_research_task",
                "run_id": run_id,
            }
        )

    # ------------------------------------------------------------------
    # HELPER: Bridges blocking generators to async loop
    # ------------------------------------------------------------------

    async def _stream_sync_generator(
        self, generator_func: Callable, *args, **kwargs
    ) -> AsyncGenerator[Any, None]:
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def producer():
            try:
                for item in generator_func(*args, **kwargs):
                    loop.call_soon_threadsafe(queue.put_nowait, item)
                loop.call_soon_threadsafe(queue.put_nowait, None)
            except Exception as e:
                LOG.error(f"🧵[THREAD-ERR] {e}")
                loop.call_soon_threadsafe(queue.put_nowait, e)

        threading.Thread(target=producer, daemon=True).start()

        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    # ------------------------------------------------------------------
    # HELPER: Poll run status until terminal (Retained for other tasks)
    # ------------------------------------------------------------------

    async def _wait_for_run_completion(
        self,
        run_id: str,
        thread_id: str,
        timeout: float = _WORKER_RUN_TIMEOUT,
        poll_interval: float = _WORKER_POLL_INTERVAL,
    ) -> str:
        LOG.info(
            "⏳ [DELEGATE] Waiting for worker run %s to complete (timeout=%ss)...",
            run_id,
            timeout,
        )
        elapsed = 0.0
        while elapsed < timeout:
            try:
                run = await self._native_exec.retrieve_run(run_id)
                status_value = run.status.value if hasattr(run.status, "value") else str(run.status)
                LOG.critical(
                    "██████ [DELEGATE_POLL] run_id=%s status=%s elapsed=%.1fs ██████",
                    run_id,
                    status_value,
                    elapsed,
                )
                if status_value in _TERMINAL_RUN_STATES:
                    LOG.critical(
                        "██████ [DELEGATE_POLL] run_id=%s reached terminal state=%s ██████",
                        run_id,
                        status_value,
                    )
                    return status_value
            except Exception as e:
                LOG.warning("⚠️[DELEGATE_POLL] Error polling run %s: %s", run_id, e)
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
        LOG.error("❌[DELEGATE_POLL] run_id=%s timed out after %ss.", run_id, timeout)
        raise asyncio.TimeoutError(f"Worker run {run_id} did not complete within {timeout}s")

    # ------------------------------------------------------------------
    # HELPER: Lifecycle cleanup
    # ------------------------------------------------------------------
    async def _ephemeral_clean_up(
        self, assistant_id: str, thread_id: Optional[str], delete_thread: bool = False
    ):
        LOG.info(f"🧹[CLEANUP] Assistant: {assistant_id} | Thread: {thread_id}")

        user_id = getattr(self, "_batfish_owner_user_id", None)

        if delete_thread and thread_id:
            try:
                if not user_id:
                    LOG.warning(
                        "⚠️ [CLEANUP] Cannot delete thread %s — user_id not resolved.", thread_id
                    )
                else:
                    await self._native_exec.delete_thread(thread_id, user_id=user_id)
            except Exception as e:
                LOG.warning(f"⚠️[CLEANUP] Thread delete failed: {e}")

        if user_id and assistant_id:
            try:
                await self._assistant_manager.delete_assistant(
                    assistant_id=assistant_id, user_id=user_id, permanent=True
                )
            except Exception as e:
                LOG.warning(f"⚠️ [CLEANUP] Assistant delete failed: {e}")

    @asynccontextmanager
    async def _capture_tool_outputs(self, capture_dict: Dict[str, str]):
        original = self.submit_tool_output

        async def intercept(
            thread_id,
            assistant_id,
            tool_call_id,
            content,
            action=None,
            is_error=False,
            **kwargs,
        ):
            capture_dict[tool_call_id] = str(content)
            await original(
                thread_id,
                assistant_id,
                tool_call_id,
                content,
                action,
                is_error,
                **kwargs,
            )

        self.submit_tool_output = intercept
        try:
            yield
        finally:
            self.submit_tool_output = original

    # ------------------------------------------------------------------
    # EPHEMERAL FACTORIES
    # ------------------------------------------------------------------
    async def create_ephemeral_worker_assistant(self):
        user_id = getattr(self, "_batfish_owner_user_id", None)
        if not user_id:
            raise RuntimeError(
                "create_ephemeral_worker_assistant: _batfish_owner_user_id has not been "
                "resolved yet — ensure it is set before calling this method."
            )
        return await self._assistant_manager.create_ephemeral_worker_assistant(user_id=user_id)

    async def create_ephemeral_junior_engineer(self):
        user_id = getattr(self, "_batfish_owner_user_id", None)
        if not user_id:
            raise RuntimeError(
                "create_ephemeral_junior_engineer: _batfish_owner_user_id has not been "
                "resolved yet — ensure it is set before calling this method."
            )
        return await self._assistant_manager.create_ephemeral_junior_engineer(user_id=user_id)

    async def create_ephemeral_thread(self):
        user_id = getattr(self, "_batfish_owner_user_id", None)
        if not user_id:
            raise RuntimeError(
                "create_ephemeral_thread: _batfish_owner_user_id has not been "
                "resolved yet — ensure it is set before calling this method."
            )
        return await self._native_exec.create_thread(user_id=user_id)

    async def create_ephemeral_message(self, thread_id, content, assistant_id):
        return await self._native_exec.create_message(
            thread_id=thread_id,
            assistant_id=assistant_id,
            content=content,
        )

    async def create_ephemeral_run(self, assistant_id, thread_id, meta_data: Dict | None = None):
        user_id = getattr(self, "_batfish_owner_user_id", None)
        if not user_id:
            raise RuntimeError(
                "create_ephemeral_run: _batfish_owner_user_id has not been "
                "resolved yet — ensure it is set before calling this method."
            )
        return await self._native_exec.create_run(
            assistant_id=assistant_id,
            thread_id=thread_id,
            user_id=user_id,
            meta_data=meta_data,
        )

    async def _fetch_worker_final_report(
        self,
        thread_id: str,
        max_attempts: int = 5,
        retry_delay: float = 3.0,
    ) -> str | None:
        """
        Retained as a fallback helper for other components if needed.
        Delegation handler now bypasses this completely in favor of stream capture.
        """
        for attempt in range(1, max_attempts + 1):
            try:
                messages = await self._native_exec.get_formatted_messages(thread_id)
                if not messages:
                    continue

                for idx, msg in enumerate(reversed(messages)):
                    role = msg.get("role")
                    content = msg.get("content")
                    tool_calls = msg.get("tool_calls")

                    if role != "assistant" or tool_calls:
                        continue
                    if not isinstance(content, str) or not content.strip():
                        continue

                    stripped = content.strip()
                    if stripped.startswith("[") and stripped.endswith("]"):
                        continue

                    return stripped
            except Exception as e:
                LOG.exception("❌ [WORKER_FETCH] Error: %s", e)
            if attempt < max_attempts:
                await asyncio.sleep(retry_delay)

        return None

    # ------------------------------------------------------------------
    # HANDLER 1: Research Delegation
    # ------------------------------------------------------------------
    async def handle_delegate_research_task(
        self, thread_id, run_id, assistant_id, arguments_dict, tool_call_id, decision
    ) -> AsyncGenerator[str, None]:

        self._scratch_pad_thread = thread_id
        LOG.info(f"🔄[DELEGATE] STARTING. Run: {run_id}")

        if isinstance(arguments_dict, str):
            try:
                args = json.loads(arguments_dict)
            except Exception:
                args = {"task": arguments_dict}
        else:
            args = arguments_dict

        yield self._research_status("Initializing delegation worker...", "in_progress", run_id)

        action = None
        try:
            action = await self._native_exec.create_action(
                tool_name="delegate_research_task",
                run_id=run_id,
                tool_call_id=tool_call_id,
                function_args=arguments_dict,
                decision=decision,
            )
        except Exception as e:
            LOG.error(f"❌[DELEGATE] Action creation failed: {e}")

        ephemeral_worker = None
        ephemeral_thread = None
        execution_had_error = False
        ephemeral_run = None

        try:
            origin_user_id = getattr(self, "_batfish_owner_user_id", None)

            if not origin_user_id:
                run_obj = await self._native_exec.retrieve_run(run_id)
                origin_user_id = run_obj.user_id
                self._batfish_owner_user_id = origin_user_id

            LOG.info(
                "RESEARCH_DELEGATE ▸ origin_user_id=%s | scratch_pad_thread=%s",
                origin_user_id,
                self._scratch_pad_thread,
            )

            ephemeral_worker = await self.create_ephemeral_worker_assistant()
            ephemeral_thread = await self.create_ephemeral_thread()
            self._research_worker_thread = ephemeral_thread

            LOG.critical(
                "██████ [WORKER_CREATED] id=%s name=%s deep_research=%s "
                "web_access=%s meta_data=%s ██████",
                ephemeral_worker.id,
                getattr(ephemeral_worker, "name", "?"),
                getattr(ephemeral_worker, "deep_research", "?"),
                getattr(ephemeral_worker, "web_access", "?"),
                getattr(ephemeral_worker, "meta_data", "?"),
            )

            prompt = (
                f"TASK: {args.get('task')}\n"
                f"REQ: {args.get('requirements')}\n\n"
                f"⚠️ MANDATORY EXECUTION RULES — NO EXCEPTIONS:\n"
                f"1. Your FIRST action MUST be tool calls: fire `read_scratchpad()` "
                f"AND `perform_web_search()` simultaneously. Do NOT reason first.\n"
                f"2. Your training knowledge is NOT an acceptable source. "
                f"Every fact MUST come from a live URL retrieved in this session.\n"
                f"3. You MUST call `append_scratchpad` with your verified result "
                f"BEFORE sending any text reply.\n"
                f"4. A ✅ [VERIFIED] entry requires an exact value AND a live source URL. "
                f"No URL = no verification = task failure.\n"
                f"5. Sending a confirmation without having called `append_scratchpad` "
                f"means you have failed. The supervisor cannot see your text — "
                f"only the scratchpad."
            )

            msg = await self.create_ephemeral_message(
                ephemeral_thread.id, prompt, ephemeral_worker.id
            )

            ephemeral_run = await self.create_ephemeral_run(
                ephemeral_worker.id,
                ephemeral_thread.id,
                meta_data={
                    "batfish_owner_user_id": origin_user_id,
                    "scratch_pad_thread": self._scratch_pad_thread,
                },
            )

            yield self._research_status("Worker active. Streaming...", "in_progress", run_id)

            # ----------------------------------------
            # Retrieve the users inference api key
            # -----------------------------------------
            run_obj = await self._native_exec.retrieve_run(run_id)
            inference_api_key = run_obj.meta_data.get("api_key") if run_obj.meta_data else None
            delegated_model = (
                run_obj.meta_data.get("delegated_model") if run_obj.meta_data else None
            )

            if not inference_api_key:
                raise RuntimeError(
                    f"DELEGATE ABORT: No api_key found in run {run_id} meta_data. "
                    f"meta_data={run_obj.meta_data}"
                )

            sync_stream = self.project_david_client.synchronous_inference_stream
            sync_stream.setup(
                thread_id=ephemeral_thread.id,
                assistant_id=ephemeral_worker.id,
                message_id=msg.id,
                run_id=ephemeral_run.id,
                api_key=inference_api_key,
            )

            LOG.critical(
                "🎬 WORKER STREAM STARTING - worker=%s thread=%s run=%s model=%s",
                ephemeral_worker.id,
                ephemeral_thread.id,
                ephemeral_run.id,
                "together-ai/Qwen/Qwen3-Next-80B-A3B-Instruct-FP8",
            )

            captured_stream_content = ""
            raw_event_count = 0
            passed_guard1 = 0
            passed_guard2 = 0

            async for event in self._stream_sync_generator(
                sync_stream.stream_events,
                model=delegated_model,
            ):
                raw_event_count += 1
                event_type = type(event).__name__

                LOG.critical(
                    f"👀 [RAW EVENT DUMP] Event {raw_event_count} | Type: {event_type} | Payload: {getattr(event, 'model_dump', lambda: str(event))()}"
                )

                # ✅ INTERCEPT: ScratchpadEvent
                if isinstance(event, ScratchpadEvent):
                    LOG.critical(
                        f"📝 [WORKER SCRATCHPAD EVENT] Action: {event.operation} | State: {event.state} | Entry: {event.entry}"
                    )

                    payload = {
                        "type": "scratchpad_status",
                        "run_id": run_id,
                        "operation": event.operation,
                        "state": event.state,
                        "origin": "research_worker",
                    }
                    if event.tool is not None:
                        payload["tool"] = event.tool
                    if event.activity is not None:
                        payload["activity"] = event.activity
                    if event.assistant_id is not None:
                        payload["assistant_id"] = event.assistant_id
                    entry_val = event.entry or event.content or ""
                    if entry_val:
                        payload["entry"] = entry_val

                    yield json.dumps(payload)
                    continue

                # 🛑 GUARD 1: Status Events
                guard1_triggered = (
                    hasattr(event, "tool")
                    or hasattr(event, "status")
                    or getattr(event, "type", "") == "status"
                )
                if guard1_triggered:
                    # ---> CATCH INSTANT FAILURE AND READ DB ERROR <---
                    if getattr(event, "status", None) == "failed":
                        try:
                            failed_run_obj = await self._native_exec.retrieve_run(ephemeral_run.id)
                            last_err = getattr(
                                failed_run_obj, "last_error", "No error recorded in DB"
                            )
                            if hasattr(last_err, "model_dump"):  # Handle DB objects
                                last_err = last_err.model_dump()
                            LOG.critical(
                                f"🚨 [FATAL RUN ERROR] Engine killed the worker! DB Reason: {last_err}"
                            )
                        except Exception as e:
                            LOG.critical(
                                f"🚨[FATAL RUN ERROR] Run failed, couldn't fetch reason: {e}"
                            )
                    continue
                passed_guard1 += 1

                # 🛑 GUARD 2: Tool Call Payload
                guard2_triggered = getattr(event, "tool_calls", None) or getattr(
                    event, "function_call", None
                )
                if guard2_triggered:
                    tool_calls = getattr(event, "tool_calls", [])
                    tc_list = tool_calls if isinstance(tool_calls, list) else [tool_calls]
                    for tc in tc_list:
                        func = getattr(tc, "function", None)
                        if func:
                            name = getattr(func, "name", "unknown")
                            args = getattr(func, "arguments", "")
                            LOG.critical(
                                f"🛠️[WORKER EXECUTES TOOL] Worker {ephemeral_worker.id} called: {name} | Args: {args}"
                            )
                    continue
                passed_guard2 += 1

                chunk_content = getattr(event, "content", None) or getattr(event, "text", None)
                chunk_reasoning = getattr(event, "reasoning", None)

                if chunk_reasoning:
                    yield json.dumps(
                        {
                            "stream_type": "delegation",
                            "chunk": {
                                "type": "reasoning",
                                "content": chunk_reasoning,
                                "run_id": run_id,
                            },
                        }
                    )

                if chunk_content and isinstance(chunk_content, str):
                    captured_stream_content += chunk_content
                    yield json.dumps(
                        {
                            "stream_type": "delegation",
                            "chunk": {
                                "type": "content",
                                "content": chunk_content,
                                "run_id": run_id,
                            },
                        }
                    )

            LOG.critical(
                "██████ [STREAM_SUMMARY] worker=%s | total_raw_events=%d | "
                "passed_guard1=%d | passed_guard2=%d | captured_content_length=%d ██████",
                ephemeral_worker.id,
                raw_event_count,
                passed_guard1,
                passed_guard2,
                len(captured_stream_content),
            )

            yield self._research_status(
                "Worker stream finished. Finalizing payload...", "in_progress", run_id
            )

            try:
                await self._native_exec.update_run_status(
                    ephemeral_run.id, StatusEnum.completed.value
                )
            except Exception as e:
                LOG.warning(f"⚠️ Could not manually close worker run {ephemeral_run.id}: {e}")

            final_content = captured_stream_content.strip()

            if not final_content:
                LOG.critical(
                    "██████[DELEGATE_FALLBACK] Stream captured no text (raw_events=%d). "
                    "Injecting synthetic fallback to unblock supervisor. ██████",
                    raw_event_count,
                )
                final_content = (
                    "SYSTEM STATUS: The delegated worker successfully executed its tools and finished its run, "
                    "but failed to return a textual summary. Please read the shared scratchpad immediately "
                    "to review the verified facts and data it appended, then continue your synthesis."
                )
            else:
                LOG.critical(
                    "██████ [DELEGATE_SUCCESS] Captured %d chars directly from worker stream. ██████",
                    len(final_content),
                )

            LOG.critical(
                "\n================ WORKER FINAL RETURN PAYLOAD ================\n"
                f"Worker ID: {ephemeral_worker.id}\n"
                f"Content handed back to Supervisor via `delegate_research_task`:\n"
                f"{final_content}\n"
                "==============================================================\n"
            )

            await self.submit_tool_output(
                thread_id=thread_id,
                assistant_id=assistant_id,
                tool_call_id=tool_call_id,
                content=final_content,
                action=action,
                is_error=execution_had_error,
            )

            if action:
                await self._native_exec.update_action_status(
                    action.id,
                    (
                        StatusEnum.completed.value
                        if not execution_had_error
                        else StatusEnum.failed.value
                    ),
                )

        except Exception as e:
            execution_had_error = True
            LOG.error(f"❌[DELEGATE] Error: {e}", exc_info=True)
            yield self._research_status(f"Error: {str(e)}", "error", run_id)

        finally:
            if ephemeral_worker:
                await self._ephemeral_clean_up(
                    ephemeral_worker.id,
                    ephemeral_thread.id if ephemeral_thread else None,
                    self._delete_ephemeral_thread,
                )

                # -------------------------------------------------
                # Scrub the users inference api key from the db
                # -------------------------------------------------
                await self._native_exec.update_run_fields(run_id, meta_data={"api_key": "***"})

            yield self._research_status(
                "Delegation complete.",
                "completed" if not execution_had_error else "error",
                run_id,
            )
