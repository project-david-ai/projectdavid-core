from __future__ import annotations

import asyncio
import json
import os
import threading
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Callable, Dict, Optional

import httpx
from projectdavid.events import ScratchpadEvent, WebStatusEvent
from projectdavid_common.utilities.logging_service import LoggingUtility
from projectdavid_common.validation import StatusEnum

from src.api.entities_api.services.native_execution_service import (
    NativeExecutionService,
)
from src.api.entities_api.utilities.assistant_manager import AssistantManager

LOG = LoggingUtility()


class _RegisteredProviderInferenceClient:
    """Per-child-invocation adapter to the authenticated Core BYOK SSE contract."""

    def __init__(self, *, base_url: str, core_api_key: str, provider_key_id: str):
        self.base_url = base_url
        self.core_api_key = core_api_key
        self.provider_key_id = provider_key_id

    async def stream_inference_response(
        self,
        *,
        model: str,
        thread_id: str,
        message_id: str,
        run_id: str,
        assistant_id: str,
        api_key: str | None = None,
        timeout: float = 600.0,
        **kwargs,
    ) -> AsyncGenerator[dict, None]:
        if api_key is not None:
            raise RuntimeError("Registered inference cannot accept a plaintext key")
        payload = {
            "model": model,
            "thread_id": thread_id,
            "message_id": message_id,
            "run_id": run_id,
            "assistant_id": assistant_id,
            "provider_key_id": self.provider_key_id,
            "stream": True,
        }
        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=10.0),
            headers={"X-API-Key": self.core_api_key},
        ) as client:
            async with client.stream(
                "POST", "/v1/completions", json=payload
            ) as response:
                response.raise_for_status()
                finished = False
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        finished = True
                        break
                    try:
                        chunk = json.loads(data)
                    except ValueError as exc:
                        raise RuntimeError(
                            "Invalid delegated inference stream frame"
                        ) from exc
                    if not isinstance(chunk, dict):
                        raise RuntimeError("Invalid delegated inference stream chunk")
                    if chunk.get("type") == "error":
                        raise RuntimeError("Delegated inference stream failed")
                    yield chunk
                if not finished:
                    raise RuntimeError(
                        "Delegated inference stream ended without completion"
                    )


_TERMINAL_RUN_STATES = {"completed", "failed", "cancelled", "expired"}
_WORKER_RUN_TIMEOUT = 1200
_WORKER_POLL_INTERVAL = 2.0


class DelegationMixin:
    """
    Ephemeral worker lifecycle management for multi-agent delegation flows.
    Handles research delegation and engineer delegation via spawned ephemeral
    assistants running on isolated threads and runs.

    Owns:
        __init__()                         cooperative, calls super().__init__().
                                            Only mixin in the stack with __init__
                                            participation. Initialises delegation
                                            state attributes.
        _stream_sync_generator()           bridges a blocking sync generator into
                                            an async generator via a daemon thread
                                            and asyncio.Queue
        _wait_for_run_completion()         async poller with configurable timeout
        _ephemeral_clean_up()              deletes ephemeral assistant and
                                            optionally the ephemeral thread
        _capture_tool_outputs()            async context manager that intercepts
                                            submit_tool_output calls into a dict
        create_ephemeral_worker_assistant()
        create_ephemeral_junior_engineer()
        create_ephemeral_thread()
        create_ephemeral_message()
        create_ephemeral_run()             factory helpers, all require
                                            self._run_user_id to be set first
        _fetch_worker_final_report()       fallback message retrieval, retained
                                            for other components
        handle_delegate_research_task()    async generator, full research
                                            delegation lifecycle
        handle_delegate_engineer_task()    async generator (implied by symmetry)

    Requires on self:
        self._native_exec                  NativeExecMixin
        self._run_user_id                  set in QwenBaseWorker.stream() before
                                            delegation is triggered
        self.submit_tool_output()          ConsumerToolHandlersMixin
        self.project_david_client          ServiceRegistryMixin, used for
                                            synchronous_inference_stream
        self._assistant_manager            lazy property, self-initialising
                                            via AssistantManager()

    Instance state (set in __init__):
        self._delete_ephemeral_thread      bool, controls thread cleanup
        self._research_worker_thread       ephemeral thread id for research flow
        self._run_user_id                  owner resolved from run at stream time

    Contract:
        DelegationMixin is the only mixin in _ProviderMixins that calls
        super().__init__(). All other mixins are stateless at construction.
        Do not add __init__ to other mixins without updating the _ProviderMixins
        and OrchestratorCore contract comments.
        All factory methods guard on self._run_user_id via getattr and raise
        RuntimeError explicitly  this is the correct pattern for cross-mixin
        state access and should be followed in any new factory methods added here.
        Credential handoff is request-scoped. No plaintext credential is
        persisted in parent Run metadata or serialized into tool events.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._delete_ephemeral_thread = False
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
                LOG.error(f"[THREAD-ERR] {e}")
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
            " [DELEGATE] Waiting for worker run %s to complete (timeout=%ss)...",
            run_id,
            timeout,
        )
        elapsed = 0.0
        while elapsed < timeout:
            try:
                run = await self._native_exec.retrieve_run(run_id)
                status_value = (
                    run.status.value
                    if hasattr(run.status, "value")
                    else str(run.status)
                )
                LOG.critical(
                    " [DELEGATE_POLL] run_id=%s status=%s elapsed=%.1fs ",
                    run_id,
                    status_value,
                    elapsed,
                )
                if status_value in _TERMINAL_RUN_STATES:
                    LOG.critical(
                        " [DELEGATE_POLL] run_id=%s reached terminal state=%s ",
                        run_id,
                        status_value,
                    )
                    return status_value
            except Exception as e:
                LOG.warning("[DELEGATE_POLL] Error polling run %s: %s", run_id, e)
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
        LOG.error("[DELEGATE_POLL] run_id=%s timed out after %ss.", run_id, timeout)
        raise asyncio.TimeoutError(
            f"Worker run {run_id} did not complete within {timeout}s"
        )

    # ------------------------------------------------------------------
    # HELPER: Lifecycle cleanup
    # ------------------------------------------------------------------
    async def _ephemeral_clean_up(
        self, assistant_id: str, thread_id: Optional[str], delete_thread: bool = False
    ):
        LOG.info(f"[CLEANUP] Assistant: {assistant_id} | Thread: {thread_id}")

        user_id = getattr(self, "_batfish_owner_user_id", None)

        if delete_thread and thread_id:
            try:
                if not user_id:
                    LOG.warning(
                        " [CLEANUP] Cannot delete thread %s  user_id not resolved.",
                        thread_id,
                    )
                else:
                    await self._native_exec.delete_thread(thread_id, user_id=user_id)
            except Exception as e:
                LOG.warning(f"[CLEANUP] Thread delete failed: {e}")

        if user_id and assistant_id:
            try:
                await self._assistant_manager.delete_assistant(
                    assistant_id=assistant_id, user_id=user_id, permanent=True
                )
            except Exception as e:
                LOG.warning(f" [CLEANUP] Assistant delete failed: {e}")

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
                "resolved yet  ensure it is set before calling this method."
            )
        return await self._assistant_manager.create_ephemeral_worker_assistant(
            user_id=user_id
        )

    async def create_ephemeral_junior_engineer(self):
        user_id = getattr(self, "_batfish_owner_user_id", None)
        if not user_id:
            raise RuntimeError(
                "create_ephemeral_junior_engineer: _batfish_owner_user_id has not been "
                "resolved yet  ensure it is set before calling this method."
            )
        return await self._assistant_manager.create_ephemeral_junior_engineer(
            user_id=user_id
        )

    async def create_ephemeral_thread(self):
        user_id = getattr(self, "_batfish_owner_user_id", None)
        if not user_id:
            raise RuntimeError(
                "create_ephemeral_thread: _batfish_owner_user_id has not been "
                "resolved yet  ensure it is set before calling this method."
            )
        return await self._native_exec.create_thread(user_id=user_id)

    async def create_ephemeral_message(self, thread_id, content, assistant_id):
        return await self._native_exec.create_message(
            thread_id=thread_id,
            assistant_id=assistant_id,
            content=content,
        )

    async def create_ephemeral_run(
        self,
        assistant_id,
        thread_id,
        meta_data: Dict | None = None,
        model: Optional[str] = None,
        *,
        trusted_metadata: bool = False,
    ):
        user_id = getattr(self, "_batfish_owner_user_id", None)
        if not user_id:
            raise RuntimeError(
                "create_ephemeral_run: _batfish_owner_user_id has not been "
                "resolved yet  ensure it is set before calling this method."
            )
        return await self._native_exec.create_run(
            assistant_id=assistant_id,
            thread_id=thread_id,
            user_id=user_id,
            meta_data=meta_data,
            model=model,
            trusted_metadata=trusted_metadata,
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
                LOG.exception(" [WORKER_FETCH] Error: %s", e)
            if attempt < max_attempts:
                await asyncio.sleep(retry_delay)

        return None

    # ------------------------------------------------------------------
    # HANDLER 1: Research Delegation
    # ------------------------------------------------------------------
    async def handle_delegate_research_task(
        self,
        thread_id,
        run_id,
        assistant_id,
        arguments_dict,
        tool_call_id,
        decision,
        *,
        inference_api_key: Optional[str] = None,
        provider_key_id: Optional[str] = None,
        project_david_api_key: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:

        self._scratch_pad_thread = thread_id
        LOG.info(f"[DELEGATE] STARTING. Run: {run_id}")

        if isinstance(arguments_dict, str):
            try:
                args = json.loads(arguments_dict)
            except Exception:
                args = {"task": arguments_dict}
        else:
            args = arguments_dict

        yield self._research_status(
            "Initializing delegation worker...", "in_progress", run_id
        )

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
            LOG.error(f"[DELEGATE] Action creation failed: {e}")

        ephemeral_worker = None
        ephemeral_thread = None
        execution_had_error = False
        worker_failure_reason = None
        ephemeral_run = None
        delegation_succeeded = False

        try:
            if action is None:
                raise RuntimeError(
                    "DELEGATE ABORT: No durable parent Action was created."
                )

            # Provider secrets must not be propagated to child HTTP requests.
            if not provider_key_id and not inference_api_key:
                raise RuntimeError(
                    "DELEGATE ABORT: No credential was supplied to delegation."
                )
            if not project_david_api_key:
                raise RuntimeError(
                    "DELEGATE ABORT: Project David API key was not supplied to delegation."
                )

            origin_user_id = getattr(self, "_batfish_owner_user_id", None)

            if not origin_user_id:
                run_obj = await self._native_exec.retrieve_run(run_id)
                origin_user_id = run_obj.user_id
                self._batfish_owner_user_id = origin_user_id

            LOG.info(
                "RESEARCH_DELEGATE  origin_user_id=%s | scratch_pad_thread=%s",
                origin_user_id,
                self._scratch_pad_thread,
            )

            # The delegated model can differ from the temporary Assistant
            # default. Resolve it before creating the child Run so the persisted
            # Run model and the SDK stream model always agree.
            parent_run = await self._native_exec.retrieve_run(run_id)
            parent_metadata = parent_run.meta_data or {}
            root_run_id = parent_metadata.get("root_run_id") or run_id
            inference_model = parent_metadata.get("delegated_model")
            if not isinstance(inference_model, str) or not inference_model.strip():
                raise RuntimeError(
                    f"DELEGATE ABORT: No delegated model found in run {run_id} meta_data."
                )
            inference_model = inference_model.strip()
            if provider_key_id:
                # A single registered key cannot authorize a different provider.
                # The child endpoint also revalidates key ownership and provider.
                parent_provider = str(parent_run.model).strip().lower().split("/", 1)[0]
                child_provider = inference_model.lower().split("/", 1)[0]
                if parent_provider != child_provider:
                    raise RuntimeError(
                        "DELEGATE ABORT: Delegated model requires a credential "
                        "registered for its own provider."
                    )

            ephemeral_worker = await self.create_ephemeral_worker_assistant()
            ephemeral_thread = await self.create_ephemeral_thread()
            self._research_worker_thread = ephemeral_thread

            LOG.critical(
                " [WORKER_CREATED] id=%s name=%s deep_research=%s "
                "web_access=%s meta_data=%s ",
                ephemeral_worker.id,
                getattr(ephemeral_worker, "name", "?"),
                getattr(ephemeral_worker, "deep_research", "?"),
                getattr(ephemeral_worker, "web_access", "?"),
                getattr(ephemeral_worker, "meta_data", "?"),
            )

            prompt = (
                f"TASK: {args.get('task')}\n"
                f"REQ: {args.get('requirements')}\n\n"
                f" MANDATORY EXECUTION RULES  NO EXCEPTIONS:\n"
                f"1. Your FIRST action MUST be tool calls: fire `read_scratchpad()` "
                f"AND `perform_web_search()` simultaneously. Do NOT reason first.\n"
                f"2. Your training knowledge is NOT an acceptable source. "
                f"Every fact MUST come from a live URL retrieved in this session.\n"
                f"3. You MUST call `append_scratchpad` with your verified result "
                f"BEFORE sending any text reply.\n"
                f"4. A  [VERIFIED] entry requires an exact value AND a live source URL. "
                f"No URL = no verification = task failure.\n"
                f"5. Sending a confirmation without having called `append_scratchpad` "
                f"means you have failed. The supervisor cannot see your text  "
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
                    "root_run_id": root_run_id,
                    "parent_run_id": run_id,
                    "parent_action_id": action.id,
                },
                model=inference_model,
                trusted_metadata=True,
            )

            # Persist lineage before starting the worker stream.
            await self._native_exec.link_research_delegation(
                parent_run_id=run_id,
                child_run_id=ephemeral_run.id,
                parent_action_id=action.id,
                worker_assistant_id=ephemeral_worker.id,
                worker_model=inference_model,
            )

            await self._native_exec.update_action_status(action.id, "processing")

            yield self._research_status(
                "Worker active. Streaming...", "in_progress", run_id
            )

            user_project_david_client = self._create_project_david_client(
                api_key=project_david_api_key,
                base_url=os.getenv("ASSISTANTS_BASE_URL"),
            )

            sync_stream = user_project_david_client.synchronous_inference_stream
            sync_stream.setup(
                thread_id=ephemeral_thread.id,
                assistant_id=ephemeral_worker.id,
                message_id=msg.id,
                run_id=ephemeral_run.id,
                api_key=None if provider_key_id else inference_api_key,
            )
            if provider_key_id:
                # Preserve SDK stream_events semantics (tool validation, retries,
                # typed events) while replacing only its per-request HTTP transport.
                sync_stream.inference_client = _RegisteredProviderInferenceClient(
                    base_url=user_project_david_client.base_url,
                    core_api_key=project_david_api_key,
                    provider_key_id=provider_key_id,
                )

            LOG.critical(
                " WORKER STREAM STARTING - worker=%s thread=%s run=%s model=%s",
                ephemeral_worker.id,
                ephemeral_thread.id,
                ephemeral_run.id,
                inference_model,
            )

            captured_stream_content = ""
            raw_event_count = 0
            passed_guard1 = 0
            passed_guard2 = 0

            async for event in self._stream_sync_generator(
                sync_stream.stream_events,
                model=inference_model,
            ):
                raw_event_count += 1
                event_type = type(event).__name__

                LOG.debug(
                    "[WORKER EVENT] event=%s type=%s",
                    raw_event_count,
                    event_type,
                )

                #  INTERCEPT: ScratchpadEvent
                if isinstance(event, ScratchpadEvent):
                    LOG.critical(
                        " [WORKER SCRATCHPAD EVENT] Action: %s | State: %s | Entry: %s",
                        event.operation,
                        event.state,
                        event.entry,
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

                # ----------------------------------------
                # INTERCEPT: WebStatusEvent
                #
                # The delegated worker runs through its own
                # Project David SDK stream. Relay its web
                # lifecycle events back onto the parent run
                # so the caller's SDK can observe them.
                # ----------------------------------------
                if isinstance(event, WebStatusEvent):
                    payload = {
                        "type": "web_status",
                        "run_id": run_id,
                        "status": event.status,
                    }

                    if event.message is not None:
                        payload["message"] = event.message

                    if event.tool is not None:
                        payload["tool"] = event.tool

                    yield json.dumps(payload)
                    continue

                #  GUARD 1: Status Events
                guard1_triggered = (
                    hasattr(event, "tool")
                    or hasattr(event, "status")
                    or getattr(event, "type", "") == "status"
                )
                if guard1_triggered:
                    # ---> CATCH INSTANT FAILURE AND READ DB ERROR <---
                    if getattr(event, "status", None) == "failed":
                        execution_had_error = True

                        event_message = getattr(event, "message", None)
                        if (
                            event_message
                            and str(event_message).strip().lower() != "none"
                        ):
                            worker_failure_reason = str(event_message)

                        try:
                            failed_run_obj = await self._native_exec.retrieve_run(
                                ephemeral_run.id
                            )

                            last_err = getattr(
                                failed_run_obj,
                                "last_error",
                                None,
                            )

                            if hasattr(last_err, "model_dump"):
                                last_err = last_err.model_dump()

                            if last_err:
                                worker_failure_reason = str(last_err)

                            LOG.critical(
                                "FATAL RUN ERROR: worker=%s reason=%s",
                                ephemeral_worker.id,
                                worker_failure_reason,
                            )

                        except Exception as lifecycle_exc:
                            LOG.critical(
                                "Failed retrieving worker failure state: %s",
                                lifecycle_exc,
                            )

                        if not worker_failure_reason:
                            worker_failure_reason = (
                                "Delegated worker inference stream failed."
                            )
                    continue
                passed_guard1 += 1

                #  GUARD 2: Tool Call Payload
                guard2_triggered = getattr(event, "tool_calls", None) or getattr(
                    event, "function_call", None
                )
                if guard2_triggered:
                    tool_calls = getattr(event, "tool_calls", [])
                    tc_list = (
                        tool_calls if isinstance(tool_calls, list) else [tool_calls]
                    )
                    for tc in tc_list:
                        func = getattr(tc, "function", None)
                        if func:
                            name = getattr(func, "name", "unknown")
                            args = getattr(func, "arguments", "")
                            LOG.critical(
                                "[WORKER EXECUTES TOOL] Worker %s called: %s | Args: %s",
                                ephemeral_worker.id,
                                name,
                                args,
                            )
                    continue
                passed_guard2 += 1

                chunk_content = getattr(event, "content", None) or getattr(
                    event, "text", None
                )
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
                " [STREAM_SUMMARY] worker=%s | total_raw_events=%d | "
                "passed_guard1=%d | passed_guard2=%d | captured_content_length=%d ",
                ephemeral_worker.id,
                raw_event_count,
                passed_guard1,
                passed_guard2,
                len(captured_stream_content),
            )

            yield self._research_status(
                "Worker stream finished. Finalizing payload...", "in_progress", run_id
            )

            final_content = captured_stream_content.strip()

            # Streamed content alone does not prove child Run completion.
            child = await self._native_exec.retrieve_run(ephemeral_run.id)
            child_status = (
                child.status.value
                if hasattr(child.status, "value")
                else str(child.status)
            )
            if child_status != "completed":
                execution_had_error = True
                if not worker_failure_reason:
                    worker_failure_reason = (
                        "Delegated worker ended without a completed Run "
                        f"(status={child_status})."
                    )

            if execution_had_error:
                failure_reason = (
                    worker_failure_reason
                    or "Delegated worker failed before completing the task."
                )

                LOG.error(
                    "[DELEGATE_FAILURE] worker=%s; discarding %d chars of partial stream content; reason=%s",
                    ephemeral_worker.id,
                    len(captured_stream_content),
                    failure_reason,
                )

                final_content = (
                    "Delegated worker failed before completing the task. "
                    + f"Reason: {failure_reason}"
                )

            elif not final_content:
                LOG.critical(
                    "[DELEGATE_FALLBACK] Stream captured no text (raw_events=%d).",
                    raw_event_count,
                )
                final_content = (
                    "SYSTEM STATUS: The delegated worker successfully executed "
                    "its tools and finished its run, but failed to return a "
                    "textual summary. Please read the shared scratchpad "
                    "immediately to review the verified facts and data it "
                    "appended, then continue your synthesis."
                )

            else:
                LOG.critical(
                    "[DELEGATE_SUCCESS] Captured %d chars directly from worker stream.",
                    len(final_content),
                )

            LOG.critical(
                "\n================ WORKER FINAL RETURN PAYLOAD ================\n"
                "Worker ID: %s\n"
                "Content handed back to Supervisor via `delegate_research_task`:\n"
                "%s\n"
                "==============================================================\n",
                ephemeral_worker.id,
                final_content,
            )

            await self.submit_tool_output(
                thread_id=thread_id,
                assistant_id=assistant_id,
                tool_call_id=tool_call_id,
                content=final_content,
                action=action,
                is_error=execution_had_error,
            )

            delegation_succeeded = not execution_had_error

        except Exception as exc:
            execution_had_error = True
            LOG.error("[DELEGATE] Research delegation failed: %s", exc, exc_info=True)
            safe_reason = (
                str(exc)
                if str(exc).startswith("DELEGATE ABORT:")
                else "Internal worker execution error; consult Core logs."
            )

            if ephemeral_run is not None:
                try:
                    failed_child = await self._native_exec.retrieve_run(
                        ephemeral_run.id
                    )
                    failed_state = (
                        failed_child.status.value
                        if hasattr(failed_child.status, "value")
                        else str(failed_child.status)
                    )
                    if failed_state not in _TERMINAL_RUN_STATES:
                        await self._native_exec.update_run_status(
                            ephemeral_run.id, StatusEnum.failed.value
                        )
                except Exception:
                    LOG.error(
                        "[DELEGATE] Failed marking child Run failed",
                        exc_info=True,
                    )

            if action is not None:
                try:
                    await self.submit_tool_output(
                        thread_id=thread_id,
                        assistant_id=assistant_id,
                        tool_call_id=tool_call_id,
                        content=safe_reason,
                        action=action,
                        is_error=True,
                    )
                except Exception:
                    LOG.error(
                        "[DELEGATE] Failed submitting error tool output",
                        exc_info=True,
                    )

            yield self._research_status(f"Error: {safe_reason}", "error", run_id)

        finally:
            # Preserve a worker that has not yet reached a terminal state.
            safe_to_cleanup = ephemeral_run is None
            if ephemeral_run is not None:
                try:
                    child = await self._native_exec.retrieve_run(ephemeral_run.id)
                    child_status = (
                        child.status.value
                        if hasattr(child.status, "value")
                        else str(child.status)
                    )

                    if child_status in _TERMINAL_RUN_STATES:
                        await self._native_exec.retire_research_worker_actor(
                            child_run_id=ephemeral_run.id,
                        )
                        safe_to_cleanup = True
                    else:
                        LOG.warning(
                            "[DELEGATE] Preserving nonterminal worker %s "
                            "(run=%s state=%s)",
                            getattr(ephemeral_worker, "id", None),
                            ephemeral_run.id,
                            child_status,
                        )
                except Exception:
                    LOG.error(
                        "[DELEGATE] Failed verifying/retiring worker actor",
                        exc_info=True,
                    )

            if action is not None:
                try:
                    await self._native_exec.update_action_status(
                        action.id,
                        (
                            StatusEnum.completed.value
                            if delegation_succeeded
                            else StatusEnum.failed.value
                        ),
                    )
                except Exception:
                    LOG.error(
                        "[DELEGATE] Failed finalizing parent Action %s",
                        action.id,
                        exc_info=True,
                    )

            if ephemeral_worker is not None and safe_to_cleanup:
                try:
                    await self._ephemeral_clean_up(
                        ephemeral_worker.id,
                        ephemeral_thread.id if ephemeral_thread else None,
                        self._delete_ephemeral_thread,
                    )
                except Exception:
                    LOG.error(
                        "[DELEGATE] Worker cleanup failed",
                        exc_info=True,
                    )

        yield self._research_status(
            "Delegation complete." if delegation_succeeded else "Delegation failed.",
            "completed" if delegation_succeeded else "error",
            run_id,
        )
