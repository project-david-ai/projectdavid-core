from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Generator, Optional

from dotenv import load_dotenv
from entities_api.clients.async_to_sync import async_to_sync_stream
from entities_api.orchestration.engine.orchestrator_core import OrchestratorCore
from projectdavid_common import ValidationInterface
from projectdavid_common.utilities.logging_service import LoggingUtility

from src.api.entities_api.dependencies import get_redis
from src.api.entities_api.orchestration.mixins import (
    AssistantCacheMixin,
    CodeInterpreterMixin,
    ConsumerToolHandlersMixin,
    ContextMixin,
    FileSearchMixin,
    JsonUtilsMixin,
    PlatformToolHandlersMixin,
    ShellExecutionMixin,
    ToolRoutingMixin,
)

# TODO: Move this to the clients cache
from src.api.entities_api.orchestration.workers.deepseek.deepseek_async_client import (
    AsyncDeepSeekClient,
)

load_dotenv()
LOG = LoggingUtility()


class _ProviderMixins(
    AssistantCacheMixin,
    JsonUtilsMixin,
    ContextMixin,
    ToolRoutingMixin,
    PlatformToolHandlersMixin,
    ConsumerToolHandlersMixin,
    CodeInterpreterMixin,
    ShellExecutionMixin,
    FileSearchMixin,
):
    """All helper behaviour is inherited from these mixins."""

    pass


class DeepSeekChatInference(_ProviderMixins, OrchestratorCore):
    """
    Generic DeepSeek provider that follows the same streaming contract
    as HyperbolicDs1 while keeping the async DeepSeek client.
    """

    def __init__(
        self,
        *,
        assistant_id: str | None = None,
        thread_id: str | None = None,
        redis=None,
        api_key: str | None = None,
        assistant_cache: dict | None = None,
        **extra,
    ) -> None:
        self._assistant_cache = assistant_cache or {}
        self.redis = redis or get_redis()
        self.assistant_id = assistant_id
        self.thread_id = thread_id
        self.api_key = api_key
        self.model_name = extra.get("model_name", "deepseek-ai/DeepSeek-V3")
        self.max_context_window = extra.get("max_context_window", 128000)
        self.threshold_percentage = extra.get("threshold_percentage", 0.8)
        self.setup_services()
        LOG.debug("DeepSeekChatInference ready (assistant=%s)", assistant_id or "<lazy>")

    @property
    def assistant_cache(self) -> dict:
        return self._assistant_cache

    @assistant_cache.setter
    def assistant_cache(self, value: dict) -> None:
        if hasattr(self, "_assistant_cache"):
            raise AttributeError("assistant_cache already initialised")
        self._assistant_cache = value

    def get_assistant_cache(self) -> dict:
        return self._assistant_cache

    @staticmethod
    def _filter_fc(chunk_json: str) -> Optional[str]:
        try:
            if json.loads(chunk_json).get("type") == "function_call":
                return None
        except Exception:
            pass
        return chunk_json

    def stream(
        self,
        thread_id: str,
        message_id: str,
        run_id: str,
        assistant_id: str,
        model: Any,
        *,
        stream_reasoning: bool = True,
        api_key: Optional[str] = None,
    ) -> Generator[str, None, None]:
        self.start_cancellation_listener(run_id)
        redis = get_redis()
        stream_key = f"stream:{run_id}"
        if mapped := self._get_model_map(model):
            model = mapped
        messages = self._set_up_context_window(assistant_id, thread_id, trunk=True)
        api_key = api_key or self.api_key
        if not api_key:
            err = {
                "type": "error",
                "content": f"Run {run_id}: DeepSeek API key missing",
            }
            yield json.dumps(err)
            self._shunt_to_redis_stream(redis, stream_key, err)
            return
        client = AsyncDeepSeekClient(
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        )
        start_chunk = {"type": "status", "status": "started", "run_id": run_id}
        yield json.dumps(start_chunk)
        self._shunt_to_redis_stream(redis, stream_key, start_chunk)
        assistant_reply = accumulated = reasoning_buf = ""
        partial_tag = ""
        fc_buffer: list[str] = []
        in_reasoning = in_function_call = code_mode = False
        code_buf = ""
        tag_re = re.compile("(<think>|</think>|<fc>|</fc>)")
        FLUSH_MS = 80
        last_fc_ts = time.monotonic()

        def _flush_fc():
            nonlocal fc_buffer, last_fc_ts
            if not fc_buffer:
                return
            content = "".join(fc_buffer)
            payload_fc = {"type": "content", "content": content}
            self._shunt_to_redis_stream(redis, stream_key, payload_fc)
            if p := self._filter_fc(json.dumps(payload_fc)):
                yield p
            fc_buffer.clear()
            last_fc_ts = time.monotonic()

        try:
            async_stream = client.stream_chat_completion(
                prompt_or_messages=messages, model=model, temperature=0.6, top_p=0.9
            )
            for token in async_to_sync_stream(async_stream):
                if self.check_cancellation_flag():
                    err = {"type": "error", "content": "Run cancelled"}
                    if p := self._filter_fc(json.dumps(err)):
                        yield p
                    self._shunt_to_redis_stream(redis, stream_key, err)
                    break
                if not token:
                    continue
                raw = token
                cleaned = (
                    raw.replace("[content]", "").replace("<fc ", "<fc>").replace("</ fc>", "</fc>")
                )
                cleaned = re.sub("<fc\\s*>", "<fc>", cleaned, flags=re.I)
                cleaned = re.sub("</fc\\s*>", "</fc>", cleaned, flags=re.I)
                cleaned = re.sub("<think\\s*>", "<think>", cleaned, flags=re.I)
                cleaned = re.sub("</think\\s*>", "</think>", cleaned, flags=re.I)
                piece = partial_tag + cleaned
                partial_tag = ""
                segs, last = ([], 0)
                for m in tag_re.finditer(piece):
                    if last < m.start():
                        segs.append(piece[last : m.start()])
                    segs.append(m.group())
                    last = m.end()
                segs.append(piece[last:])
                for seg in segs:
                    if seg.startswith("<") and (not tag_re.fullmatch(seg)):
                        partial_tag = seg
                        break
                    if seg == "<think>":
                        in_reasoning = True
                        if stream_reasoning:
                            msg = {"type": "reasoning", "content": seg}
                            if p := self._filter_fc(json.dumps(msg)):
                                yield p
                            self._shunt_to_redis_stream(redis, stream_key, msg)
                        continue
                    if seg == "</think>":
                        in_reasoning = False
                        if stream_reasoning:
                            msg = {"type": "reasoning", "content": seg}
                            if p := self._filter_fc(json.dumps(msg)):
                                yield p
                            self._shunt_to_redis_stream(redis, stream_key, msg)
                        continue
                    if seg == "<fc>":
                        in_function_call = True
                        fc_buffer = []
                        last_fc_ts = time.monotonic()
                        continue
                    if seg == "</fc>":
                        in_function_call = False
                        yield from _flush_fc()
                        continue
                    if in_function_call:
                        fc_buffer.append(seg)
                        last_fc_ts = time.monotonic()
                        continue
                    if in_reasoning:
                        reasoning_buf += seg
                        if stream_reasoning:
                            msg = {"type": "reasoning", "content": seg}
                            if p := self._filter_fc(json.dumps(msg)):
                                yield p
                            self._shunt_to_redis_stream(redis, stream_key, msg)
                        continue
                    assistant_reply += seg
                    accumulated += seg
                    partial_ci = (
                        self.parse_code_interpreter_partial(accumulated) if not code_mode else None
                    )
                    if partial_ci:
                        code_mode = True
                        code_buf = partial_ci["code"]
                        start_hot = {"type": "hot_code", "content": "```python\n"}
                        if p := self._filter_fc(json.dumps(start_hot)):
                            yield p
                        self._shunt_to_redis_stream(redis, stream_key, start_hot)
                        if code_buf and hasattr(self, "_process_code_interpreter_chunks"):
                            res, code_buf = self._process_code_interpreter_chunks("", code_buf)
                            for r in res:
                                if p := self._filter_fc(r):
                                    yield p
                                self._shunt_to_redis_stream(redis, stream_key, json.loads(r))
                        continue
                    if code_mode:
                        res, code_buf = self._process_code_interpreter_chunks(seg, code_buf)
                        for r in res:
                            if p := self._filter_fc(r):
                                yield p
                            self._shunt_to_redis_stream(redis, stream_key, json.loads(r))
                        continue
                    msg = {"type": "content", "content": seg}
                    if p := self._filter_fc(json.dumps(msg)):
                        yield p
                    self._shunt_to_redis_stream(redis, stream_key, msg)
                if (
                    in_function_call
                    and fc_buffer
                    and ((time.monotonic() - last_fc_ts) * 1000 > FLUSH_MS)
                ):
                    yield from _flush_fc()
        except Exception as exc:
            err = {"type": "error", "content": f"DeepSeek SDK error: {exc}"}
            if p := self._filter_fc(json.dumps(err)):
                yield p
            self._shunt_to_redis_stream(redis, stream_key, err)
            return
        end_chunk = {"type": "status", "status": "complete", "run_id": run_id}
        yield json.dumps(end_chunk)
        self._shunt_to_redis_stream(redis, stream_key, end_chunk)
        if assistant_reply:
            self.finalize_conversation(
                reasoning_buf + assistant_reply, thread_id, assistant_id, run_id
            )
        if accumulated and self.parse_and_set_function_calls(accumulated, assistant_reply):
            self.project_david_client.runs.update_run_status(
                run_id, ValidationInterface.StatusEnum.pending_action
            )
        elif not self.get_function_call_state():
            self.project_david_client.runs.update_run_status(
                run_id, ValidationInterface.StatusEnum.completed
            )

    def process_conversation(
        self,
        thread_id: str,
        message_id: str,
        run_id: str,
        assistant_id: str,
        model: Any,
        *,
        stream_reasoning: bool = False,
        api_key: Optional[str] = None,
    ) -> Generator[str, None, None]:
        yield from self.stream(
            thread_id=thread_id,
            message_id=message_id,
            run_id=run_id,
            assistant_id=assistant_id,
            model=model,
            stream_reasoning=stream_reasoning,
            api_key=api_key,
        )
        if self.get_function_call_state():
            yield from self.process_tool_calls(
                thread_id, run_id, assistant_id, model=model, api_key=api_key
            )
