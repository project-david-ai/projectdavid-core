from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Dict, List, Optional

from redis import Redis

from src.api.entities_api.db.database import SessionLocal
from src.api.entities_api.orchestration.engine.inference_arbiter import InferenceArbiter
from src.api.entities_api.orchestration.engine.inference_provider_selector import (
    InferenceProviderSelector,
)
from src.api.entities_api.services.inference_resolver import InferenceResolver


class StatelessInferenceService:
    """Execute one provider request without conversation orchestration or writes."""

    def __init__(
        self,
        *,
        redis: Optional[Redis] = None,
        selector: Optional[InferenceProviderSelector] = None,
        session_factory=SessionLocal,
    ) -> None:
        if selector is None:
            if redis is None:
                raise ValueError("redis is required when selector is not injected")
            selector = InferenceProviderSelector(InferenceArbiter(redis=redis))

        self.selector = selector
        self.session_factory = session_factory

    async def create_completion(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        provider_api_key: Optional[str],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        worker, provider_model = self.selector.select_provider_worker(model)
        stream = self._open_provider_stream(
            worker=worker,
            model=model,
            provider_model=provider_model,
            messages=messages,
            provider_api_key=provider_api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        content_parts: List[str] = []
        async for chunk in stream:
            error = self._extract_error(chunk)
            if error:
                raise RuntimeError(error)
            text = self._extract_text(chunk)
            if text:
                content_parts.append(text)

        return "".join(content_parts)

    def _open_provider_stream(
        self,
        *,
        worker: Any,
        model: str,
        provider_model: str,
        messages: List[Dict[str, str]],
        provider_api_key: Optional[str],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        if hasattr(worker, "_stream_vllm_raw"):
            target_url = self._resolve_vllm_target(worker, model)
            return worker._stream_vllm_raw(
                messages=messages,
                model=provider_model,
                temperature=temperature,
                max_tokens=max_tokens,
                think=False,
                tools=None,
                base_url=target_url,
            )

        if hasattr(worker, "_stream_ollama_raw"):
            return worker._stream_ollama_raw(
                messages=messages,
                model=provider_model,
                temperature=temperature,
                max_tokens=max_tokens,
                think=False,
                tools=None,
            )

        if not provider_api_key:
            raise ValueError("A provider API key is required for this model")

        client = worker._get_client_instance(api_key=provider_api_key)
        return client.stream_chat_completion(
            messages=messages,
            model=provider_model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=True,
        )

    def _resolve_vllm_target(self, worker: Any, model: str) -> str:
        db = self.session_factory()
        try:
            resolved = InferenceResolver.resolve_vllm_url(db, model)
        finally:
            db.close()

        return resolved or worker.base_url

    @staticmethod
    def _extract_error(chunk: Any) -> str:
        if not isinstance(chunk, dict):
            return ""

        if chunk.get("type") == "error":
            return str(chunk.get("message") or chunk.get("error") or "Inference failed")

        choices = chunk.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            choice = choices[0]
            if choice.get("finish_reason") == "error":
                delta = choice.get("delta")
                if isinstance(delta, dict) and delta.get("content"):
                    return str(delta["content"])
                return "Inference provider returned an error"

        return ""

    @classmethod
    def _extract_text(cls, chunk: Any) -> str:
        if isinstance(chunk, str):
            try:
                chunk = json.loads(chunk)
            except (TypeError, ValueError):
                return chunk

        if not isinstance(chunk, dict):
            return ""

        choices = chunk.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0] if isinstance(choices[0], dict) else {}
            delta = choice.get("delta")
            if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                return delta["content"]
            if isinstance(choice.get("text"), str):
                return choice["text"]
            message = choice.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]

        message = chunk.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]

        if isinstance(chunk.get("response"), str):
            return chunk["response"]

        if chunk.get("type") == "content" and isinstance(chunk.get("content"), str):
            return chunk["content"]

        return ""
