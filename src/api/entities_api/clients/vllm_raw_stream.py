# src/api/entities_api/clients/vllm_raw_stream.py
"""
VLLMRawStream
=============
Drop-in replacement for OllamaNativeStream that targets vLLM's
/v1/completions endpoint (raw text, no OpenAI chat wrapper).

For TEXT-ONLY requests the pipeline is unchanged:
    vLLM /v1/completions  (prompt string, chat template applied here)

For MULTIMODAL requests (any message has list content with image blocks)
the pipeline automatically upgrades to:
    vLLM /v1/chat/completions  (messages array, vLLM handles the template)

For SOVEREIGN FORGE deployments (Ray Serve, URL contains /vllm_dep_):
    Text-only:   POST prompt string directly to the deployment URL.
    Multimodal:  POST raw hydrated messages directly to the deployment URL.
                 _normalise_for_chat is intentionally bypassed —
                 inference_worker._build_engine_input expects the internal
                 {type:image, image:data:...} format and handles PIL extraction
                 and HF tokenizer chat template application itself.
                 _http_stream is used (not _http_stream_chat) because
                 inference_worker.__call__ always returns SSE in completions
                 format (choices[0].text) regardless of input format.

Supported model families (CHAT_TEMPLATE_REGISTRY):
    Qwen, DeepSeek, Mistral, Llama, Phi, Gemma, GPT-OSS
    Unknown models fall back to Qwen format.
"""

from __future__ import annotations

import json
import os
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from projectdavid_common.utilities.logging_service import LoggingUtility

from src.api.entities_api.clients.renderers.deepseek import _render_deepseek
from src.api.entities_api.clients.renderers.gemma import _render_gemma
from src.api.entities_api.clients.renderers.gpt_oss import _render_gpt_oss
from src.api.entities_api.clients.renderers.llama3 import _render_llama3
from src.api.entities_api.clients.renderers.mistral import _render_mistral
from src.api.entities_api.clients.renderers.moonshot import _render_moonshot
from src.api.entities_api.clients.renderers.phi import _render_phi
from src.api.entities_api.clients.renderers.qwen import _render_qwen

load_dotenv()
LOG = LoggingUtility()

# Marker that identifies a Sovereign Forge / Ray Serve deployment URL.
# When present in resolved_base, POST directly to the URL — no path appended.
_SOVEREIGN_FORGE_URL_MARKER = "/vllm_dep_"


# ── Multimodal detection ──────────────────────────────────────────────────────


def _is_multimodal(messages: List[Dict]) -> bool:
    """
    Return True if any message carries a list content payload (i.e. a
    Qwen/OpenAI multimodal content array with image blocks).

    Plain-text messages always have str content; multimodal messages
    have list content after hydration.
    """
    return any(isinstance(m.get("content"), list) for m in messages)


# ── Per-family chat templates (text-only path) ────────────────────────────────

CHAT_TEMPLATE_REGISTRY = [
    ("Qwen", _render_qwen),
    ("qwen", _render_qwen),
    ("DeepSeek", _render_deepseek),
    ("deepseek", _render_deepseek),
    ("Mistral", _render_mistral),
    ("mistral", _render_mistral),
    ("Llama", _render_llama3),
    ("llama", _render_llama3),
    ("Phi", _render_phi),
    ("phi", _render_phi),
    ("Gemma", _render_gemma),
    ("gemma", _render_gemma),
    ("gpt", _render_gpt_oss),
    ("GPT", _render_gpt_oss),
    ("cerebras", _render_gpt_oss),
    ("gpt-j", _render_gpt_oss),
    ("gpt-neo", _render_gpt_oss),
    ("falcon", _render_gpt_oss),
    ("Falcon", _render_gpt_oss),
    ("moonshot", _render_moonshot),
    ("Moonshot", _render_moonshot),
    ("kimi", _render_moonshot),
    ("Kimi", _render_moonshot),
]


def render_prompt(
    model_id: str,
    messages: List[Dict],
    tools: Optional[List] = None,
) -> str:
    """
    Resolve and apply the correct chat template for a given model ID.
    Falls back to Qwen format for unknown model families.
    """
    for substr, renderer in CHAT_TEMPLATE_REGISTRY:
        if substr in model_id:
            return renderer(messages, tools)

    LOG.warning(
        "VLLMRawStream: no chat template for model '%s', falling back to Qwen.",
        model_id,
    )
    return _render_qwen(messages, tools)


# ── Multimodal message normalisation (chat/completions path) ──────────────────


def _normalise_for_chat(messages: List[Dict]) -> List[Dict]:
    """
    Convert hydrated messages into the OpenAI multimodal chat format that
    vLLM's /v1/chat/completions endpoint expects.

    NOTE: Do NOT use this for Sovereign Forge dispatch. inference_worker.py's
    _build_engine_input expects the internal {type:image, image:data:...}
    format and handles PIL extraction and HF tokenizer template application
    itself. Normalising to image_url format before Sovereign Forge dispatch
    causes _build_engine_input to silently skip all image blocks.
    """
    normalised = []
    for m in messages:
        content = m.get("content")

        if not isinstance(content, list):
            normalised.append(m)
            continue

        converted_blocks = []
        for block in content:
            if not isinstance(block, dict):
                continue

            btype = block.get("type")

            if btype == "text":
                converted_blocks.append({"type": "text", "text": block.get("text", "")})

            elif btype == "image":
                data_uri = block.get("image", "")
                converted_blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                    }
                )

            elif btype == "image_url":
                converted_blocks.append(block)

            else:
                LOG.warning(
                    "_normalise_for_chat: unknown block type '%s', skipping.", btype
                )

        normalised.append({**m, "content": converted_blocks})

    return normalised


# ══════════════════════════════════════════════════════════════════════════════
# VLLMRawStream
# ══════════════════════════════════════════════════════════════════════════════


class VLLMRawStream:
    """
    Mixin / base class that provides _stream_vllm_raw().

    Routing logic (in priority order):
        1. Sovereign Forge (URL contains /vllm_dep_):
               text-only  → render_prompt → POST prompt string to deployment URL
               multimodal → POST raw hydrated messages to deployment URL
                            _build_engine_input in inference_worker handles
                            PIL extraction and HF tokenizer template application.
                            _http_stream is used (not _http_stream_chat) because
                            inference_worker.__call__ always returns SSE in
                            completions format (choices[0].text).
        2. Multimodal      → _normalise_for_chat → /v1/chat/completions
        3. Text-only       → render_prompt → /v1/completions
    """

    VLLM_DEFAULT_BASE_URL: str = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
    VLLM_REQUEST_TIMEOUT: int = int(os.getenv("VLLM_TIMEOUT", "120"))

    async def _stream_vllm_raw(
        self,
        *,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.6,
        max_tokens: int = 1024,
        think: bool = False,
        base_url: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        skip_special_tokens: bool = False,
        **kwargs,
    ) -> AsyncGenerator[Dict[str, Any], None]:

        resolved_base = (base_url or self.VLLM_DEFAULT_BASE_URL).rstrip("/")

        # ── Sovereign Forge routing (Ray Serve deployment URL) ────────────
        if _SOVEREIGN_FORGE_URL_MARKER in resolved_base:
            async for chunk in self._stream_sovereign_forge(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                base_url=resolved_base,
            ):
                yield chunk
            return

        # ── Route decision ────────────────────────────────────────────────
        multimodal = _is_multimodal(messages)

        if multimodal:
            async for chunk in self._stream_vllm_chat(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                base_url=resolved_base,
                tools=tools,
            ):
                yield chunk
        else:
            async for chunk in self._stream_vllm_completions(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                think=think,
                base_url=resolved_base,
                tools=tools,
                skip_special_tokens=skip_special_tokens,
            ):
                yield chunk

    # ── SOVEREIGN FORGE path: POST directly to Ray Serve deployment URL ───────

    async def _stream_sovereign_forge(
        self,
        *,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
        base_url: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Post directly to a Ray Serve deployment URL.

        The deployment URL IS the endpoint — no path appended.
        Ray Serve routes the request to the VLLMDeployment.__call__ handler.

        Text-only:
            render_prompt → flat prompt string → choices[0].text SSE deltas.

        Multimodal:
            Raw hydrated messages posted directly — _normalise_for_chat is
            intentionally NOT called. inference_worker._build_engine_input
            expects the internal {type:image, image:data:...} format and handles
            PIL extraction and HF tokenizer chat template application itself.
            _http_stream is used (not _http_stream_chat) because
            inference_worker.__call__ always returns SSE in completions format
            (choices[0].text) regardless of whether input was prompt or messages.
        """
        if _is_multimodal(messages):
            payload: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
                "stream_options": {"include_usage": False},
            }
            LOG.info(
                "VLLMRawStream ▸ Sovereign Forge MULTIMODAL POST %s | model=%s | max_tokens=%d",
                base_url,
                model,
                max_tokens,
            )
            async for chunk in self._http_stream(base_url, payload):
                yield chunk
        else:
            prompt = render_prompt(model_id=model, messages=messages)
            payload = {
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
                "stream_options": {"include_usage": False},
            }
            LOG.info(
                "VLLMRawStream ▸ Sovereign Forge POST %s | model=%s | max_tokens=%d",
                base_url,
                model,
                max_tokens,
            )
            async for chunk in self._http_stream(base_url, payload):
                yield chunk

    # ── TEXT-ONLY path: /v1/completions ──────────────────────────────────────

    async def _stream_vllm_completions(
        self,
        *,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
        think: bool,
        base_url: str,
        tools: Optional[List[Dict]],
        skip_special_tokens: bool,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        endpoint = f"{base_url}/v1/completions"

        if tools is None and hasattr(self, "assistant_config"):
            tools = self.assistant_config.get("tools") or self.assistant_config.get(
                "function_definitions"
            )

        prompt = render_prompt(model_id=model, messages=messages, tools=tools)
        LOG.debug("VLLMRawStream ▸ completions prompt (%d chars)", len(prompt))

        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "skip_special_tokens": skip_special_tokens,
            "stream_options": {"include_usage": False},
        }

        if not think:
            stop = ["<think>"] if "Qwen3" in model or "qwen3" in model else None
            if stop:
                payload["stop"] = stop

        LOG.info(
            "VLLMRawStream ▸ POST %s | model=%s | max_tokens=%d",
            endpoint,
            model,
            max_tokens,
        )

        async for chunk in self._http_stream(endpoint, payload):
            yield chunk

    # ── MULTIMODAL path: /v1/chat/completions (holding pattern) ──────────────
    # See WP-sovereign-multimodal-pipeline.md for planned replacement.

    async def _stream_vllm_chat(
        self,
        *,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
        base_url: str,
        tools: Optional[List[Dict]],
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Route multimodal requests through /v1/chat/completions.
        Holding pattern — will be replaced by sovereign raw pipeline.
        See: WP-sovereign-multimodal-pipeline.md
        """
        endpoint = f"{base_url}/v1/chat/completions"

        normalised_messages = _normalise_for_chat(messages)

        payload: Dict[str, Any] = {
            "model": model,
            "messages": normalised_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": False},
        }

        LOG.info(
            "VLLMRawStream ▸ MULTIMODAL POST %s | model=%s | messages=%d | max_tokens=%d",
            endpoint,
            model,
            len(normalised_messages),
            max_tokens,
        )

        async for chunk in self._http_stream_chat(endpoint, payload):
            yield chunk

    # ── Shared HTTP streaming helpers ─────────────────────────────────────────

    async def _http_stream(
        self, endpoint: str, payload: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        POST `payload` to `endpoint`, stream SSE lines.
        Adapts /v1/completions  choices[0].text
             → DeltaNormalizer  choices[0].delta.content
        """
        try:
            async with httpx.AsyncClient(timeout=self.VLLM_REQUEST_TIMEOUT) as client:
                async with client.stream(
                    "POST",
                    endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as response:

                    if response.status_code != 200:
                        body = await response.aread()
                        LOG.error(
                            "VLLMRawStream ▸ HTTP %d: %s",
                            response.status_code,
                            body.decode(errors="replace")[:300],
                        )
                        yield {
                            "choices": [
                                {
                                    "delta": {
                                        "content": f"[vLLM error {response.status_code}]"
                                    },
                                    "finish_reason": "error",
                                }
                            ]
                        }
                        return

                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if raw == "[DONE]":
                            yield {
                                "done": True,
                                "done_reason": "stop",
                                "message": {"content": ""},
                            }
                            return
                        try:
                            parsed = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        choices = parsed.get("choices", [])
                        if not choices:
                            continue
                        choice = choices[0]
                        yield {
                            "choices": [
                                {
                                    "delta": {"content": choice.get("text", "")},
                                    "finish_reason": choice.get("finish_reason"),
                                }
                            ]
                        }

        except httpx.ConnectError as exc:
            LOG.error("VLLMRawStream ▸ connect error: %s", exc)
            yield {
                "choices": [
                    {
                        "delta": {"content": "[vLLM connection failed]"},
                        "finish_reason": "error",
                    }
                ]
            }
        except httpx.TimeoutException as exc:
            LOG.error("VLLMRawStream ▸ timeout: %s", exc)
            yield {
                "choices": [
                    {"delta": {"content": "[vLLM timeout]"}, "finish_reason": "error"}
                ]
            }
        except Exception as exc:
            LOG.error("VLLMRawStream ▸ unexpected: %s", exc, exc_info=True)
            yield {
                "choices": [
                    {
                        "delta": {"content": f"[vLLM stream error: {exc}]"},
                        "finish_reason": "error",
                    }
                ]
            }

    async def _http_stream_chat(
        self, endpoint: str, payload: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        POST `payload` to `endpoint`, stream SSE lines.
        /v1/chat/completions already returns choices[0].delta.content —
        pass through unchanged so DeltaNormalizer sees the same shape.
        Used by the standard multimodal path (_stream_vllm_chat) only.
        Sovereign Forge always uses _http_stream regardless of modality.
        """
        try:
            async with httpx.AsyncClient(timeout=self.VLLM_REQUEST_TIMEOUT) as client:
                async with client.stream(
                    "POST",
                    endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as response:

                    if response.status_code != 200:
                        body = await response.aread()
                        LOG.error(
                            "VLLMRawStream ▸ chat HTTP %d: %s",
                            response.status_code,
                            body.decode(errors="replace")[:300],
                        )
                        yield {
                            "choices": [
                                {
                                    "delta": {
                                        "content": f"[vLLM error {response.status_code}]"
                                    },
                                    "finish_reason": "error",
                                }
                            ]
                        }
                        return

                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if raw == "[DONE]":
                            yield {
                                "done": True,
                                "done_reason": "stop",
                                "message": {"content": ""},
                            }
                            return
                        try:
                            parsed = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        choices = parsed.get("choices", [])
                        if not choices:
                            continue
                        choice = choices[0]
                        yield {
                            "choices": [
                                {
                                    "delta": choice.get("delta", {"content": ""}),
                                    "finish_reason": choice.get("finish_reason"),
                                }
                            ]
                        }

        except httpx.ConnectError as exc:
            LOG.error("VLLMRawStream ▸ chat connect error: %s", exc)
            yield {
                "choices": [
                    {
                        "delta": {"content": "[vLLM connection failed]"},
                        "finish_reason": "error",
                    }
                ]
            }
        except httpx.TimeoutException as exc:
            LOG.error("VLLMRawStream ▸ chat timeout: %s", exc)
            yield {
                "choices": [
                    {"delta": {"content": "[vLLM timeout]"}, "finish_reason": "error"}
                ]
            }
        except Exception as exc:
            LOG.error("VLLMRawStream ▸ chat unexpected: %s", exc, exc_info=True)
            yield {
                "choices": [
                    {
                        "delta": {"content": f"[vLLM stream error: {exc}]"},
                        "finish_reason": "error",
                    }
                ]
            }
