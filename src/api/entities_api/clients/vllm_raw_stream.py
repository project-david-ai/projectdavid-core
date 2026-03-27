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

This is required because /v1/completions is a text-only endpoint —
serialising base64 image arrays as JSON strings into a prompt string
tokenises them as raw text (thousands of tokens) and loses the vision
signal entirely.  /v1/chat/completions passes the typed content blocks
natively to the model's multimodal processor.
"""

from __future__ import annotations

import json
import os
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from projectdavid_common.utilities.logging_service import LoggingUtility

load_dotenv()
LOG = LoggingUtility()


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


def _render_qwen(messages: List[Dict], tools: Optional[List] = None) -> str:
    """Qwen2.5 / Qwen3 im_start/im_end format — TEXT ONLY."""
    parts = []

    if tools:
        tool_json = "\n".join(json.dumps(t) for t in tools)
        system_content = None
        filtered = []
        for m in messages:
            if m["role"] == "system":
                system_content = m["content"]
            else:
                filtered.append(m)

        system_block = system_content or "You are a helpful assistant."
        system_block += (
            "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            f"<tools>\n{tool_json}\n</tools>"
        )
        parts.append(f"<|im_start|>system\n{system_block}<|im_end|>")
        messages = filtered
    else:
        for m in messages:
            if m["role"] == "system":
                parts.append(f"<|im_start|>system\n{m['content']}<|im_end|>")
                messages = [x for x in messages if x is not m]
                break

    for m in messages:
        role = m["role"]
        # TEXT ONLY — list content must never reach this renderer
        content = (
            m["content"] if isinstance(m["content"], str) else json.dumps(m["content"])
        )
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def _render_mistral(messages: List[Dict], tools: Optional[List] = None) -> str:
    """Mistral [INST] format."""
    system = ""
    turns = []

    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        else:
            turns.append(m)

    if tools:
        tool_json = json.dumps(tools)
        system += f"\n\nYou have access to the following tools:\n{tool_json}"

    parts = ["<s>"]
    for i, m in enumerate(turns):
        if m["role"] == "user":
            prefix = f"{system}\n\n" if system and i == 0 else ""
            parts.append(f"[INST] {prefix}{m['content']} [/INST]")
        elif m["role"] == "assistant":
            parts.append(f" {m['content']}</s>")

    return "".join(parts)


def _render_llama3(messages: List[Dict], tools: Optional[List] = None) -> str:
    """Llama 3.x header/eot format."""
    parts = ["<|begin_of_text|>"]
    system_injected = False

    for m in messages:
        role = m["role"]
        content = (
            m["content"] if isinstance(m["content"], str) else json.dumps(m["content"])
        )

        if role == "system" and tools and not system_injected:
            tool_json = json.dumps(tools)
            content += f"\n\nTools available:\n{tool_json}"
            system_injected = True

        parts.append(
            f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        )

    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(parts)


CHAT_TEMPLATE_REGISTRY = [
    ("Qwen", _render_qwen),
    ("qwen", _render_qwen),
    ("Mistral", _render_mistral),
    ("mistral", _render_mistral),
    ("Llama", _render_llama3),
    ("llama", _render_llama3),
]


def render_prompt(
    model_id: str,
    messages: List[Dict],
    tools: Optional[List] = None,
) -> str:
    """
    Resolve and apply the correct chat template for a given model ID.
    Falls back to Qwen format.  Called only on the TEXT-ONLY path.
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

    Hydrated image blocks arrive as:
        {"type": "image", "image": "data:image/jpeg;base64,<b64>"}

    OpenAI / vLLM chat format expects:
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<b64>"}}

    Plain text content strings are left unchanged.
    """
    normalised = []
    for m in messages:
        content = m.get("content")

        if not isinstance(content, list):
            # Plain text — pass straight through
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
                # Hydrated format → OpenAI image_url format
                data_uri = block.get("image", "")
                converted_blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                    }
                )

            elif btype == "image_url":
                # Already in the right format — pass through
                converted_blocks.append(block)

            else:
                # Unknown block type — skip
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

    Routing logic:
        • Text-only  → /v1/completions   (prompt string, chat template rendered here)
        • Multimodal → /v1/chat/completions  (messages array, vLLM owns the template)
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

    # ── MULTIMODAL path: /v1/chat/completions ────────────────────────────────

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

        vLLM applies the model's chat template and multimodal processor
        internally — we send the messages array with properly formatted
        image_url blocks and receive delta.content chunks back, which is
        already the shape DeltaNormalizer expects.
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

        # /v1/chat/completions already returns delta.content — no re-wrapping needed
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
                        # delta.content already present — pass straight through
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
