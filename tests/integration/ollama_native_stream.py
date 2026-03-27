"""
ollama_native_stream.py
────────────────────────────────────────────────────────────────────────────
Standalone async streaming method for the Ollama native /api/chat endpoint.

Drop this into OllamaDefaultBaseWorker (or use standalone for testing).

Ollama chunk shape:
  {
    "model": "qwen3:4b",
    "created_at": "...",
    "message": {
      "role": "assistant",
      "content": "",          # ← visible reply delta
      "thinking": "Okay"      # ← reasoning delta (only on thinking models)
    },
    "done": false
  }

  Final chunk has "done": true and a usage summary.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, AsyncGenerator, Dict, List

import httpx
from projectdavid_common.utilities.logging_service import LoggingUtility

LOG = LoggingUtility()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


# ─────────────────────────────────────────────────────────────────────────────
# Standalone streaming function (no class needed for testing)
# ─────────────────────────────────────────────────────────────────────────────


async def stream_ollama_native(
    messages: List[Dict[str, str]],
    model: str = "qwen3:4b",
    *,
    base_url: str = OLLAMA_BASE_URL,
    temperature: float = 0.6,
    max_tokens: int = 10_000,
    run_id: str = "local-test",
    stream_thinking: bool = True,
) -> AsyncGenerator[str, None]:
    """
    Streams directly from the Ollama /api/chat endpoint.

    Yields JSON strings with the shape our StreamState / downstream consumers expect:

      {"type": "thinking",  "content": "<delta>",  "run_id": "<run_id>"}
      {"type": "text",      "content": "<delta>",  "run_id": "<run_id>"}
      {"type": "status",    "status":  "complete", "run_id": "<run_id>"}
      {"type": "error",     "content": "<msg>",    "run_id": "<run_id>"}
    """

    url = f"{base_url}/api/chat"

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    accumulated_thinking = ""
    accumulated_content = ""

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, json=payload) as response:

                if response.status_code != 200:
                    body = await response.aread()
                    err_msg = (
                        f"Ollama returned HTTP {response.status_code}: {body.decode()}"
                    )
                    LOG.error(err_msg)
                    yield json.dumps(
                        {"type": "error", "content": err_msg, "run_id": run_id}
                    )
                    return

                async for raw_line in response.aiter_lines():
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue

                    try:
                        chunk = json.loads(raw_line)
                    except json.JSONDecodeError as e:
                        LOG.warning(
                            "Ollama: malformed JSON line skipped (%s): %r", e, raw_line
                        )
                        continue

                    # ── Final done chunk ──────────────────────────────────
                    if chunk.get("done"):
                        usage = {
                            "prompt_tokens": chunk.get("prompt_eval_count", 0),
                            "completion_tokens": chunk.get("eval_count", 0),
                        }
                        LOG.info(
                            "Ollama stream complete. model=%s usage=%s", model, usage
                        )
                        yield json.dumps(
                            {
                                "type": "status",
                                "status": "complete",
                                "run_id": run_id,
                                "usage": usage,
                            }
                        )
                        return

                    message = chunk.get("message", {})

                    # ── Thinking delta ────────────────────────────────────
                    thinking_delta: str = message.get("thinking") or ""
                    if thinking_delta and stream_thinking:
                        accumulated_thinking += thinking_delta
                        yield json.dumps(
                            {
                                "type": "thinking",
                                "content": thinking_delta,
                                "run_id": run_id,
                            }
                        )

                    # ── Visible content delta ─────────────────────────────
                    content_delta: str = message.get("content") or ""
                    if content_delta:
                        accumulated_content += content_delta
                        yield json.dumps(
                            {
                                "type": "text",
                                "content": content_delta,
                                "run_id": run_id,
                            }
                        )

    except httpx.ConnectError as e:
        msg = f"Cannot reach Ollama at {base_url} — is it running? ({e})"
        LOG.error(msg)
        yield json.dumps({"type": "error", "content": msg, "run_id": run_id})

    except Exception as exc:
        LOG.error("Ollama native stream error: %s", exc, exc_info=True)
        yield json.dumps({"type": "error", "content": str(exc), "run_id": run_id})


# ─────────────────────────────────────────────────────────────────────────────
# Mixin — drop into OllamaDefaultBaseWorker
# ─────────────────────────────────────────────────────────────────────────────


class OllamaNativeStreamMixin:
    """
    Mixin that replaces the homebrew client streaming call with a direct
    Ollama /api/chat POST.

    Usage inside OllamaDefaultBaseWorker.stream():

        async for chunk_str in self._stream_ollama_native(ctx, model, run_id=run_id):
            chunk = json.loads(chunk_str)
            if chunk.get("type") == "error":
                ...
            yield chunk_str

    Or swap the existing raw_stream block wholesale — see _replace_raw_stream().
    """

    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    async def _stream_ollama_native(
        self,
        messages: List[Dict],
        model: str,
        *,
        run_id: str,
        temperature: float = 0.6,
        max_tokens: int = 10_000,
        stream_thinking: bool = True,
    ) -> AsyncGenerator[str, None]:
        """
        Direct replacement for:

            raw_stream_sync = client.stream_chat_completion(...)
            async for chunk in DeltaNormalizer.async_iter_deltas(
                _iter_sync_stream(raw_stream_sync), run_id
            ):

        Just do:

            async for chunk_str in self._stream_ollama_native(ctx, model, run_id=run_id):
                chunk = json.loads(chunk_str)
                ...
        """
        async for item in stream_ollama_native(
            messages=messages,
            model=model,
            base_url=getattr(self, "OLLAMA_BASE_URL", OLLAMA_BASE_URL),
            temperature=temperature,
            max_tokens=max_tokens,
            run_id=run_id,
            stream_thinking=stream_thinking,
        ):
            yield item


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test  —  run with:  python ollama_native_stream.py
# ─────────────────────────────────────────────────────────────────────────────


async def _smoke_test():
    messages = [{"role": "user", "content": "Why is the sky blue? Be brief."}]

    print("── Streaming from Ollama ──")
    async for raw in stream_ollama_native(messages, model="qwen3:4b"):
        chunk = json.loads(raw)
        ctype = chunk.get("type")

        if ctype == "thinking":
            print(f"[THINK] {chunk['content']}", end="", flush=True)
        elif ctype == "text":
            print(chunk["content"], end="", flush=True)
        elif ctype == "status":
            print(f"\n\n[DONE] usage={chunk.get('usage')}")
        elif ctype == "error":
            print(f"\n[ERROR] {chunk['content']}")


if __name__ == "__main__":
    asyncio.run(_smoke_test())
