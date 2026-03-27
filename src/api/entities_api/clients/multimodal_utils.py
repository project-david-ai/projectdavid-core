# src/api/entities_api/clients/multimodal_utils.py
"""
Shared multimodal utilities for OpenAI-compatible providers.

Used by:
    - vllm_raw_stream.py   (vLLM /v1/chat/completions path)
    - qwen_base.py         (TogetherAI, Hyperbolic, etc.)
    - deepseek_worker.py   (Hyperbolic DeepSeek/VL models)
    - default_worker.py    (NVIDIA, etc.)
    - gpt_oss_worker.py    (GPT-OSS providers)

NOT used by:
    - ollama_client.py     (Ollama has its own format: images as a sibling
                            key with raw base64 — see _normalise_for_ollama)

The functions here produce the standard OpenAI multimodal message format:
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}

This is the format accepted by vLLM, TogetherAI, Hyperbolic, and any other
provider that follows the OpenAI vision spec.

Provider image limits (enforced via max_images parameter):
    - TogetherAI  : unlimited (default)
    - vLLM local  : unlimited (default)
    - Ollama      : 1 per message (enforced in ollama_client._normalise_for_ollama)
    - Hyperbolic  : 1 per request (pass max_images=1 in deepseek_worker)
"""

from __future__ import annotations

from typing import Dict, List, Optional

from projectdavid_common.utilities.logging_service import LoggingUtility

LOG = LoggingUtility()


def is_multimodal(messages: List[Dict]) -> bool:
    """
    Return True if any message carries list content (hydrated image blocks).

    Plain-text messages always have str content.  After hydration, messages
    with image attachments have list content containing typed blocks.
    """
    return any(isinstance(m.get("content"), list) for m in messages)


def normalise_for_chat(
    messages: List[Dict],
    max_images: Optional[int] = None,
) -> List[Dict]:
    """
    Convert hydrated messages into the OpenAI multimodal chat format.

    Accepts the internal hydrated format produced by NativeExecutionService
    .hydrate_messages() and converts it to the OpenAI vision spec that vLLM,
    TogetherAI, Hyperbolic, and compatible providers accept.

    Hydrated input:
        {"role": "user", "content": [
            {"type": "text",  "text":  "What is in this image?"},
            {"type": "image", "image": "data:image/jpeg;base64,/9j/..."},
        ]}

    OpenAI output:
        {"role": "user", "content": [
            {"type": "text",     "text": "What is in this image?"},
            {"type": "image_url","image_url": {"url": "data:image/jpeg;base64,/9j/..."}},
        ]}

    Plain text messages (str content) pass through untouched.
    Already-normalised image_url blocks also pass through unchanged.

    Args:
        messages:   The context window to normalise.
        max_images: Optional hard cap on the total number of image blocks
                    included across ALL messages in the context. Excess
                    images are dropped with a warning. Use this for
                    providers with strict per-request image limits:
                        Hyperbolic → max_images=1
                        Ollama     → enforced separately in ollama_client
                        All others → None (unlimited, default)
    """
    normalised = []
    images_included = 0

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
                # Internal hydrated format → OpenAI image_url format
                if max_images is not None and images_included >= max_images:
                    LOG.warning(
                        "normalise_for_chat: max_images=%d reached — dropping excess image "
                        "block. This provider only supports %d image(s) per request.",
                        max_images,
                        max_images,
                    )
                    continue

                data_uri = block.get("image", "")
                converted_blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                    }
                )
                images_included += 1

            elif btype == "image_url":
                # Already in OpenAI format — still respect max_images cap
                if max_images is not None and images_included >= max_images:
                    LOG.warning(
                        "normalise_for_chat: max_images=%d reached — dropping excess "
                        "image_url block.",
                        max_images,
                    )
                    continue

                converted_blocks.append(block)
                images_included += 1

            else:
                LOG.warning(
                    "normalise_for_chat: unknown block type '%s', skipping.", btype
                )

        normalised.append({**m, "content": converted_blocks})

    if max_images is not None:
        LOG.info(
            "normalise_for_chat: %d/%d image(s) included in normalised context.",
            images_included,
            max_images,
        )

    return normalised
