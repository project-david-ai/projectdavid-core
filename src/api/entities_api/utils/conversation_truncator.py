# src/api/entities_api/services/conversation_truncator.py
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, List, Union

from projectdavid_common import LoggingUtility
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

LOG = LoggingUtility()
hf_logging.set_verbosity_error()


@lru_cache(maxsize=8)
def _load_tokenizer(model_name: str):
    """Load and cache tokenizer — one load per model name per process."""
    #  — model name is caller-controlled, revision pinning not applicable
    return AutoTokenizer.from_pretrained(model_name)  # nosec B615


def _extract_text(content: Any) -> str:
    """
    Safely extract a plain string from any message content for tokenization.

    - str  → returned as-is
    - list → Qwen/multimodal content array: join all 'text' block values
    - None → empty string

    IMPORTANT: this is used ONLY for token counting.  The original content
    structure is never modified — multimodal list payloads must reach vLLM
    intact.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    # Fallback for any unexpected type
    return str(content)


def _merge_content(
    existing: Union[str, list], incoming: Union[str, list]
) -> Union[str, list]:
    """
    Merge two message contents that share the same role.

    Rules:
    - str  + str  → concatenate with newline (original behaviour)
    - list + list → concatenate the block arrays
    - list + str  → append the str as a text block to the list
    - str  + list → promote str to a text block, then append the list blocks
    """
    if isinstance(existing, list) and isinstance(incoming, list):
        return existing + incoming

    if isinstance(existing, list):
        # incoming is a plain string — wrap and append
        return existing + [{"type": "text", "text": incoming}]

    if isinstance(incoming, list):
        # existing is a plain string — promote, then extend
        return [{"type": "text", "text": existing}] + incoming

    # Both plain strings
    return existing + "\n" + incoming


class ConversationTruncator:
    """
    Utility to trim a conversation to <= `threshold_percentage`
    of `max_context_window` tokens (system messages are never dropped).
    Consecutive messages from the same role are merged.

    Multimodal messages (content is a list of typed blocks) are handled
    throughout:
      - Token counting extracts only text blocks for estimation.
      - Merging concatenates block arrays rather than string-concatenating.
      - The original list structure is never replaced with a plain string.
    """

    FALLBACK_MODEL = os.getenv("TRUNCATOR_MODEL", "gpt2")

    def __init__(
        self, model_name: str, max_context_window: int, threshold_percentage: float
    ) -> None:
        self.max_context_window = max_context_window
        self.threshold_percentage = threshold_percentage
        self.tokenizer = self._safe_load_tokenizer(model_name)

    @classmethod
    def _safe_load_tokenizer(cls, model_name: str):
        try:
            return _load_tokenizer(model_name)
        except Exception as exc:
            LOG.warning(
                "Tokenizer %s unavailable (%s) — falling back to %s",
                model_name,
                exc.__class__.__name__,
                cls.FALLBACK_MODEL,
            )
            return _load_tokenizer(cls.FALLBACK_MODEL)

    def _count_tokens_batch(self, texts: List[str]) -> List[int]:
        if not texts:
            return []
        # Guard: coerce any None that slipped through _extract_text to ""
        safe_texts = [t if isinstance(t, str) else "" for t in texts]
        encoded = self.tokenizer(
            safe_texts,
            add_special_tokens=False,
            return_attention_mask=False,
            return_length=True,
        )
        return encoded["length"]

    def count_tokens(self, text: str) -> int:
        """Return token count for a single string (special tokens excluded)."""
        return len(self.tokenizer.encode(text or "", add_special_tokens=False))

    def truncate(self, conversation: List[dict]) -> List[dict]:
        """
        Trim conversation and merge consecutive same-role messages.

        Multimodal messages are token-counted by extracting their text blocks;
        the full content structure is preserved in the returned messages.
        """
        system_msgs = [m for m in conversation if m.get("role") == "system"]
        other_msgs = [m for m in conversation if m.get("role") != "system"]

        # Extract plain text for tokenisation — never pass raw content lists
        all_texts = [_extract_text(m["content"]) for m in system_msgs + other_msgs]
        all_counts = self._count_tokens_batch(all_texts)

        sys_tokens = sum(all_counts[: len(system_msgs)])
        msg_counts = list(zip(other_msgs, all_counts[len(system_msgs) :]))
        oth_tokens = sum(c for _, c in msg_counts)

        threshold_tokens = self.max_context_window * self.threshold_percentage

        if sys_tokens + oth_tokens <= threshold_tokens:
            return self.merge_consecutive_messages(conversation)

        budget = threshold_tokens - sys_tokens
        while msg_counts and oth_tokens > budget:
            _, cost = msg_counts.pop(0)
            oth_tokens -= cost

        kept_others = [m for m, _ in msg_counts]

        # Restore original order using a pre-built id-index
        original_index = {id(m): i for i, m in enumerate(conversation)}
        combined = sorted(
            system_msgs + kept_others, key=lambda m: original_index[id(m)]
        )

        return self.merge_consecutive_messages(combined)

    @staticmethod
    def merge_consecutive_messages(conversation: List[dict]) -> List[dict]:
        """
        Merge back-to-back messages from the same role.

        Handles multimodal (list) content correctly via _merge_content().
        """
        if not conversation:
            return conversation

        merged = [dict(conversation[0])]  # shallow copy so we don't mutate original

        for msg in conversation[1:]:
            last = merged[-1]
            if msg["role"] == last["role"]:
                last["content"] = _merge_content(last["content"], msg["content"])
            else:
                merged.append(dict(msg))

        return merged
