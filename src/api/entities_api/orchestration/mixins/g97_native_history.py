"""Adapt saved tool transcripts for Together Bonsai's native API."""

from __future__ import annotations

import json
import re
from typing import Any

_LEGACY_HEADING = re.compile(
    r"(?ms)^\s*\U0001f539\s*\*\*(?:STRICT TOOL USAGE PROTOCOL|"
    r"FORMATTING FUNCTION CALLS|FUNCTION CALL WRAPPING)\*\*.*?"
    r"(?=^\s*\U0001f539\s*\*\*|^\s*###\s|\Z)"
)


def prepare_bonsai_native_history(
    messages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Preserve native tool batches and adapt legacy results outside them."""
    outgoing: list[dict[str, Any]] = []
    pending: set[str] = set()

    for original in messages:
        msg = dict(original)
        role = msg.get("role")
        if pending:
            call_id = msg.get("tool_call_id")
            if role != "tool" or not isinstance(call_id, str) or call_id not in pending:
                raise ValueError("G97_INCOMPLETE_NATIVE_HISTORY_MISSING_TOOL_OUTPUT")
            pending.remove(call_id)
            msg.pop("name", None)
            outgoing.append(msg)
            continue

        if role == "system" and isinstance(msg.get("content"), str):
            msg["content"] = _LEGACY_HEADING.sub("\n", msg["content"]) + (
                "\n\n[G97 NATIVE TOOL MODE] Use ONLY the tools in the native API "
                "tools parameter to invoke a function. Never write tool-call "
                "JSON or <fc> blocks in message content."
            )

        if role == "assistant" and msg.get("tool_calls"):
            raw_calls = msg["tool_calls"]
            if not isinstance(raw_calls, list):
                raise ValueError("G97_MALFORMED_NATIVE_HISTORY")
            for item in raw_calls:
                if not isinstance(item, dict):
                    raise ValueError("G97_MALFORMED_NATIVE_HISTORY")
                call_id = item.get("id")
                function = item.get("function")
                if (
                    not isinstance(call_id, str)
                    or not call_id
                    or not isinstance(function, dict)
                ):
                    raise ValueError("G97_MALFORMED_NATIVE_HISTORY")
                if not isinstance(function.get("name"), str):
                    raise ValueError("G97_MALFORMED_NATIVE_HISTORY")
                args = function.get("arguments")
                if not isinstance(args, str) or not isinstance(json.loads(args), dict):
                    raise ValueError("G97_MALFORMED_NATIVE_HISTORY")
                if call_id in pending:
                    raise ValueError("G97_DUPLICATE_NATIVE_TOOL_IDS")
                pending.add(call_id)
        elif role == "tool":
            msg = {
                "role": "user",
                "content": "Legacy tool result (unlinked):\n"
                + str(msg.get("content") or ""),
            }

        outgoing.append(msg)

    if pending:
        raise ValueError("G97_INCOMPLETE_NATIVE_HISTORY_AT_END")
    return outgoing


def to_native_assistant_envelope(calls: list[dict[str, Any]]) -> str:
    """Persist the exact dispatched IDs in MessageService's recognised form."""
    if not calls:
        raise ValueError("G97_NO_NATIVE_CALLS")
    output = []
    ids = set()
    for call in calls:
        call_id, name, arguments = (
            call.get("id"),
            call.get("name"),
            call.get("arguments"),
        )
        if not isinstance(call_id, str) or not call_id or call_id in ids:
            raise ValueError("G97_INVALID_OR_DUPLICATE_NATIVE_CALL_ID")
        if not isinstance(name, str) or not name or not isinstance(arguments, dict):
            raise ValueError("G97_INVALID_NATIVE_CALL")
        ids.add(call_id)
        output.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(
                        arguments, ensure_ascii=False, allow_nan=False
                    ),
                },
            }
        )
    return json.dumps(output, ensure_ascii=False)
