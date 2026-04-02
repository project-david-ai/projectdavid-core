import json
from typing import Dict, List, Optional


def _render_phi(messages: List[Dict], tools: Optional[List] = None) -> str:
    """Microsoft Phi-3 / Phi-3.5 format using <|system|>, <|user|>, <|assistant|> tokens."""
    parts = []

    system_content = None
    filtered = []
    for m in messages:
        if m["role"] == "system":
            system_content = m["content"]
        else:
            filtered.append(m)

    if system_content:
        if tools:
            tool_json = "\n".join(json.dumps(t) for t in tools)
            system_content += (
                "\n\nYou have access to the following tools:\n"
                f"<tools>\n{tool_json}\n</tools>"
            )
        parts.append(f"<|system|>\n{system_content}<|end|>")
    elif tools:
        tool_json = "\n".join(json.dumps(t) for t in tools)
        parts.append(
            f"<|system|>\nYou are a helpful assistant.\n\nYou have access to the following tools:\n"
            f"<tools>\n{tool_json}\n</tools><|end|>"
        )

    for m in filtered:
        role = m["role"]
        content = (
            m["content"] if isinstance(m["content"], str) else json.dumps(m["content"])
        )
        if role == "user":
            parts.append(f"<|user|>\n{content}<|end|>")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{content}<|end|>")

    parts.append("<|assistant|>")
    return "\n".join(parts)
