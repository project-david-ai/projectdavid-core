import json
from typing import Dict, List, Optional


def _render_deepseek(messages: List[Dict], tools: Optional[List] = None) -> str:
    """DeepSeek V2/V3/R1 chat format with tool support."""
    parts = []

    system_content = None
    filtered = []
    for m in messages:
        if m["role"] == "system":
            system_content = m["content"]
        else:
            filtered.append(m)

    if tools:
        tool_json = "\n".join(json.dumps(t) for t in tools)
        system_block = system_content or "You are a helpful assistant."
        system_block += (
            "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            f"<tools>\n{tool_json}\n</tools>"
        )
        parts.append(system_block)
    elif system_content:
        parts.append(system_content)

    for m in filtered:
        role = m["role"]
        content = (
            m["content"] if isinstance(m["content"], str) else json.dumps(m["content"])
        )
        if role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")

    parts.append("Assistant:")
    return "\n\n".join(parts)
