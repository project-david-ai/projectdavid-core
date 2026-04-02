import json
from typing import Dict, List, Optional


def _render_moonshot(messages: List[Dict], tools: Optional[List] = None) -> str:
    """Moonshot / Kimi chat format."""
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
                "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
                "You are provided with function signatures within <tools></tools> XML tags:\n"
                f"<tools>\n{tool_json}\n</tools>"
            )
        parts.append(f"<|system|>{system_content}")
    elif tools:
        tool_json = "\n".join(json.dumps(t) for t in tools)
        parts.append(
            f"<|system|>You are a helpful assistant.\n\n"
            f"You are provided with function signatures within <tools></tools> XML tags:\n"
            f"<tools>\n{tool_json}\n</tools>"
        )

    for m in filtered:
        role = m["role"]
        content = (
            m["content"] if isinstance(m["content"], str) else json.dumps(m["content"])
        )
        if role == "user":
            parts.append(f"<|user|>{content}")
        elif role == "assistant":
            parts.append(f"<|assistant|>{content}")

    parts.append("<|assistant|>")
    return "\n".join(parts)
