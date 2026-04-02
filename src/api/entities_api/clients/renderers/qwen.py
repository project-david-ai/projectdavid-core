import json
from typing import Dict, List, Optional


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
        content = (
            m["content"] if isinstance(m["content"], str) else json.dumps(m["content"])
        )
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)
