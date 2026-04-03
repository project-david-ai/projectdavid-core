import json
from typing import Dict, List, Optional


def _render_gpt_oss(messages: List[Dict], tools: Optional[List] = None) -> str:
    """
    GPT-OSS format for open-weight GPT-style models (GPT-J, GPT-NeoX,
    GPT-2 fine-tunes, Cerebras-GPT, etc.).
    Uses ### Human / ### Assistant delimiters.
    """
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
        parts.append(system_content)

    for m in filtered:
        role = m["role"]
        content = (
            m["content"] if isinstance(m["content"], str) else json.dumps(m["content"])
        )
        if role == "user":
            parts.append(f"### Human: {content}")
        elif role == "assistant":
            parts.append(f"### Assistant: {content}")

    parts.append("### Assistant:")
    return "\n\n".join(parts)
