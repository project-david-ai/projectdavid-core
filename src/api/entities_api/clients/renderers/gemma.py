import json
from typing import Dict, List, Optional


def _render_gemma(messages: List[Dict], tools: Optional[List] = None) -> str:
    """Google Gemma 2 / Gemma 3 start_of_turn/end_of_turn format."""
    parts = []

    system_content = None
    filtered = []
    for m in messages:
        if m["role"] == "system":
            system_content = m["content"]
        else:
            filtered.append(m)

    # Gemma has no native system role — prepend system content to the first user turn.
    first_user_prefix = ""
    if system_content:
        if tools:
            tool_json = "\n".join(json.dumps(t) for t in tools)
            system_content += (
                "\n\nYou have access to the following tools:\n"
                f"<tools>\n{tool_json}\n</tools>"
            )
        first_user_prefix = f"{system_content}\n\n"
    elif tools:
        tool_json = "\n".join(json.dumps(t) for t in tools)
        first_user_prefix = (
            f"You have access to the following tools:\n"
            f"<tools>\n{tool_json}\n</tools>\n\n"
        )

    for i, m in enumerate(filtered):
        role = m["role"]
        content = (
            m["content"] if isinstance(m["content"], str) else json.dumps(m["content"])
        )
        if role == "user":
            prefix = first_user_prefix if i == 0 else ""
            parts.append(f"<start_of_turn>user\n{prefix}{content}<end_of_turn>")
        elif role == "assistant":
            parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")

    parts.append("<start_of_turn>model\n")
    return "\n".join(parts)
