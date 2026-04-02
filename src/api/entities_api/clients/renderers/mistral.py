import json
from typing import Dict, List, Optional


def _render_mistral(messages: List[Dict], tools: Optional[List] = None) -> str:
    """Mistral [INST] format."""
    system = ""
    turns = []

    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        else:
            turns.append(m)

    if tools:
        tool_json = json.dumps(tools)
        system += f"\n\nYou have access to the following tools:\n{tool_json}"

    parts = ["<s>"]
    for i, m in enumerate(turns):
        if m["role"] == "user":
            prefix = f"{system}\n\n" if system and i == 0 else ""
            parts.append(f"[INST] {prefix}{m['content']} [/INST]")
        elif m["role"] == "assistant":
            parts.append(f" {m['content']}</s>")

    return "".join(parts)
