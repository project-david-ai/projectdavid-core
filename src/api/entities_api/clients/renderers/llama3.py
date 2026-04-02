import json
from typing import Dict, List, Optional


def _render_llama3(messages: List[Dict], tools: Optional[List] = None) -> str:
    """Llama 3.x header/eot format."""
    parts = ["<|begin_of_text|>"]
    system_injected = False

    for m in messages:
        role = m["role"]
        content = (
            m["content"] if isinstance(m["content"], str) else json.dumps(m["content"])
        )

        if role == "system" and tools and not system_injected:
            tool_json = json.dumps(tools)
            content += f"\n\nTools available:\n{tool_json}"
            system_injected = True

        parts.append(
            f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        )

    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(parts)
