from __future__ import annotations

from datetime import datetime, timezone

CAPABILITY_KEY = "_q_local_model_capabilities"


def resolve_q_tool_mode(run_metadata, requested_model):
    if not isinstance(run_metadata, dict):
        return False

    if CAPABILITY_KEY not in run_metadata:
        return None

    policy = run_metadata[CAPABILITY_KEY]

    if not isinstance(policy, dict):
        return False

    if not isinstance(requested_model, str) or not requested_model.startswith("vllm/"):
        return False

    variant = requested_model[len("vllm/") :]

    if (
        not variant
        or policy.get("model") != requested_model
        or policy.get("variant_id") != variant
    ):
        return False

    return (
        policy.get("tools_enabled") is True
        and policy.get("tool_calling_status") == "verified"
    )


def build_tool_free_system_message(config):
    if not isinstance(config, dict):
        config = {}

    instructions = config.get("instructions") or ""

    if not isinstance(instructions, str):
        instructions = str(instructions)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    return {
        "role": "system",
        "content": (
            f"Today's date and time: {today}\n\n"
            "### ASSISTANT INSTRUCTIONS\n"
            f"{instructions}\n\n"
            "External tools are unavailable for this inference. "
            "Respond directly without attempting tool calls."
        ),
    }
