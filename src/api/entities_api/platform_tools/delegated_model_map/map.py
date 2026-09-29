# ---------------------------------------------------------
# 1. The Configuration Map
# ---------------------------------------------------------

# together-ai/deepseek-ai/DeepSeek-V3.1
# hyperbolic/Qwen/Qwen3-Coder-480B-A35B-Instruct
# together-ai/Qwen/Qwen3-Next-80B-A3B-Instruct-FP8
# together-ai/Qwen/Qwen3-VL-235B-A22B-Instruct-FP
# together-ai/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8
# together-ai/openai/gpt-oss-120b

DELEGATED_DEEP_SEARCH_MAP = {
    "together-ai": "together-ai/moonshotai/Kimi-K3",
    "hyperbolic": "hyperbolic/Qwen/Qwen3-Coder-480B-A35B-Instruct",
    "fireworks": "fireworks/accounts/fireworks/models/deepseek-r1",
    "openai": "openai/o1-preview",
    "anthropic": "anthropic/claude-3-5-sonnet-20241022",
    "default": "openai/gpt-4o",  # Fallback
}
