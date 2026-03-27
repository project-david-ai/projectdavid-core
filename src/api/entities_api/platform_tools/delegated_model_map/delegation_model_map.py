from entities_api.platform_tools.delegated_model_map.deep_search import (
    DELEGATED_DEEP_SEARCH_MAP,
)


# ---------------------------------------------------------
# 2. The Logic (The method you will add to your class)
# ---------------------------------------------------------
def get_delegated_model(requested_model: str) -> str:
    """
    Splits the requested model string to find the provider,
    then returns the 'Heavy Lifter' model for that provider.
    """
    if not requested_model or "/" not in requested_model:
        print(
            f"  [Log] No provider prefix found in '{requested_model}'. Using default."
        )
        return DELEGATED_DEEP_SEARCH_MAP["default"]

    # Split ONLY on the first slash.
    # "together-ai/Qwen/Qwen..." -> ["together-ai", "Qwen/Qwen..."]
    provider = requested_model.split("/", 1)[0]

    # Debug print to show what was parsed
    print(f"  [Log] Detected Provider: '{provider}'")

    # Return mapped model or fallback if provider is unknown
    return DELEGATED_DEEP_SEARCH_MAP.get(provider, DELEGATED_DEEP_SEARCH_MAP["default"])
