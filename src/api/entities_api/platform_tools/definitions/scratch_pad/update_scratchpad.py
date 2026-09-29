update_scratchpad = {
    "type": "function",
    "function": {
        "name": "update_scratchpad",
        "description": (
            "Replaces the scratchpad's working content. Use this to update, "
            "restructure, or rewrite the current shared plan or working state."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The new working content for the scratchpad.",
                }
            },
            "required": ["content"],
            "additionalProperties": False,
        },
    },
}
