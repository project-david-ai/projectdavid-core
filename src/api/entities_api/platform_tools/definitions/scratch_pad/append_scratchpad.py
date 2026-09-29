append_scratchpad = {
    "type": "function",
    "function": {
        "name": "append_scratchpad",
        "description": (
            "Appends an entry to the scratchpad without replacing its working "
            "content. Use this to record findings, facts, URLs, numbers, "
            "progress updates, or other shared information."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "note": {
                    "type": "string",
                    "description": "The text to append as a new scratchpad entry.",
                }
            },
            "required": ["note"],
            "additionalProperties": False,
        },
    },
}
