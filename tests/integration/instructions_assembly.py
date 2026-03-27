from entities_api.orchestration.instructions.assembler import assemble_instructions

instructions = assemble_instructions(
    include_keys=[
        # "TOOL_USAGE_PROTOCOL",
        # "FUNCTION_CALL_FORMATTING",
        # "FUNCTION_CALL_WRAPPING",
        "DEVELOPER_INSTRUCTIONS"
        # Add "validations" or "error_handling" here if globally required
    ]
)
print(instructions)
