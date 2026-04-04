# src/api/entities_api/utilities/level2_utils.py


def format_level2_error(tool_name: str, error: Exception) -> str:
    """Standardized Level 2 Hint for both Platform and Consumer tools."""
    raw_error = str(error)
    return (
        f"Platform Error in '{tool_name}': {raw_error}. "
        "Instructions: Please analyze this error. If it is a syntax or logic error, "
        "correct your code/arguments and retry. If a file is missing, verify the filename."
    )
