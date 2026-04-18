from entities_api.orchestration.instructions.definitions import (
    LEVEL_3_WEB_USE_INSTRUCTIONS,
)

# ---------------------------------------------------------------------------
# Base protocol keys — apply to EVERY assistant regardless of opted-in tools.
# Tool-specific instruction bundles live in TOOL_INSTRUCTION_MAP and are
# appended at build time based on the assistant's declared tools array.
# ---------------------------------------------------------------------------
L2_INSTRUCTIONS = [
    "TOOL_USAGE_PROTOCOL",
    "FUNCTION_CALL_FORMATTING",
    "FUNCTION_CALL_WRAPPING",
]

L3_INSTRUCTIONS = [
    "L3_IDENTITY",
    "L3_PLANNING_PROTOCOL",
    "TOOL_DECISION_PROTOCOL",
    "L3_PARALLEL_EXECUTION",
    "CITATION_PROTOCOL",
    "TOOL_USAGE_PROTOCOL",
    "FUNCTION_CALL_FORMATTING",
    "FUNCTION_CALL_WRAPPING",
    "L3_SYNTAX_ENFORCEMENT",
]

# ---------------------------------------------------------------------------
# Per-tool instruction bundles.
# Keys correspond to canonical platform tool `type` strings in PLATFORM_TOOL_MAP.
# Only appended to instruction_keys when the assistant has opted in by including
# {"type": "<key>"} in its tools array at creation/update time.
# ---------------------------------------------------------------------------
TOOL_INSTRUCTION_MAP = {
    "code_interpreter": ["CODE_INTERPRETER", "CODE_FILE_HANDLING"],
    "file_search": ["VECTOR_SEARCH_EXAMPLES"],
    "web_search": ["WEB_SEARCH_RULES"],
    "computer": [],  # no instruction bundle yet
}


LEVEL_4_SUPERVISOR_INSTRUCTIONS = [
    "L4_SUPERVISOR_IDENTITY",
    "L4_TRIAGE_PROTOCOL",
    "L4_PLANNING_PROTOCOL",
    "L4_TOOL_ORCHESTRATION_PROTOCOL",
    "L4_DELEGATION_PROTOCOL",
    "L4_SCRATCHPAD_MANAGEMENT_PROTOCOL",
    "L4_EXECUTION_LOOP",
    "L4_CITATION_INTEGRITY",
    "L4_ANTI_STALL",
    "L4_SUPERVISOR_MOMENTUM",
    "L4_URL_PROTOCOL",
    "L4_FINAL_SYNTHESIS_PROTOCOL",
    "TOOL_USAGE_PROTOCOL",
    "FUNCTION_CALL_FORMATTING",
    "FUNCTION_CALL_WRAPPING",
    "CITATION_PROTOCOL",
]


L4_RESEARCH_INSTRUCTIONS = [
    "L4_WORKER_IDENTITY",
    "L4_WORKER_SCRATCHPAD_PROTOCOL",
    "L4_TOOL_CHEATSHEET",
    "TOOL_STRATEGY",
    "DIRECT_URL_EXCEPTION",
    "L4_PARALLEL_EXECUTION",
    "L4_EXECUTION_ALGORITHM",
    "RICH_MEDIA_HANDLING",
    "L4_STOPPING_CRITERIA",
    "FUNCTION_CALL_FORMATTING",
    "FUNCTION_CALL_WRAPPING",
    "L4_WORKER_REPORTING_FORMAT",
]


L3_WEB_USE_INSTRUCTIONS = [
    "WEB_CORE_IDENTITY",
    "STOP_RULE",
    "RICH_MEDIA_HANDLING",
    "CITATION_PROTOCOL",
    "TOOL_STRATEGY",
    "BATCH_OPERATIONS",
    "CONTEXT_MANAGEMENT",
    "ERROR_HANDLING",
    "RESEARCH_QUALITY_PROTOCOL",
    "RESEARCH_PROGRESS_TRACKING",
    "POST_EXECUTION_PROTOCOL",
]


L4_SENIOR_ENGINEER_INSTRUCTIONS = [
    "SE_IDENTITY",
    "SE_TRIAGE_PROTOCOL",
    "SE_SCRATCHPAD_PROTOCOL",
    "SE_DELEGATION_PROTOCOL",
    "SE_PLATFORM_AWARENESS",
    "SE_COMMAND_STRATEGY",
    "SE_EXECUTION_LOOP",
    "SE_ANTI_STALL",
    "SE_EVIDENCE_INTEGRITY",
    "SE_FINAL_OUTPUT_PROTOCOL",
    "TOOL_USAGE_PROTOCOL",
    "FUNCTION_CALL_FORMATTING",
    "FUNCTION_CALL_WRAPPING",
]

L4_JUNIOR_ENGINEER_INSTRUCTIONS = [
    "JE_IDENTITY",
    "JE_COMMUNICATION_AND_SCRATCHPAD_PROTOCOL",
    "JE_EXECUTION_ALGORITHM",
    "JE_TOOL_USAGE",
    "JE_STOPPING_CRITERIA",
    "JE_EVIDENCE_INTEGRITY",
    "TOOL_USAGE_PROTOCOL",
    "FUNCTION_CALL_FORMATTING",
    "FUNCTION_CALL_WRAPPING",
]


NO_CORE_INSTRUCTIONS = ["NONE"]
