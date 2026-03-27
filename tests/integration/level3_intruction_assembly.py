from entities_api.orchestration.instructions.assembler import assemble_instructions
from entities_api.orchestration.instructions.include_lists import (
    L3_INSTRUCTIONS,
    NO_CORE_INSTRUCTIONS,
)


def assemble_core_instructions(include_keys, decision_telemetry=True):
    # 1. Work on a local copy to avoid mutating the imported list
    keys_to_process = list(include_keys)

    # 2. Handle the Telemetry switch
    if decision_telemetry:
        if "TOOL_DECISION_PROTOCOL" not in keys_to_process:
            keys_to_process.append("TOOL_DECISION_PROTOCOL")
    else:
        if "TOOL_DECISION_PROTOCOL" in keys_to_process:
            keys_to_process.remove("TOOL_DECISION_PROTOCOL")

    # 3. SORTING LOGIC: Re-order based on our Ratified Level 3 sequence.
    # Any keys not in our L3 list (like tool-specific commandments)
    # will appear after the core but before the final example.
    def get_rank(key):
        try:
            return L3_INSTRUCTIONS.index(key)
        except ValueError:
            # If a key is tool-specific (e.g., VECTOR_SEARCH),
            # place it just before the final example.
            return len(L3_INSTRUCTIONS) - 1.5

    sorted_keys = sorted(keys_to_process, key=get_rank)

    # 4. Final assembly
    return assemble_instructions(include_keys=sorted_keys)


# Execute
print(assemble_core_instructions(NO_CORE_INSTRUCTIONS))
