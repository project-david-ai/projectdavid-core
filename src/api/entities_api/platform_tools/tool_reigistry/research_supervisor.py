from entities_api.platform_tools.definitions.delegation.delegate_research_task import (
    delegate_research_task,
)
from entities_api.platform_tools.definitions.scratch_pad.append_scratchpad import (
    append_scratchpad,
)
from entities_api.platform_tools.definitions.scratch_pad.read_scratchpad import (
    read_scratchpad,
)
from entities_api.platform_tools.definitions.scratch_pad.update_scratchpad import (
    update_scratchpad,
)

SUPERVISOR_TOOLS = [
    read_scratchpad,
    update_scratchpad,
    append_scratchpad,
    delegate_research_task,
]
