from src.api.entities_api.orchestration.instructions.agentic_instructions import (
    LEVEL_3_INSTRUCTIONS,
)
from src.api.entities_api.orchestration.instructions.deep_research_instructions import (
    LEVEL_4_SUPERVISOR_INSTRUCTIONS,
    RESEARCH_WORKERS_INSTRUCTIONS,
)
from src.api.entities_api.orchestration.instructions.general_instructions import (
    GENERAL_INSTRUCTIONS,
)
from src.api.entities_api.orchestration.instructions.network_engineering_instructions import (
    JUNIOR_ENGINEER_INSTRUCTIONS,
    SENIOR_ENGINEER_INSTRUCTIONS,
)
from src.api.entities_api.orchestration.instructions.web_use_instructions import (
    LEVEL_3_WEB_USE_INSTRUCTIONS,
)

CORE_INSTRUCTIONS = (
    GENERAL_INSTRUCTIONS
    | LEVEL_3_INSTRUCTIONS
    | LEVEL_3_WEB_USE_INSTRUCTIONS
    | RESEARCH_WORKERS_INSTRUCTIONS
    | LEVEL_4_SUPERVISOR_INSTRUCTIONS
    | SENIOR_ENGINEER_INSTRUCTIONS
    | JUNIOR_ENGINEER_INSTRUCTIONS
)
