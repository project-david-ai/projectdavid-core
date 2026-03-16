# src/api/entities_api/orchestration/mixins/create_supervisor.py
from entities_api.platform_tools.tool_reigistry.research_supervisor import \
    SUPERVISOR_TOOLS
from projectdavid import Entity

from src.api.entities_api.orchestration.instructions.assembler import \
    assemble_instructions
from src.api.entities_api.orchestration.instructions.include_lists import \
    LEVEL_4_SUPERVISOR_INSTRUCTIONS


def create_supervisor_assistant(client: Entity):
    """
    Creates the 'Brain' of the operation.
    """

    supervisor_system_prompt = assemble_instructions(include_keys=LEVEL_4_SUPERVISOR_INSTRUCTIONS)

    supervisor = client.assistants.create_assistant(
        name="Deep Research Supervisor",
        model="gpt-4o",  # Needs a smart model to plan
        instructions=supervisor_system_prompt,  # <--- Level 4 Instructions
        tools=SUPERVISOR_TOOLS,
        # ^^^ This list contains:
        # 1. delegate_research_task
        # 2. read_scratchpad
        # 3. update_scratchpad
        # 4. append_scratchpad
        # IT DOES NOT CONTAIN: perform_web_search, read_web_page, etc.
    )
    return supervisor
