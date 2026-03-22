""" """

import os

from dotenv import load_dotenv
from entities_api.orchestration.instructions.assembler import \
    assemble_instructions
from projectdavid import Entity

# ------------------------------------------------------------------
# 0.  SDK init + env
# ------------------------------------------------------------------
load_dotenv()

client = Entity(
    base_url=os.getenv("BASE_URL", "http://localhost:80"),
    api_key=os.getenv("ENTITIES_API_KEY"),
)


tool_instructions = assemble_instructions(
    include_keys=[
        "TOOL_USAGE_PROTOCOL",
        "FUNCTION_CALL_FORMATTING",
        "FUNCTION_CALL_WRAPPING",
    ]
)


# -------------------------------------------
# create_assistant
# --------------------------------------------
assistant = client.assistants.create_assistant(
    name="Test Assistant",
    model="gpt-oss-120b",
    instructions="You are a helpful AI assistant, your name is Nexa.",
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_flight_times",
                "description": "Get flight times",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "departure": {"type": "string"},
                        "arrival": {"type": "string"},
                    },
                    "required": ["departure", "arrival"],
                },
            },
        },
    ],
)

print(assistant.id)
print(assistant.instructions)


retrieve_assistant = client.assistants.retrieve_assistant(assistant.id)
print(retrieve_assistant)
