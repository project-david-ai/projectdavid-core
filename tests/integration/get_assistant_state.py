import os

from config_orc_fc import config
from dotenv import load_dotenv
from projectdavid import Entity

# ------------------------------------------------------------------
# 0.  SDK init + env
# ------------------------------------------------------------------
load_dotenv(".tests.env")

client = Entity(
    base_url=os.getenv("PROJECT_DAVID_PLATFORM_BASE_URL"),
    api_key=os.getenv("DEV_PROJECT_DAVID_CORE_TEST_USER_KEY"),
)

assistant = client.assistants.update_assistant(
    assistant_id=os.getenv("DEV_PROJECT_DAVID_ASSISTANT"),
)


print(f"Agent Mode:{assistant.agent_mode}")
print(f"Telemetry: {assistant.decision_telemetry}")
print(f"Web access:{assistant.web_access}")
print(f"Deep Research:{assistant.deep_research}")
print(assistant.id)
