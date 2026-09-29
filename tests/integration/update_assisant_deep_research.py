import os

from config_orc_fc import config
from dotenv import load_dotenv
from projectdavid import Entity

# ------------------------------------------------------------------
# 0.  SDK init + env
# ------------------------------------------------------------------
load_dotenv()

client = Entity(
    base_url=os.getenv("BASE_URL", "http://localhost:9000"),
    api_key=os.getenv("ENTITIES_API_KEY"),
)

update_assistant = client.assistants.update_assistant(
    assistant_id=config.get("assistant_id"),
    agent_mode=False,
    decision_telemetry=False,
    web_access=False,
    deep_research=False,
    engineer=False,
)


print(update_assistant.agent_mode)
print(update_assistant.decision_telemetry)
print(update_assistant.web_access)
print(update_assistant.deep_research)
print(update_assistant.id)
print(update_assistant.engineer)
