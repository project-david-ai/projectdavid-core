import os

from config_orc_fc import config
from dotenv import load_dotenv
from projectdavid import Entity

# ------------------------------------------------------------------
# 0.  SDK init + env
# ------------------------------------------------------------------
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent / ".tests.env"

load_dotenv(ENV_FILE, override=True)

client = Entity(
    base_url=os.getenv("BASE_URL", "http://localhost:80"),
    api_key=os.getenv("DEV_PROJECT_DAVID_CORE_TEST_USER_KEY"),
)

update_assistant = client.assistants.update_assistant(
    assistant_id=os.getenv("ASSISTANT_ID"),
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
