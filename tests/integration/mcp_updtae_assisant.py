import os

from dotenv import load_dotenv
from projectdavid import Entity


load_dotenv(".tests.env")


base_url = os.getenv("PROJECT_DAVID_PLATFORM_BASE_URL")
api_key = os.getenv("DEV_PROJECT_DAVID_CORE_TEST_USER_KEY")
assistant_id = os.getenv("ASSISTANT_ID")

assert base_url, "PROJECT_DAVID_PLATFORM_BASE_URL is not set"
assert api_key, "DEV_PROJECT_DAVID_CORE_TEST_USER_KEY is not set"
assert assistant_id, "ASSISTANT_ID is not set"


client = Entity(
    base_url=base_url,
    api_key=api_key,
)


update_assistant = client.assistants.update_assistant(
    assistant_id=assistant_id, max_tokens=8192, max_turns=4
)
print(update_assistant.max_tokens)
print(update_assistant.max_turns)
