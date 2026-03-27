import os

from dotenv import load_dotenv
from projectdavid import Entity

# Load environment variables
load_dotenv()

# Initialize the ProjectDavid SDK
client = Entity(
    base_url=os.getenv("PROJECT_DAVID_PLATFORM_BASE_URL"),
    api_key=os.getenv("DEV_PROJECT_DAVID_CORE_TEST_USER_KEY"),
)
activiate = client.models.activate_base(
    "unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit"
)


print(activiate)
