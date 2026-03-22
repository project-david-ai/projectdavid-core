import os

from dotenv import load_dotenv
from projectdavid import Entity

load_dotenv()

client = Entity(
    base_url=os.getenv("PROJECT_DAVID_PLATFORM_BASE_URL"),
    api_key=os.getenv("DEV_PROJECT_DAVID_CORE_ADMIN_KEY"),
)

api_key = client.keys.create_key(
    user_id=os.getenv("DEV_PROJECT_DAVID_CORE_TEST_USER_KEY"),
)

print(api_key)
