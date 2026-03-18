import os

from dotenv import load_dotenv
from projectdavid import Entity

load_dotenv()

client = Entity(
    base_url=os.getenv("PROJECT_DAVID_PLATFORM_BASE_URL"),
    api_key=os.getenv("DEV_PROJECT_DAVID_CORE_ADMIN_KEY"),
)

user = client.users.create_user(
    name="Sam Flynn",
    email="sam@encom.com",
)

print(user)
