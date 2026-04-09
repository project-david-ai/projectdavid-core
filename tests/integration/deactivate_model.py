import os

from dotenv import load_dotenv
from projectdavid import Entity

load_dotenv(".tests.env")

client = Entity(
    api_key=os.getenv("DEV_PROJECT_DAVID_CORE_ADMIN_KEY"),
    base_url="http://localhost:9000",
)

print("🔄 Deactivating all fine-tuned models...")
deactivate = client.models.deactivate_all()
print(deactivate)
