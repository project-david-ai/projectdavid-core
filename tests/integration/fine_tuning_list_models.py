import os
import time

from dotenv import load_dotenv
from projectdavid import Entity

# ------------------------------------------------------------------
# 0.  SDK init + env
# ------------------------------------------------------------------
load_dotenv(".tests.env")
client = Entity(api_key=os.getenv("DEV_PROJECT_DAVID_CORE_TEST_USER_KEY"))
print(client)

models = client.registry.list()
for m in models.items:
    print(f"{m.id}  {m.name}  {m.parameter_count}  {m.endpoint}")
