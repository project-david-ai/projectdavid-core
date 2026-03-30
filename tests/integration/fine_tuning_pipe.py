""" """

import os
from dotenv import load_dotenv
from projectdavid import Entity

# ------------------------------------------------------------------
# 0.  SDK init + env
# ------------------------------------------------------------------
load_dotenv()

client = Entity(api_key=os.getenv("DEV_PROJECT_DAVID_CORE_TEST_USER_KEY"))
print(client)


# -------------------------------------
# Create fine tuning dataset
# --------------------------------------

dataset = client.datasets.create(
    file_path="my_data.jsonl", name="Specialized Knowledge Base", fmt="jsonl"
)
print(f"📦 Dataset ID: {dataset.id}")
