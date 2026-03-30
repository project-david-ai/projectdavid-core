""" """

import os
from dotenv import load_dotenv
from projectdavid import Entity

# ------------------------------------------------------------------
# 0.  SDK init + env
# ------------------------------------------------------------------
load_dotenv(".tests.env")

client = Entity(api_key=os.getenv("DEV_PROJECT_DAVID_CORE_TEST_USER_KEY"))
print(client)


# -------------------------------------
# Create fine tuning dataset
# - stages training data
# --------------------------------------

dataset = client.datasets.create(
    file_path="projectdavid_sdk_finetune.jsonl",
    name="Specialized Knowledge Base",
    fmt="jsonl",
)
print(f"📦 Dataset ID: {dataset.id}")
# --------------------------------------
# Dataset ID: ds_XZrxCs7Imo0v3VRBLCeNCA
# --------------------------------------
