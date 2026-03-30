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
# --------------------------------------
# Dataset ID: ds_XZrxCs7Imo0v3VRBLCeNCA
# --------------------------------------
retrieve_data_set = client.datasets.retrieve(dataset_id="ds_XZrxCs7Imo0v3VRBLCeNCA")

print(retrieve_data_set)
