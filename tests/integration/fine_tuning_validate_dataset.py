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

# -----------------------------------------
# Validation time
# ------------------------------------------
import time

client.datasets.prepare("ds_XZrxCs7Imo0v3VRBLCeNCA")

# Poll until active
while True:
    ds = client.datasets.retrieve(dataset_id="ds_XZrxCs7Imo0v3VRBLCeNCA")
    print(f"Status: {ds.status}")
    if ds.status == "active":
        print(
            f"✅ Dataset ready — {ds.train_samples} train / {ds.eval_samples} eval samples"
        )
        break
    if ds.status == "failed":
        raise Exception(f"Dataset preparation failed: {ds}")
    time.sleep(3)


# -------------------------------------------------------
# Dispatch the training job
# ----------------------------------------------------
# Submit the job to the Cluster Mesh
job = client.training.create(
    dataset_id="ds_XZrxCs7Imo0v3VRBLCeNCA",
    base_model="unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit",
    framework="unsloth",
    config={"learning_rate": 2e-4, "num_train_epochs": 1, "lora_r": 16},
)
print(f"🔥 Job {job.id} dispatched to Node:")
