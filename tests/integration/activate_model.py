import os
import time

from dotenv import load_dotenv
from projectdavid import Entity

# Load environment variables
load_dotenv(".tests.env")


# -------------------------------------------
# This is provided by the user after
# a successful fine tuning run
# ---------------------------------------------
FINE_TUNED_MODEL_ID = "ftm_G05BERHAEvSRr2KTyUqWIJ"


# ----------------------------------------
# Fine tuned models are user scoped
# ------------------------------------------

client = Entity(
    api_key=os.getenv("DEV_PROJECT_DAVID_CORE_TEST_USER_KEY"),
)

model = client.models.retrieve(FINE_TUNED_MODEL_ID)
print(model)

# ----------------------------------------
# Model activations are admin scoped
# ------------------------------------------
client = Entity(
    api_key=os.getenv("DEV_PROJECT_DAVID_CORE_TEST_USER_KEY"),
)


def run_activation():
    print(f"🎯 Activating model: {FINE_TUNED_MODEL_ID}")
    result = client.models.activate(FINE_TUNED_MODEL_ID)
    print(f"✅ Success: {result.activated}")


run_activation()
