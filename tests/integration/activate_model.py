import json
import os
import time

from dotenv import load_dotenv
from projectdavid import Entity

# Load environment variables
load_dotenv()

# Initialize the ProjectDavid SDK
client = Entity(
    base_url=os.getenv("PROJECT_DAVID_PLATFORM_BASE_URL"),
    api_key=os.getenv("DEV_PROJECT_DAVID_CORE_TEST_USER_KEY"),
)


# VALID_MODEL_ID = "ftm_mK9ZN21nxLH3jJNGcAMQjc"
VALID_MODEL_ID = "ftm_7i8THeyHtMvMk6Ns5TGxZ9"


def run_activation():
    print(f"🎯 Activating model: {VALID_MODEL_ID}")
    result = client.models.activate(VALID_MODEL_ID)
    print(f"✅ Success: {result.activated}")


run_activation()
