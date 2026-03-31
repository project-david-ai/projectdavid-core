import os

from dotenv import load_dotenv
from projectdavid import Entity

# Load environment variables
load_dotenv(".tests.env")

# ---------------------------------------------
# This is provided by the user after
# a successful fine tuning run
# ---------------------------------------------
FINE_TUNED_MODEL_ID = "ftm_G05BERHAEvSRr2KTyUqWIJ"

# ----------------------------------------
# Confirm existence of fine-tuned adapters
# Fine-tuned models are user scoped
# ------------------------------------------
client = Entity(
    api_key=os.getenv("DEV_PROJECT_DAVID_CORE_TEST_USER_KEY"),
)

model = client.models.retrieve(FINE_TUNED_MODEL_ID)
print(model)

# ----------------------------------------
# Model activations are admin scoped
# ------------------------------------------
admin_client = Entity(
    api_key=os.getenv("DEV_PROJECT_DAVID_CORE_ADMIN_KEY"),
)

# -------------------------------------------------------------
# Register base model in the catalog (admin, idempotent)
# The server-side activation will resolve this HF path → bm_...
# before creating the InferenceDeployment record.
# -------------------------------------------------------------
registered = admin_client.registry.register(
    hf_model_id="unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit",
    name="Qwen2.5 1.5B Instruct (Unsloth 4bit)",
    family="qwen",
    parameter_count="1.5B",
)
print(f"📦 Base model registered: {registered.id}")

# -------------------------------------------------------------
# Activate the fine-tuned model using its ftm_... ID
# -------------------------------------------------------------
print(f"🎯 Activating model: {FINE_TUNED_MODEL_ID}")
result = admin_client.models.activate(FINE_TUNED_MODEL_ID)
print(f"✅ Result: {result}")
