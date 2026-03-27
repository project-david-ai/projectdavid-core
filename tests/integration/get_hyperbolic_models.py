import os

import dotenv
from openai import OpenAI

dotenv.load_dotenv()

client = OpenAI(
    api_key=os.getenv("HYPERBOLIC_API_KEY"), base_url="https://api.hyperbolic.xyz/v1"
)

# Fetch the list of models
models = client.models.list()

# Print all model IDs
for model in models.data:
    print(model.id)
