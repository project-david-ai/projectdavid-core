## src/api/training/constants/models.py
SUPPORTED_BASE_MODELS = [
    {
        "id": "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        "name": "Llama 3.2 (1B) - 4bit",
        "family": "llama",
        "parameter_count": "1B",
    },
    {
        "id": "unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit",
        "name": "Qwen 2.5 (1.5B) - 4bit",
        "family": "qwen",
        "parameter_count": "1.5B",
    },
]
