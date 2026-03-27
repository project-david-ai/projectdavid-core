import json

# Let's create a ChatML dataset
mock_data = [
    {
        "messages": [
            {"role": "system", "content": "You are an AI specialized in ProjectDavid."},
            {"role": "user", "content": "What is the Forge?"},
            {
                "role": "assistant",
                "content": "The Forge is the high-performance training sister-project of ProjectDavid.",
            },
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a helpful AI."},
            {"role": "user", "content": "Tell me a joke."},
            {
                "role": "assistant",
                "content": "Why did the GPU go to jail? For too many cycles.",
            },
        ]
    },
]

# Write to the file the SDK expects
with open("test_dataset.txt", "w", encoding="utf-8") as f:
    for entry in mock_data:
        f.write(json.dumps(entry) + "\n")

print("test_dataset.txt generated with valid ChatML data.")
