import json

import requests

url = "http://localhost:8001/v1/chat/completions"
headers = {"Content-Type": "application/json"}
payload = {
    "model": "david-ft",
    "messages": [
        {
            "role": "user",
            "content": "Confirm your identity. Are you the model fine-tuned on London-to-Sydney flight data?",
        }
    ],
    "temperature": 0,
}

print("🤔 Querying the fine-tuned brain...")
response = requests.post(url, headers=headers, json=payload)

if response.status_code == 200:
    content = response.json()["choices"][0]["message"]["content"]
    print(f"\n✨ [david-ft]: {content}")
else:
    print(f"❌ Error: {response.status_code}")
    print(response.text)
