import httpx

VLLM_URL = "http://localhost:8001/v1/chat/completions"
MODEL_ID = "ftm_G05BERHAEvSRr2KTyUqWIJ"

history = []

print(f"🤖 Chatting with {MODEL_ID} — type 'quit' to exit\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ("quit", "exit"):
        break
    if not user_input:
        continue

    history.append({"role": "user", "content": user_input})

    response = httpx.post(
        VLLM_URL,
        json={
            "model": MODEL_ID,
            "messages": history,
            "max_tokens": 200,
        },
        timeout=30.0,
    )
    response.raise_for_status()

    reply = response.json()["choices"][0]["message"]["content"]
    history.append({"role": "assistant", "content": reply})

    print(f"Model: {reply}\n")
