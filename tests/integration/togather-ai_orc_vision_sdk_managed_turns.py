"""
vLLM Vision Inference Test
---------------------------------------------------
Tests multimodal (image + text) streaming via vLLM using
the unified SDK stream. Sends two Picsum images alongside
a text prompt and streams the model's visual response.
"""

import base64
import json
import os
import time

from dotenv import load_dotenv
from projectdavid import (ContentEvent, DecisionEvent, Entity, ReasoningEvent,
                          ToolCallRequestEvent)

load_dotenv()

# ------------------------------------------------------------------
# ANSI Colors
# ------------------------------------------------------------------
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
GREY = "\033[90m"
MAGENTA = "\033[95m"
RESET = "\033[0m"

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
BASE_URL = os.getenv("BASE_URL", "http://localhost:9000")
API_KEY = os.getenv("ENTITIES_API_KEY")

MODEL_ID = "together-ai/Qwen/Qwen3-VL-8B-Instruct"


VISION_PROMPT = "What are the differences between these two images? Please describe them in detail."


# ------------------------------------------------------------------
# Optional: encode a local image as base64
# ------------------------------------------------------------------
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# base64_image = encode_image("local_chart.png")

# ------------------------------------------------------------------
# Multimodal payload — OpenAI spec format
# MessagesClient handles download, upload to Samba, and DB storage
# ------------------------------------------------------------------
payload_content = [
    {
        "type": "text",
        "text": VISION_PROMPT,
    },
    {"type": "image_url", "image_url": {"url": "https://picsum.photos/id/1015/800/600"}},
    {"type": "image_url", "image_url": {"url": "https://picsum.photos/id/1016/800/600"}},
    # Uncomment to also send a local base64 image:
    # {
    #     "type": "image_url",
    #     "image_url": {
    #         "url": f"data:image/png;base64,{base64_image}"
    #     }
    # },
]

# ------------------------------------------------------------------
# SDK Init
# ------------------------------------------------------------------
client = Entity(base_url=BASE_URL, api_key=API_KEY)

# ------------------------------------------------------------------
# Thread + Message Setup
# ------------------------------------------------------------------
print(f"\n{CYAN}[▶] Setting up thread and uploading images...{RESET}")

assistant = client.assistants.create_assistant(
    name="Vision Test",
    instructions="You are a helpful assistant that can answer questions about images.",
)

thread = client.threads.create_thread()

message = client.messages.create_message(
    thread_id=thread.id,
    assistant_id=assistant.id,
    role="user",
    content=payload_content,
)

print(f"{GREEN}[✓] Assistant:   {assistant.id}{RESET}")
print(f"{GREEN}[✓] Thread:      {thread.id}{RESET}")
print(f"{GREEN}[✓] Message:     {message.id}{RESET}")
print(f"{GREEN}[✓] Attachments: {message.attachments}{RESET}")

# Sanity check — confirm formatted messages contain the hydrated image array
formatted = client.messages.get_formatted_messages(thread.id)
user_msgs = [m for m in formatted if m.get("role") == "user"]
last_user = user_msgs[-1] if user_msgs else {}
content = last_user.get("content", "")
if isinstance(content, list):
    image_blocks = [b for b in content if b.get("type") == "image"]
    print(f"{GREEN}[✓] Hydrated image blocks in formatted payload: {len(image_blocks)}{RESET}\n")
else:
    print(f"{YELLOW}[!] No image blocks found in formatted payload — check hydration{RESET}\n")

# ------------------------------------------------------------------
# Run + Stream Setup
# ------------------------------------------------------------------
run = client.runs.create_run(assistant_id=assistant.id, thread_id=thread.id)

stream = client.synchronous_inference_stream
stream.setup(
    thread_id=thread.id,
    assistant_id=assistant.id,
    message_id=message.id,
    run_id=run.id,
    api_key=os.getenv("TOGETHER_API_KEY"),
)

# ------------------------------------------------------------------
# Stream Loop
# ------------------------------------------------------------------
print(f"{CYAN}[▶] MODEL:    {MODEL_ID}{RESET}")
print(f"{CYAN}[▶] PROMPT:   {VISION_PROMPT}{RESET}\n")
print("-" * 60)

last_tick = time.perf_counter()
global_start = last_tick
content_started = False

try:
    for event in stream.stream_events(
        model=MODEL_ID,
    ):
        now = time.perf_counter()
        delta = now - last_tick
        last_tick = now

        color = {
            ContentEvent: GREEN,
            ToolCallRequestEvent: YELLOW,
            ReasoningEvent: CYAN,
            DecisionEvent: MAGENTA,
        }.get(type(event), RESET)

        if isinstance(event, ContentEvent):
            # Print header once at the start of a content run, then
            # stream tokens cleanly without repeating the prefix.
            if not content_started:
                print(f"\n{GREEN}[Assistant]{RESET} ", end="", flush=True)
                content_started = True
            print(event.content, end="", flush=True)

        else:
            # Close off any in-progress content block with a newline
            if content_started:
                print()
                content_started = False

            print(
                f"{GREY}[{delta:+.4f}s]{RESET} "
                f"| {color}{event.__class__.__name__:<25}{RESET} "
                f"| {json.dumps(event.to_dict())}"
            )

except Exception as e:
    if content_started:
        print()
    print(f"\n{RED}[ERROR] {e}{RESET}")

finally:
    if content_started:
        print()
    total = time.perf_counter() - global_start
    print(f"\n{YELLOW}{'=' * 50}")
    print(f"  TOTAL: {total:.4f}s")
    print(f"{'=' * 50}{RESET}\n")
