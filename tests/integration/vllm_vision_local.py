"""
vLLM Vision Inference Test — Local Files
---------------------------------------------------
Tests the base64 local file path end-to-end.
Images are read from disk, encoded as data URIs, and
sent through the full pipeline:

  encode → SDK → Samba → file_id → Redis → hydrate → vLLM

No network image fetching — purely local files.
"""

import base64
import json
import os
import time

from config_orc_fc import config
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
ASSISTANT_ID = config.get("assistant_id")

MODEL_ID = "vllm/Qwen/Qwen2.5-VL-3B-Instruct"
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://vllm_server:8000")

VISION_PROMPT = (
    "These are two locally encoded images. "
    "Describe what you see in each one — include colours, shapes, and any text visible. "
    "Then clearly state what is different between them."
)


# ------------------------------------------------------------------
# Local image encoder
# ------------------------------------------------------------------
def encode_image(path: str) -> str:
    """Read a local image file and return a base64 data URI."""
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    mime_map = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
        "gif": "image/gif",
    }
    mime = mime_map.get(ext, "image/jpeg")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


# ------------------------------------------------------------------
# Locate test images
# These ship alongside this test script.
# Adjust paths if running from a different working directory.
# ------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_1 = os.path.join(SCRIPT_DIR, "test_local_1.jpg")
IMAGE_2 = os.path.join(SCRIPT_DIR, "test_local_2.jpg")

for path in (IMAGE_1, IMAGE_2):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Test image not found: {path}\n"
            f"Run the image generator script first, or place test_local_1.jpg "
            f"and test_local_2.jpg in the same directory as this script."
        )

print(f"\n{CYAN}[▶] Encoding local images...{RESET}")
b64_image_1 = encode_image(IMAGE_1)
b64_image_2 = encode_image(IMAGE_2)
print(f"{GREEN}[✓] Image 1: {IMAGE_1} ({len(b64_image_1)} chars){RESET}")
print(f"{GREEN}[✓] Image 2: {IMAGE_2} ({len(b64_image_2)} chars){RESET}")

# ------------------------------------------------------------------
# Multimodal payload — base64 data URIs, no network fetching
# ------------------------------------------------------------------
payload_content = [
    {
        "type": "text",
        "text": VISION_PROMPT,
    },
    {
        "type": "image_url",
        "image_url": {"url": b64_image_1},
    },
    {
        "type": "image_url",
        "image_url": {"url": b64_image_2},
    },
]

# ------------------------------------------------------------------
# SDK Init
# ------------------------------------------------------------------
client = Entity(base_url=BASE_URL, api_key=API_KEY)

# ------------------------------------------------------------------
# Thread + Message Setup
# ------------------------------------------------------------------
print(f"\n{CYAN}[▶] Setting up thread and uploading images to Samba...{RESET}")

thread = client.threads.create_thread()

message = client.messages.create_message(
    thread_id=thread.id,
    assistant_id=ASSISTANT_ID,
    role="user",
    content=payload_content,
)

print(f"{GREEN}[✓] Thread:      {thread.id}{RESET}")
print(f"{GREEN}[✓] Message:     {message.id}{RESET}")
print(f"{GREEN}[✓] Attachments: {message.attachments}{RESET}")

if len(message.attachments) != 2:
    print(
        f"{RED}[!] Expected 2 attachments, got {len(message.attachments)} — check SDK upload path{RESET}"
    )
else:
    print(f"{GREEN}[✓] Both local images uploaded and persisted as file_ids{RESET}")

# ------------------------------------------------------------------
# Hydration sanity check
# ------------------------------------------------------------------
formatted = client.messages.get_formatted_messages(thread.id)
user_msgs = [m for m in formatted if m.get("role") == "user"]
last_user = user_msgs[-1] if user_msgs else {}
content = last_user.get("content", "")

if isinstance(content, list):
    image_blocks = [b for b in content if b.get("type") == "image"]
    print(f"{GREEN}[✓] Hydrated image blocks: {len(image_blocks)}{RESET}\n")
else:
    print(f"{YELLOW}[!] Content is plain string — hydration did not produce image blocks{RESET}")
    print(f"{YELLOW}    content preview: {str(content)[:120]}{RESET}\n")

# ------------------------------------------------------------------
# Run + Stream Setup
# ------------------------------------------------------------------
run = client.runs.create_run(assistant_id=ASSISTANT_ID, thread_id=thread.id)

stream = client.synchronous_inference_stream
stream.setup(
    thread_id=thread.id,
    assistant_id=ASSISTANT_ID,
    message_id=message.id,
    run_id=run.id,
)

# ------------------------------------------------------------------
# Stream Loop
# ------------------------------------------------------------------
print(f"{CYAN}[▶] MODEL:    {MODEL_ID}{RESET}")
print(f"{CYAN}[▶] VLLM URL: {VLLM_BASE_URL}{RESET}")
print(f"{CYAN}[▶] PROMPT:   {VISION_PROMPT}{RESET}\n")
print("-" * 60)

last_tick = time.perf_counter()
global_start = last_tick
content_started = False

try:
    for event in stream.stream_events(
        model=MODEL_ID,
        meta_data={"vllm_base_url": VLLM_BASE_URL},
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
            if not content_started:
                print(f"\n{GREEN}[Assistant]{RESET} ", end="", flush=True)
                content_started = True
            print(event.content, end="", flush=True)

        else:
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
