"""
vLLM Vision Inference Test — Local Files
---------------------------------------------------
Tests the base64 local file path end-to-end.
Images are read from disk, resized to meet strict Qwen2-VL
pixel constraints, encoded as data URIs, and sent through the full pipeline.
"""

import base64
import json
import os
import time
import io
import math
from PIL import Image

from dotenv import load_dotenv
from projectdavid import (
    ContentEvent,
    DecisionEvent,
    Entity,
    ReasoningEvent,
    ToolCallRequestEvent,
)

load_dotenv("../../.tests.env")

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
BASE_URL = os.getenv("BASE_URL", "http://localhost:80")
API_KEY = os.getenv("DEV_PROJECT_DAVID_CORE_TEST_USER_KEY")

MODEL_ID = "vllm/Qwen/Qwen2.5-VL-3B-Instruct-AWQ"
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://inference_worker:8000")

VISION_PROMPT = (
    "These are two locally encoded images. "
    "Describe what you see in each one — include colours, shapes, and any text visible. "
    "Then clearly state what is different between them."
)


# ------------------------------------------------------------------
# Local image encoder with resizing bounds
# ------------------------------------------------------------------
def encode_image(path: str, min_pixels: int = 3136, max_pixels: int = 12544) -> str:
    """
    Read a local image file, resize it to fit within [min_pixels, max_pixels]
    maintaining aspect ratio, and return a base64 data URI.
    """
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    mime_map = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
        "gif": "image/gif",
    }
    mime = mime_map.get(ext, "image/jpeg")

    pil_format_map = {
        "image/jpeg": "JPEG",
        "image/png": "PNG",
        "image/webp": "WEBP",
        "image/gif": "GIF",
    }
    save_format = pil_format_map.get(mime, "JPEG")

    with Image.open(path) as img:
        # Convert to RGB if saving as JPEG to prevent alpha channel errors
        if save_format == "JPEG" and img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        width, height = img.size
        num_pixels = width * height

        # Calculate scaling factor if outside bounds
        scale = 1.0
        if num_pixels > max_pixels:
            scale = math.sqrt(max_pixels / num_pixels)
        elif num_pixels < min_pixels:
            scale = math.sqrt(min_pixels / num_pixels)

        if scale != 1.0:
            new_width = max(1, int(width * scale))
            new_height = max(1, int(height * scale))
            print(
                f"{YELLOW}[ℹ] Resizing {os.path.basename(path)} from {width}x{height} "
                f"to {new_width}x{new_height} (Pixels: {new_width * new_height}){RESET}"
            )
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        else:
            print(
                f"{GREY}[ℹ] {os.path.basename(path)} is {width}x{height} (Pixels: {num_pixels}) - No resize needed{RESET}"
            )

        # Save to memory buffer
        buffer = io.BytesIO()
        img.save(buffer, format=save_format)
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:{mime};base64,{b64}"


# ------------------------------------------------------------------
# Locate test images
# ------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_1 = os.path.join(SCRIPT_DIR, "test_local_1.jpg")
IMAGE_2 = os.path.join(SCRIPT_DIR, "test_local_2.jpg")

for path in (IMAGE_1, IMAGE_2):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Test image not found: {path}")

print(f"\n{CYAN}[▶] Encoding local images with Qwen2.5-VL bounds...{RESET}")
# Passing the exact limits from your activation settings
b64_image_1 = encode_image(IMAGE_1, min_pixels=3136, max_pixels=12544)
b64_image_2 = encode_image(IMAGE_2, min_pixels=3136, max_pixels=12544)

print(f"{GREEN}[✓] Image 1 Ready ({len(b64_image_1)} base64 chars){RESET}")
print(f"{GREEN}[✓] Image 2 Ready ({len(b64_image_2)} base64 chars){RESET}")

# ------------------------------------------------------------------
# Multimodal payload
# ------------------------------------------------------------------
payload_content = [
    {"type": "text", "text": VISION_PROMPT},
    {"type": "image_url", "image_url": {"url": b64_image_1}},
    {"type": "image_url", "image_url": {"url": b64_image_2}},
]

# ------------------------------------------------------------------
# SDK Init
# ------------------------------------------------------------------
client = Entity(base_url=BASE_URL, api_key=API_KEY)

print(f"\n{CYAN}[▶] Setting up thread and uploading images...{RESET}")
thread = client.threads.create_thread()

assistant = client.assistants.create_assistant(
    name="test_assistant",
    instructions="You are a helpful AI assistant, your name is Nexa.",
)

message = client.messages.create_message(
    thread_id=thread.id,
    assistant_id=assistant.id,
    role="user",
    content=payload_content,
)

print(f"{GREEN}[✓] Thread:      {thread.id}{RESET}")
print(f"{GREEN}[✓] Message:     {message.id}{RESET}")
print(f"{GREEN}[✓] Attachments: {message.attachments}{RESET}")

if len(message.attachments) != 2:
    print(f"{RED}[!] Expected 2 attachments, got {len(message.attachments)}{RESET}")
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
    print(
        f"{YELLOW}[!] Content is plain string — hydration did not produce image blocks{RESET}"
    )

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
