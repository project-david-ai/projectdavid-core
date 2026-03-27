"""
Project David — Raw Token Stream Profiler
==========================================
Approach 1: LOCAL tiny model (same family = same raw format)
Runs Qwen2.5-0.5B-Instruct locally. Fits in ~1GB RAM.
Profiles the ACTUAL raw character-by-character token stream before
any tool-call parsing happens.

Why this works for your chicken-egg problem:
  Qwen2.5-0.5B  ←→  same <tool_call> XML format ←→  Qwen3-235B
  The raw emission format is FAMILY-LEVEL, not size-level.
  Profile small, apply big.

Install:
    pip install transformers accelerate torch sentencepiece

Run:
    python profile_local_raw.py
"""

import json
import time
from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# ── Config ────────────────────────────────────────────────────────────────────

# Swap for any small Qwen variant. 0.5B fits on CPU. 1.5B is better quality.
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and country, e.g. London, UK",
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
}

MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant with access to tools."},
    {"role": "user", "content": "What is the weather like in London right now?"},
]

# ══════════════════════════════════════════════════════════════════════════════
# Load model
# ══════════════════════════════════════════════════════════════════════════════

print(f"\nLoading {MODEL_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,  # CPU-safe; use bfloat16 if you have GPU
    device_map="auto",
)
print("Model loaded.\n")

# ══════════════════════════════════════════════════════════════════════════════
# Build prompt — apply_chat_template with tools injects the tool schema
# ══════════════════════════════════════════════════════════════════════════════

prompt_ids = tokenizer.apply_chat_template(
    MESSAGES,
    tools=[TOOL_SCHEMA],
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
)

# Show what the fully-rendered prompt looks like (the system prompt injection)
rendered_prompt = tokenizer.apply_chat_template(
    MESSAGES,
    tools=[TOOL_SCHEMA],
    tokenize=False,
    add_generation_prompt=True,
)
print("═" * 60)
print("RENDERED PROMPT (what the model actually sees)")
print("═" * 60)
print(rendered_prompt)
print()

# ══════════════════════════════════════════════════════════════════════════════
# Stream raw tokens
# ══════════════════════════════════════════════════════════════════════════════

streamer = TextIteratorStreamer(
    tokenizer,
    skip_prompt=True,  # Don't re-emit the input
    skip_special_tokens=False,  # ← CRITICAL: keep <tool_call> etc visible
)

gen_kwargs = dict(
    input_ids=prompt_ids,
    streamer=streamer,
    max_new_tokens=300,
    do_sample=False,
    temperature=None,
    top_p=None,
)

print("═" * 60)
print("RAW TOKEN STREAM (skip_special_tokens=False)")
print("Each print = one decoded chunk as emitted by the model")
print("═" * 60)

# Run generation in background thread
thread = Thread(target=model.generate, kwargs=gen_kwargs)
thread.start()

raw_chunks = []
chunk_metadata = []

for i, chunk in enumerate(streamer):
    ts = time.perf_counter()
    raw_chunks.append(chunk)
    chunk_metadata.append({"index": i, "chunk": chunk, "ts": ts})
    print(f"[{i:03d}] {repr(chunk)}")

thread.join()

full_raw = "".join(raw_chunks)

print("\n" + "═" * 60)
print("FULL RAW OUTPUT (joined)")
print("═" * 60)
print(full_raw)

# ══════════════════════════════════════════════════════════════════════════════
# Structural analysis — what markers appear in the stream?
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("STRUCTURAL ANALYSIS")
print("═" * 60)

markers_of_interest = [
    "<tool_call>",
    "</tool_call>",
    "<|tool_call|>",
    "<|/tool_call|>",  # alternate Qwen format
    "✿FUNCTION✿",
    "✿RESULT✿",  # some Qwen variants
    "<function_calls>",
    "</function_calls>",  # Claude-style (won't appear but check)
    '{"name":',
    '"arguments"',
    "<think>",
    "</think>",  # Qwen3 reasoning tokens
]

print("\nMarkers found in output:")
for m in markers_of_interest:
    idx = full_raw.find(m)
    if idx != -1:
        context = full_raw[max(0, idx - 10) : idx + len(m) + 30]
        print(f"  ✓ {repr(m):30s}  at char {idx:4d}  context: {repr(context)}")
    else:
        print(f"  ✗ {repr(m)}")

# ══════════════════════════════════════════════════════════════════════════════
# Chunk boundary analysis — where do boundaries fall relative to markers?
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("CHUNK BOUNDARY vs MARKER ALIGNMENT")
print("Does <tool_call> arrive as one chunk or split across tokens?")
print("═" * 60)

# Find which chunk index contains key markers
running = ""
for meta in chunk_metadata:
    prev_len = len(running)
    running += meta["chunk"]
    for m in ["<tool_call>", "</tool_call>", "<think>", "</think>"]:
        pos = running.find(m)
        if pos != -1 and pos >= prev_len:
            print(
                f"  Marker {repr(m):20s} first appears in chunk [{meta['index']:03d}]"
            )
            print(f"    chunk content: {repr(meta['chunk'])}")

# ══════════════════════════════════════════════════════════════════════════════
# Save profiling data as JSON for Project David
# ══════════════════════════════════════════════════════════════════════════════

profile_data = {
    "model_id": MODEL_ID,
    "full_raw_output": full_raw,
    "chunk_count": len(raw_chunks),
    "chunks": [{"index": m["index"], "chunk": m["chunk"]} for m in chunk_metadata],
    "markers_found": {
        marker: full_raw.find(marker)
        for marker in markers_of_interest
        if full_raw.find(marker) != -1
    },
}

with open("raw_profile.json", "w") as f:
    json.dump(profile_data, f, indent=2)

print("\n✅ Profiling data saved to raw_profile.json")
print("   Feed this into Project David to calibrate your parser.\n")
