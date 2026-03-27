"""
Project David — vLLM Raw Token Stream Profiler
===============================================
Targets /v1/completions (NOT /v1/chat/completions).
This bypasses vLLM's chat template + tool-call parser entirely.
You get the raw generated text exactly as the model emits it.

Usage:
------
1. Start vLLM server (in a separate terminal):
       vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8000

2. Run this script:
       python profile_vllm_raw.py

Swap MODEL_ID and rendered prompt to cycle through families.
"""

import json

import httpx

VLLM_BASE = "http://localhost:8000"

# ── Model profiles — swap these to cycle through families ─────────────────────

PROFILES = [
    {
        "name": "Qwen2.5-0.5B-Instruct",
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "prompt": (
            "<|im_start|>system\n"
            "You are a helpful assistant with access to tools.\n\n"
            "## Tools\n\n"
            '{"type":"function","function":{"name":"get_weather","description":"Get weather.","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}\n'
            "<|im_end|>\n"
            "<|im_start|>user\nWhat is the weather in London?\n<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
    },
    {
        "name": "Mistral-7B-Instruct-v0.3",
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "prompt": (
            "<s>[INST] You have access to the following tools:\n"
            '[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}]\n\n'
            "What is the weather in London? [/INST]"
        ),
    },
    {
        "name": "Llama-3.2-1B-Instruct",
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "prompt": (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You are a helpful assistant with access to tools.\n"
            'Tools: [{"name": "get_weather", "parameters": {"location": {"type": "string"}}}]'
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            "What is the weather in London?<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
    },
    # Add more families as needed:
    # Gemma3, Phi-3, DeepSeek-V3, Command-R...
]


# ══════════════════════════════════════════════════════════════════════════════
# Health check
# ══════════════════════════════════════════════════════════════════════════════


def check_server():
    try:
        r = httpx.get(f"{VLLM_BASE}/health", timeout=5)
        if r.status_code == 200:
            print("✅ vLLM server is up\n")
            return True
    except Exception:
        pass
    print("❌ vLLM server not reachable at", VLLM_BASE)
    print("   Start it with: vllm serve <model_id> --port 8000")
    return False


def list_loaded_models():
    try:
        r = httpx.get(f"{VLLM_BASE}/v1/models", timeout=5)
        models = r.json().get("data", [])
        print("Loaded models:")
        for m in models:
            print(f"  - {m['id']}")
        print()
        return [m["id"] for m in models]
    except Exception as e:
        print(f"Could not list models: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# Core profiler — hits /v1/completions (raw text, no chat wrapper)
# ══════════════════════════════════════════════════════════════════════════════


def profile_vllm_raw(profile: dict):
    name = profile["name"]
    model_id = profile["model_id"]
    prompt = profile["prompt"]

    print("═" * 60)
    print(f"PROFILING: {name}")
    print(f"Endpoint:  {VLLM_BASE}/v1/completions")
    print("═" * 60)
    print("\nRendered prompt sent to model:")
    print(prompt)
    print()

    payload = {
        "model": model_id,
        "prompt": prompt,
        "max_tokens": 300,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": False},
        # Ensure special tokens are included in output
        "skip_special_tokens": False,
    }

    raw_chunks = []
    chunk_index = 0

    print("RAW SSE STREAM (/v1/completions):\n")

    with httpx.Client(timeout=120) as http:
        with http.stream(
            "POST",
            f"{VLLM_BASE}/v1/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
        ) as resp:

            print(f"HTTP {resp.status_code}\n")

            if resp.status_code != 200:
                print(f"ERROR: {resp.read().decode()}")
                return

            for line in resp.iter_lines():
                if not line:
                    continue
                if not line.startswith("data:"):
                    print(f"SSE meta: {repr(line)}")
                    continue

                raw = line[5:].strip()
                if raw == "[DONE]":
                    print("\n[DONE]")
                    break

                try:
                    parsed = json.loads(raw)
                    # /v1/completions delta is in choices[0].text
                    text = parsed["choices"][0].get("text", "")
                    finish = parsed["choices"][0].get("finish_reason")

                    raw_chunks.append(text)
                    print(f"[{chunk_index:03d}] {repr(text)}", end="")
                    if finish:
                        print(f"  ← finish_reason={finish}", end="")
                    print()
                    chunk_index += 1

                except Exception as e:
                    print(f"  parse error: {e} | raw={repr(raw)}")

    full_output = "".join(raw_chunks)

    # ── Structural analysis ──────────────────────────────────────────────────

    print("\n" + "─" * 60)
    print("FULL OUTPUT:")
    print(full_output)

    print("\n" + "─" * 60)
    print("MARKER SCAN:")

    markers = [
        "<tool_call>",
        "</tool_call>",  # Qwen
        "[TOOL_CALLS]",  # Mistral
        "<|python_tag|>",  # Llama
        "```python",
        "```json",  # Gemma / Cohere
        "Action:",  # Cohere
        "<|tool▁calls▁begin|>",  # DeepSeek
        "<|tool_call|>",  # Phi
        "<think>",
        "</think>",  # Qwen3 reasoning
        '{"name":',
        '"arguments":',
        '"parameters":',
    ]

    for m in markers:
        idx = full_output.find(m)
        if idx != -1:
            ctx = full_output[max(0, idx - 5) : idx + len(m) + 30]
            print(f"  ✓ {repr(m):30s} at char {idx:4d} → {repr(ctx)}")
        else:
            print(f"  ✗ {repr(m)}")

    print("\n" + "─" * 60)
    print("CHUNK BOUNDARY ANALYSIS:")
    print("(Where do structural markers land relative to chunk edges?)\n")

    running = ""
    for i, chunk in enumerate(raw_chunks):
        prev = len(running)
        running += chunk
        for m in ["<tool_call>", "[TOOL_CALLS]", "<|python_tag|>", "<think>"]:
            pos = running.find(m)
            if pos != -1 and pos >= prev:
                print(
                    f"  Marker {repr(m)} first seen in chunk [{i:03d}]: {repr(chunk)}"
                )
                surrounding = raw_chunks[max(0, i - 1) : i + 3]
                print(f"  Context chunks: {surrounding}")

    # ── Save ────────────────────────────────────────────────────────────────

    out = {
        "model": name,
        "endpoint": "vllm_v1_completions",
        "full_output": full_output,
        "chunk_count": len(raw_chunks),
        "chunks": raw_chunks,
        "markers": {
            m: full_output.find(m) for m in markers if full_output.find(m) != -1
        },
    }

    fname = f"vllm_profile_{name.replace('/', '_').replace('-', '_')}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved: {fname}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Run
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not check_server():
        exit(1)

    loaded = list_loaded_models()

    for profile in PROFILES:
        # Only profile models that are actually loaded in the server
        if loaded and not any(profile["model_id"] in m for m in loaded):
            print(f"⚠️  Skipping {profile['name']} — not loaded in vLLM server")
            print(f"   Run: vllm serve {profile['model_id']} --port 8000\n")
            continue
        profile_vllm_raw(profile)

    print("✅ All vLLM profiles complete.")
