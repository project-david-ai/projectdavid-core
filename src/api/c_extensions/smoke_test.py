"""
smoke_test.py — correctness check + timing benchmark for delta_normalizer_core

Run from the c_extensions/ directory after building:
    python setup.py build_ext --inplace
    python smoke_test.py
"""

import asyncio
import sys
import time

sys.path.insert(0, "entities_api")

import delta_normalizer_core as _C
from delta_normalizer import DeltaNormalizer

# ── Helpers ────────────────────────────────────────────────────────────────────


async def collect(tokens):
    events = []

    async def gen():
        for t in tokens:
            yield t

    async for ev in DeltaNormalizer.async_iter_deltas(gen(), run_id="test"):
        events.append(ev)
    return events


def make_openai_token(content):
    return {"choices": [{"delta": {"content": content}, "finish_reason": None}]}


# ── Correctness tests ──────────────────────────────────────────────────────────


async def test_plain_content():
    tokens = [make_openai_token(w) for w in ["Hello", " ", "world", "!"]]
    evs = await collect(tokens)
    text = "".join(e["content"] for e in evs if e["type"] == "content")
    assert text == "Hello world!", f"plain content failed: {text!r}"
    print("✓ plain content")


async def test_think_block():
    raw = "<think>I am thinking</think>final answer"
    tokens = [make_openai_token(c) for c in raw]
    evs = await collect(tokens)
    reasoning = "".join(e["content"] for e in evs if e["type"] == "reasoning")
    content = "".join(e["content"] for e in evs if e["type"] == "content")
    assert "I am thinking" in reasoning, f"reasoning missing: {reasoning!r}"
    assert "final answer" in content, f"content missing:   {content!r}"
    print("✓ think block")


async def test_tool_call_xml():
    payload = '{"name": "web_search", "arguments": {"query": "test"}}'
    raw = f"<tool_call>{payload}</tool_call>"
    tokens = [make_openai_token(c) for c in raw]
    evs = await collect(tokens)
    tc = [e for e in evs if e["type"] == "tool_call"]
    assert len(tc) == 1, f"expected 1 tool_call, got {len(tc)}"
    assert tc[0]["content"]["name"] == "web_search", tc[0]
    print("✓ tool_call_xml")


async def test_naked_json():
    payload = '{"name": "calc", "arguments": {"x": 1}}'
    tokens = [make_openai_token(c) for c in payload]
    evs = await collect(tokens)
    args = [e for e in evs if e["type"] == "call_arguments"]
    combined = "".join(
        (e["content"] if isinstance(e["content"], str) else "") for e in args
    )
    assert "{" in combined, f"naked json not captured: {combined!r}"
    print("✓ naked json")


async def test_fc_block():
    payload = '{"name": "search", "arguments": {}}'
    raw = f"<fc>{payload}</fc>"
    tokens = [make_openai_token(c) for c in raw]
    evs = await collect(tokens)
    tc = [e for e in evs if e["type"] == "tool_call"]
    assert len(tc) == 1, f"expected 1 tool_call from <fc>, got {len(tc)}"
    assert tc[0]["content"]["name"] == "search", tc[0]
    print("✓ fc block")


# ── Benchmark ──────────────────────────────────────────────────────────────────


async def benchmark():
    # Simulate ~2000 tokens of mixed content + think block + tool call
    chunk = (
        "Here is some analysis. " * 50
        + "<think>"
        + "deep reasoning " * 100
        + "</think>"
        + "Final answer: "
        + "<tool_call>"
        + '{"name": "execute", "arguments": {"cmd": "ls -la"}}'
        + "</tool_call>"
        + "Done."
    )
    tokens = [make_openai_token(c) for c in chunk]  # one char per token (worst case)

    N = 50
    start = time.perf_counter()
    for _ in range(N):
        await collect(tokens)
    elapsed = time.perf_counter() - start

    c_ext = "C extension" if _C else "pure Python"
    print(
        f"\n{'─'*50}\n"
        f"  Benchmark ({c_ext})\n"
        f"  {len(tokens)} tokens × {N} iterations\n"
        f"  Total : {elapsed:.3f}s\n"
        f"  Per run: {elapsed/N*1000:.2f}ms\n"
        f"{'─'*50}"
    )


# ── Main ───────────────────────────────────────────────────────────────────────


async def main():
    print("=== DeltaNormalizer C-extension smoke test ===\n")
    await test_plain_content()
    await test_think_block()
    await test_tool_call_xml()
    await test_naked_json()
    await test_fc_block()
    await benchmark()
    print("\nAll tests passed ✓")


asyncio.run(main())
