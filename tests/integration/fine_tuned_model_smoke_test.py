"""
test_inference.py

Direct test script for a Ray Serve / vLLM deployment.

Usage:
    # Basic smoke test (auto-discovers deployment name)
    python test_inference.py

    # Target a specific deployment
    python test_inference.py --deployment vllm_dep_c9NV0hTptZ0ObdSAts0uo6

    # Full options
    python test_inference.py \\
        --host http://localhost:8002 \\
        --deployment vllm_dep_c9NV0hTptZ0ObdSAts0uo6 \\
        --model unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit \\
        --max-tokens 256 \\
        --stream

Ports (from docker-compose):
    Ray Serve HTTP  → localhost:8002  (mapped from inference_worker:8000)
    Ray Dashboard   → localhost:8265
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Optional

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_HOST = "http://localhost:8002"
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9

TESTS = [
    {
        "name": "Smoke test — single turn",
        "messages": [
            {"role": "user", "content": "Reply with exactly three words: the sky is"},
        ],
    },
    {
        "name": "System prompt respected",
        "messages": [
            {
                "role": "system",
                "content": "You are a pirate. Always respond in pirate speak.",
            },
            {"role": "user", "content": "What is 2 + 2?"},
        ],
    },
    {
        "name": "Multi-turn context",
        "messages": [
            {"role": "user", "content": "My favourite colour is vermillion."},
            {
                "role": "assistant",
                "content": "That's a vivid red-orange shade — great choice!",
            },
            {"role": "user", "content": "What is my favourite colour?"},
        ],
    },
    {
        "name": "Reasoning / arithmetic",
        "messages": [
            {
                "role": "user",
                "content": "Think step by step. If a train travels 120 km in 90 minutes, what is its speed in km/h?",
            },
        ],
    },
    {
        "name": "Long-ish generation",
        "messages": [
            {
                "role": "user",
                "content": "Write a 3-sentence summary of the water cycle.",
            },
        ],
        "max_tokens": 512,
    },
]

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _post(url: str, payload: dict, timeout: int = 60) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Connection error: {e.reason}") from e


def _get(url: str, timeout: int = 10) -> dict:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        raise RuntimeError(str(e)) from e


# ---------------------------------------------------------------------------
# Deployment discovery
# ---------------------------------------------------------------------------


def discover_deployment(host: str) -> Optional[str]:
    """
    Hit the Ray Serve /-/routes endpoint to find the first vllm_ deployment.
    Returns the deployment name (without leading slash) or None.
    """
    try:
        routes = _get(f"{host}/-/routes")
        for route, name in routes.items():
            if route.lstrip("/").startswith("vllm_"):
                return route.lstrip("/")
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Single inference call
# ---------------------------------------------------------------------------


def run_inference(
    host: str,
    deployment: str,
    messages: list,
    model: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
) -> dict:
    url = f"{host}/{deployment}"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    t0 = time.perf_counter()
    result = _post(url, payload)
    elapsed = time.perf_counter() - t0
    result["_elapsed_s"] = round(elapsed, 3)
    return result


# ---------------------------------------------------------------------------
# Pretty printers
# ---------------------------------------------------------------------------

RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
DIM = "\033[2m"


def _color(text: str, code: str) -> str:
    return f"{code}{text}{RESET}"


def print_result(test_name: str, result: dict, idx: int, total: int) -> None:
    elapsed = result.get("_elapsed_s", "?")
    usage = result.get("usage", {})
    choices = result.get("choices", [])
    content = choices[0]["message"]["content"] if choices else "<no output>"

    prompt_tok = usage.get("prompt_tokens", "?")
    compl_tok = usage.get("completion_tokens", "?")
    total_tok = usage.get("total_tokens", "?")

    print(_color(f"\n[{idx}/{total}] {test_name}", BOLD))
    print(
        _color(
            f"  ⏱  {elapsed}s  |  tokens: {prompt_tok}p + {compl_tok}c = {total_tok}",
            DIM,
        )
    )
    print(_color("  Response:", CYAN))
    # Indent each line of the response
    for line in content.strip().splitlines():
        print(f"    {line}")


def print_summary(passed: int, failed: int, errors: list) -> None:
    total = passed + failed
    print("\n" + "─" * 60)
    if failed == 0:
        print(_color(f"✅  All {total} tests passed.", GREEN + BOLD))
    else:
        print(_color(f"❌  {failed}/{total} tests failed.", RED + BOLD))
        for name, err in errors:
            print(_color(f"    • {name}: {err}", RED))
    print("─" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Test a Ray Serve / vLLM deployment directly."
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Base URL of the Ray Serve HTTP server (default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--deployment",
        default=None,
        help="Deployment name, e.g. vllm_dep_c9NV0hTptZ0ObdSAts0uo6  (auto-discovered if omitted)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name to pass in the request body (defaults to deployment name)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Max tokens per response (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_TOP_P,
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Run a single custom prompt instead of the built-in test suite",
    )
    parser.add_argument(
        "--list-routes",
        action="store_true",
        help="Print available Ray Serve routes and exit",
    )
    args = parser.parse_args()

    # ── Route listing ──────────────────────────────────────────────────────
    if args.list_routes:
        print(f"Fetching routes from {args.host}/-/routes …")
        try:
            routes = _get(f"{args.host}/-/routes")
            if routes:
                for route, name in routes.items():
                    print(f"  {route}  →  {name}")
            else:
                print("  (no routes registered)")
        except Exception as e:
            print(_color(f"Error: {e}", RED))
            sys.exit(1)
        return

    # ── Deployment discovery ───────────────────────────────────────────────
    deployment = args.deployment
    if not deployment:
        print(
            f"No --deployment specified, auto-discovering from {args.host}/-/routes …"
        )
        deployment = discover_deployment(args.host)
        if deployment:
            print(_color(f"  Found: {deployment}", GREEN))
        else:
            print(
                _color(
                    "  Could not auto-discover a vllm_ deployment.\n"
                    "  Pass --deployment <name> explicitly, or check --list-routes.",
                    RED,
                )
            )
            sys.exit(1)

    model = args.model or deployment

    print(f"\n{_color('Sovereign Forge — vLLM Inference Test', BOLD)}")
    print(f"  Host:       {args.host}")
    print(f"  Deployment: {deployment}")
    print(f"  Model:      {model}")
    print(
        f"  Max tokens: {args.max_tokens}  temp={args.temperature}  top_p={args.top_p}"
    )

    # ── Single custom prompt mode ──────────────────────────────────────────
    if args.prompt:
        print(f"\n{_color('Custom prompt:', CYAN)} {args.prompt}\n")
        try:
            result = run_inference(
                host=args.host,
                deployment=deployment,
                messages=[{"role": "user", "content": args.prompt}],
                model=model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            print_result("Custom prompt", result, 1, 1)
        except Exception as e:
            print(_color(f"\nError: {e}", RED))
            sys.exit(1)
        return

    # ── Test suite ─────────────────────────────────────────────────────────
    print(f"\nRunning {len(TESTS)} tests …")

    passed = 0
    failed = 0
    errors = []

    for i, test in enumerate(TESTS, 1):
        max_tok = test.get("max_tokens", args.max_tokens)
        try:
            result = run_inference(
                host=args.host,
                deployment=deployment,
                messages=test["messages"],
                model=model,
                max_tokens=max_tok,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            print_result(test["name"], result, i, len(TESTS))
            passed += 1
        except Exception as e:
            print(_color(f"\n[{i}/{len(TESTS)}] {test['name']}", BOLD))
            print(_color(f"  ❌ {e}", RED))
            failed += 1
            errors.append((test["name"], str(e)))

    print_summary(passed, failed, errors)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
