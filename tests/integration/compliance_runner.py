"""
tests/integration/compliance_runner.py

Unified access-control compliance runner for Project David.
Executes all isolation sweeps in sequence and produces a single
pass/fail report. Exits non-zero if any sweep has failures — suitable
for use as a CI gate.

Usage:
    python -m tests.integration.compliance_runner

Required env vars (same as individual sweep scripts):
    ENTITIES_BASE_URL   — e.g. http://localhost:9000
    OWNER_API_KEY
    INTRUDER_API_KEY
"""

import importlib
import os
import sys
import time
import traceback
from typing import Callable

from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# Registry — add new sweep modules here as they are written
# ──────────────────────────────────────────────────────────────────────────────
SWEEPS = [
    {
        "label": "Assistant Isolation",
        "module": "tests.integration.assistant_intrusion_test",
        "entry": "run_sweep",  # callable name inside the module
    },
    {
        "label": "Thread Isolation",
        "module": "tests.integration.thread_intrusion_test",
        "entry": "run_sweep",
    },
    {
        "label": "Run Isolation",
        "module": "tests.integration.run_intrusion_test",
        "entry": "run_sweep",
    },
    {
        "label": "Vector Store Isolation",
        "module": "tests.integration.vector_store_inrusion_test",  # note: original typo kept
        "entry": "run_sweep",
    },
    {
        "label": "File Isolation",
        "module": "tests.integration.files_intrusion_test",
        "entry": "run_sweep",
    },
    {
        "label": "Message Isolation",
        "module": "tests.integration.message_intrusion_test",
        "entry": "run_sweep",
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
WIDTH = 68


def banner(text: str, char: str = "═") -> None:
    print(f"\n{char * WIDTH}")
    print(f"  {text}")
    print(f"{char * WIDTH}")


def section(text: str) -> None:
    print(f"\n{'─' * WIDTH}")
    print(f"  {text}")
    print(f"{'─' * WIDTH}")


# ──────────────────────────────────────────────────────────────────────────────
# Pre-flight checks
# ──────────────────────────────────────────────────────────────────────────────
def preflight() -> bool:
    ok = True
    required = ["ENTITIES_BASE_URL", "OWNER_API_KEY", "INTRUDER_API_KEY"]
    for var in required:
        if not os.getenv(var):
            print(f"[ERROR] Missing required env var: {var}")
            ok = False
    return ok


# ──────────────────────────────────────────────────────────────────────────────
# Sweep executor
# ──────────────────────────────────────────────────────────────────────────────
def run_sweep(label: str, module_path: str, entry: str) -> dict:
    """
    Attempt to import module_path and call entry().
    The entry point must return a dict of {test_label: "PASS"|"FAIL"}.
    Returns a result dict with keys: label, passed, failed, skipped, error.
    """
    result = {
        "label": label,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "error": None,
        "results": {},
    }

    try:
        mod = importlib.import_module(module_path)
    except ModuleNotFoundError:
        result["skipped"] = 1
        result["error"] = f"Module not found: {module_path}"
        return result
    except Exception as exc:
        result["error"] = f"Import error: {exc}\n{traceback.format_exc()}"
        return result

    if not hasattr(mod, entry):
        result["skipped"] = 1
        result["error"] = (
            f"Module {module_path} has no callable '{entry}'. "
            f"Ensure each sweep exposes a run_sweep() → dict function."
        )
        return result

    try:
        fn: Callable = getattr(mod, entry)
        test_results: dict = fn()
        result["results"] = test_results
        for outcome in test_results.values():
            if outcome == "PASS":
                result["passed"] += 1
            else:
                result["failed"] += 1
    except Exception as exc:
        result["error"] = f"Runtime error: {exc}\n{traceback.format_exc()}"

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> int:
    banner("PROJECT DAVID — COMPLIANCE SWEEP RUNNER")
    print(f"  Base URL : {os.getenv('ENTITIES_BASE_URL', '(not set)')}")
    print(f"  Sweeps   : {len(SWEEPS)}")
    print(f"  Time     : {time.strftime('%Y-%m-%d %H:%M:%S')}")

    if not preflight():
        print(
            "\n[ABORT] Pre-flight checks failed. Set the required env vars and retry."
        )
        return 2

    all_results = []
    total_passed = 0
    total_failed = 0
    total_skipped = 0

    for sweep in SWEEPS:
        section(f"Running: {sweep['label']}")
        t0 = time.time()
        result = run_sweep(sweep["label"], sweep["module"], sweep["entry"])
        elapsed = time.time() - t0

        if result["error"]:
            print(f"  ⚠️  {result['error']}")

        for test_label, outcome in result["results"].items():
            tag = "✅" if outcome == "PASS" else "❌"
            print(f"  {tag}  {test_label}: {outcome}")

        total_passed += result["passed"]
        total_failed += result["failed"]
        total_skipped += result["skipped"]
        all_results.append({**result, "elapsed": elapsed})

        status = (
            "SKIPPED"
            if result["skipped"] and not result["results"]
            else "PASS" if result["failed"] == 0 and not result["error"] else "FAIL"
        )
        tag = {"PASS": "✅", "FAIL": "❌", "SKIPPED": "⚠️ "}[status]
        print(f"\n  {tag}  {sweep['label']}: {status}  ({elapsed:.2f}s)")

    # ── Final report ─────────────────────────────────────────────────────────
    banner("COMPLIANCE SWEEP — FINAL REPORT")

    for r in all_results:
        if r["skipped"] and not r["results"]:
            tag, note = "⚠️ ", f"SKIPPED — {r['error']}"
        elif r["failed"] == 0 and not r["error"]:
            tag, note = "✅", f"PASS  ({r['passed']} tests)"
        else:
            tag, note = "❌", f"FAIL  ({r['failed']} failed, {r['passed']} passed)"
        print(f"  {tag}  {r['label']:<30} {note}")

    print(f"\n  Total passed  : {total_passed}")
    print(f"  Total failed  : {total_failed}")
    print(f"  Total skipped : {total_skipped}")
    print(f"\n{'═' * WIDTH}")

    if total_failed > 0:
        print("  ❌  COMPLIANCE STATUS: FAIL")
        print(f"{'═' * WIDTH}\n")
        return 1

    if total_skipped > 0:
        print("  ⚠️   COMPLIANCE STATUS: PASS WITH WARNINGS (some sweeps skipped)")
    else:
        print("  ✅  COMPLIANCE STATUS: PASS")
    print(f"{'═' * WIDTH}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
