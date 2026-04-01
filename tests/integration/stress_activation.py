"""
stress_activation.py

Sovereign Forge — Activation / Deactivation Stress Test

Usage:
    python stress_activation.py
"""

import os
import time

from dotenv import load_dotenv
from projectdavid import Entity


load_dotenv(os.path.join(os.path.dirname(__file__), ".tests.env"))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FINE_TUNED_MODEL_ID = "ftm_G05BERHAEvSRr2KTyUqWIJ"
BASE_MODEL_HF_ID = "unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit"
RAPID_CYCLES = 3

# ---------------------------------------------------------------------------
# Clients — no explicit URLs, SDK resolves from environment
# ---------------------------------------------------------------------------

client = Entity(
    api_key=os.getenv("DEV_PROJECT_DAVID_CORE_TEST_USER_KEY"),
)

admin_client = Entity(
    api_key=os.getenv("DEV_PROJECT_DAVID_CORE_ADMIN_KEY"),
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

passed = 0
failed = 0


def run(label: str, fn):
    global passed, failed
    t0 = time.perf_counter()
    try:
        result = fn()
        elapsed = round(time.perf_counter() - t0, 3)
        print(f"  ✅  {label}  ({elapsed}s)")
        if result is not None:
            print(f"       ↳ {result}")
        passed += 1
        return result
    except Exception as exc:
        elapsed = round(time.perf_counter() - t0, 3)
        print(f"  ❌  {label}  ({elapsed}s)")
        print(f"       ↳ {exc}")
        failed += 1
        return None


def section(title: str):
    print(f"\n── {title} {'─' * max(0, 54 - len(title))}")


# ---------------------------------------------------------------------------
# 1. Confirm fine-tuned model exists (user-scoped)
# ---------------------------------------------------------------------------

section("Retrieve fine-tuned model")

run("Retrieve fine-tuned model", lambda: client.models.retrieve(FINE_TUNED_MODEL_ID))

# ---------------------------------------------------------------------------
# 2. Register base model (admin, idempotent)
#    Capture the bm_... ID for use in base model lifecycle steps.
# ---------------------------------------------------------------------------

section("Register base model")

registered = run(
    "Register base model (idempotent)",
    lambda: admin_client.registry.register(
        hf_model_id=BASE_MODEL_HF_ID,
        name="Qwen2.5 1.5B Instruct (Unsloth 4bit)",
        family="qwen",
        parameter_count="1.5B",
    ),
)

# registered.id is the bm_... catalog ID — no slashes, routes cleanly.
# Falls back to HF string if register failed, though that case should not occur.
BASE_MODEL_ID = registered.id if registered else BASE_MODEL_HF_ID

# ---------------------------------------------------------------------------
# 3. Basic activate / deactivate cycle
# ---------------------------------------------------------------------------

section("Basic activate / deactivate cycle")

run(
    "Activate fine-tuned model",
    lambda: admin_client.models.activate(FINE_TUNED_MODEL_ID),
)

run(
    "Deactivate fine-tuned model",
    lambda: admin_client.models.deactivate(FINE_TUNED_MODEL_ID),
)

# ---------------------------------------------------------------------------
# 4. deactivate_all → re-activate
# ---------------------------------------------------------------------------

section("deactivate_all → re-activate")

run(
    "Activate (before deactivate_all)",
    lambda: admin_client.models.activate(FINE_TUNED_MODEL_ID),
)

time.sleep(2)

run("deactivate_all", lambda: admin_client.models.deactivate_all())

run(
    "Re-activate after deactivate_all",
    lambda: admin_client.models.activate(FINE_TUNED_MODEL_ID),
)

run("Final deactivate_all", lambda: admin_client.models.deactivate_all())

# ---------------------------------------------------------------------------
# 5. Idempotency — double activate, double deactivate
# ---------------------------------------------------------------------------

section("Idempotency")

run("Activate #1", lambda: admin_client.models.activate(FINE_TUNED_MODEL_ID))

time.sleep(1)

run(
    "Activate #2 (same model — idempotent)",
    lambda: admin_client.models.activate(FINE_TUNED_MODEL_ID),
)

run("Deactivate #1", lambda: admin_client.models.deactivate(FINE_TUNED_MODEL_ID))

time.sleep(1)

run(
    "Deactivate #2 (already inactive)",
    lambda: admin_client.models.deactivate(FINE_TUNED_MODEL_ID),
)

# ---------------------------------------------------------------------------
# 6. Rapid cycles
# ---------------------------------------------------------------------------

section(f"Rapid cycles ({RAPID_CYCLES}x activate → deactivate)")

for i in range(1, RAPID_CYCLES + 1):
    run(
        f"Cycle {i} — activate",
        lambda: admin_client.models.activate(FINE_TUNED_MODEL_ID),
    )
    time.sleep(1)
    run(
        f"Cycle {i} — deactivate",
        lambda: admin_client.models.deactivate(FINE_TUNED_MODEL_ID),
    )
    time.sleep(1)

# ---------------------------------------------------------------------------
# 7. Base model lifecycle (no LoRA)
#    Uses bm_... ID from registry — not the raw HF path.
# ---------------------------------------------------------------------------

section("Base model lifecycle (no LoRA)")

run("Activate base model", lambda: admin_client.models.activate_base(BASE_MODEL_ID))

run("Deactivate base model", lambda: admin_client.models.deactivate_base(BASE_MODEL_ID))

# ---------------------------------------------------------------------------
# 8. Final cleanup
# ---------------------------------------------------------------------------

section("Final cleanup")

run("deactivate_all (leave cluster idle)", lambda: admin_client.models.deactivate_all())

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

total = passed + failed
print(f"\n{'═' * 60}")
if failed == 0:
    print(f"  ✅  All {total} steps passed.")
else:
    print(f"  ❌  {failed}/{total} steps failed.")
print(f"{'═' * 60}\n")
