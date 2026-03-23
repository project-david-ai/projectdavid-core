from src.api.entities_api.platform_tools.definitions.delegation.delegate_engineer_task import \
    delegate_engineer_task
from src.api.entities_api.platform_tools.definitions.scratch_pad.read_scratchpad import \
    read_scratchpad
from src.api.entities_api.platform_tools.definitions.scratch_pad.update_scratchpad import \
    update_scratchpad

# ============================
# Senior Network Engineer
# Tools Array
# Order mirrors execution flow:
#   1. Discover → 2. Resolve → 3. Plan → 4. Monitor → 5. Delegate
# ============================
SENIOR_ENGINEER_TOOLS = [
    update_scratchpad,  # 3. Planning — Senior OWNS the scratchpad: writes [INCIDENT], [PLAN], [FINDING], [TOMBSTONE]
    read_scratchpad,  # 4. Monitoring — reads Junior's appended ✅/🚩/⚠️ entries after each delegation returns
    delegate_engineer_task,  # 5. Delegation — dispatches a focused command set to the Junior Engineer
]
