# src/api/entities_api/constants/junior_network_engineer.py
import json

from src.api.entities_api.platform_tools.definitions.batfish.run_batfish_tool import \
    BATFISH_TOOLS_LIST
from src.api.entities_api.platform_tools.definitions.scratch_pad.append_scratchpad import \
    append_scratchpad

# ============================
# Junior Network Engineer
# Tools Array
# ============================

JUNIOR_ENGINEER_TOOLS = [
    *BATFISH_TOOLS_LIST,
    append_scratchpad,
]
