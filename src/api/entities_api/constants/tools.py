from entities_api.platform_tools.definitions.web_search.perform_web_search import (
    perform_web_search,
)

# --- Web Tools Imports ---
from entities_api.platform_tools.definitions.web_search.read_web_page import (
    read_web_page,
)
from entities_api.platform_tools.definitions.web_search.scroll_web_page import (
    scroll_web_page,
)
from entities_api.platform_tools.definitions.web_search.search_web_page import (
    search_web_page,
)

from src.api.entities_api.platform_tools.definitions.code_interpreter import (
    code_interpreter,
)
from src.api.entities_api.platform_tools.definitions.computer.computer import computer
from src.api.entities_api.platform_tools.definitions.file_search.file_search import (
    file_search,
)
from src.api.entities_api.platform_tools.definitions.scratch_pad.append_scratchpad import (
    append_scratchpad,
)
from src.api.entities_api.platform_tools.definitions.scratch_pad.read_scratchpad import (
    read_scratchpad,
)
from src.api.entities_api.platform_tools.definitions.scratch_pad.update_scratchpad import (
    update_scratchpad,
)

# Group them in the efficient "L3 Strategy" order:
# 1. Read (Get Context) -> 2. Search (Target Data) -> 3. Scroll (Fallback)
WEB_SEARCH_TOOLS = [read_web_page, search_web_page, scroll_web_page, perform_web_search]

SCRATCHPAD_TOOLS = [
    read_scratchpad,
    update_scratchpad,
    append_scratchpad,
]


PLATFORM_TOOL_MAP = {
    "code_interpreter": code_interpreter,
    "computer": computer,
    "file_search": file_search,
    "web_search": WEB_SEARCH_TOOLS,
    "scratchpad": SCRATCHPAD_TOOLS,
}
