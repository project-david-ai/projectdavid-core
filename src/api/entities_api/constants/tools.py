from entities_api.platform_tools.definitions.web_search.perform_web_search import \
    perform_web_search
# --- Web Tools Imports ---
from entities_api.platform_tools.definitions.web_search.read_web_page import \
    read_web_page
from entities_api.platform_tools.definitions.web_search.scroll_web_page import \
    scroll_web_page
from entities_api.platform_tools.definitions.web_search.search_web_page import \
    search_web_page

from src.api.entities_api.platform_tools.definitions.code_interpreter import \
    code_interpreter
from src.api.entities_api.platform_tools.definitions.computer.computer import \
    computer
from src.api.entities_api.platform_tools.definitions.file_search.file_search import \
    file_search

# Group them in the efficient "L3 Strategy" order:
# 1. Read (Get Context) -> 2. Search (Target Data) -> 3. Scroll (Fallback)
WEB_SEARCH_TOOLS = [read_web_page, search_web_page, scroll_web_page, perform_web_search]

PLATFORM_TOOL_MAP = {
    "code_interpreter": code_interpreter,
    "computer": computer,
    "file_search": file_search,
    "web_search": WEB_SEARCH_TOOLS,
}
